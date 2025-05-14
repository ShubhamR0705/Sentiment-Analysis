import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import mediapipe as mp
import librosa
import plotly.express as px
import pandas as pd
import io
import base64
from PIL import Image

# Import for comprehensive analysis
import whisper
from deepface import DeepFace
from transformers import pipeline
from moviepy import VideoFileClip

# Import for real-time emotion recognition
from tensorflow.keras.models import load_model

# Streamlit configuration
st.set_page_config(
    page_title="Emotion Analysis System",
    page_icon="😀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache the model loading to improve performance
@st.cache_resource
def load_models():
    models = {
        "whisper": whisper.load_model("base"),
        "speech_emotion": pipeline(
            "text-classification",
            model="bhadresh-savani/distilbert-base-uncased-emotion", return_all_scores=True
        ),
        "face_emotion": load_model('model_file_30epochs.h5'),
        "face_cascade": cv2.CascadeClassifier('haarcascade_frontalface_default.xml'),
        "mediapipe_face": mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        ),
        "mediapipe_face_mesh": mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    }
    return models

# Create a session state to track app state
if 'stream_active' not in st.session_state:
    st.session_state.stream_active = False
if 'emotion_history' not in st.session_state:
    st.session_state.emotion_history = []
if 'audio_buffer' not in st.session_state:
    st.session_state.audio_buffer = []

# Emotion labels for Keras model
EMOTION_LABELS = {
    0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy',
    4: 'Neutral', 5: 'Sad', 6: 'Surprise'
}

#########################################
# Enhanced Emotion Analyzer Class       #
#########################################

class EmotionAnalyzer:
    def __init__(self, models=None):
        # Initialize with pre-loaded models or load them
        if models:
            self.models = models
        else:
            self.models = load_models()
        
        # Initialize emotion history tracking
        self.emotion_history = []
        self.audio_chunks = []
        
    def extract_frames(self, video_path, output_folder=None, num_frames=10):
        """Extract evenly spaced frames from video using OpenCV."""
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, total_frames // num_frames)

        extracted_frames = []
        extracted_frame_images = []
        
        for i in range(num_frames):
            frame_pos = min(i * frame_interval, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            success, frame = cap.read()
            if success:
                if output_folder:
                    frame_path = os.path.join(output_folder, f"frame_{i}.jpg")
                    cv2.imwrite(frame_path, frame)
                    extracted_frames.append(frame_path)
                extracted_frame_images.append(frame)

        cap.release()
        return extracted_frames, extracted_frame_images

    def analyze_facial_emotions_deepface(self, video_path, use_mediapipe=True):
        """Analyze emotions from facial expressions using DeepFace and/or MediaPipe."""
        _, frames = self.extract_frames(video_path)
        emotions = []
        confidence_scores = []
        analyzed_frames = []

        # Use ThreadPoolExecutor for parallel processing of frames
        with ThreadPoolExecutor() as executor:
            if use_mediapipe:
                futures = [executor.submit(self._analyze_frame_mediapipe, frame) for frame in frames]
            else:
                futures = [executor.submit(self._analyze_frame_deepface, frame) for frame in frames]
            
            for i, future in enumerate(as_completed(futures)):
                emotion, confidence, frame = future.result()
                emotions.append(emotion)
                confidence_scores.append(confidence)
                analyzed_frames.append(frame)

        # Get the most common emotion
        if emotions:
            final_emotion = max(set(emotions), key=emotions.count)
            emotion_counts = dict(Counter(emotions))
            avg_confidence = sum(filter(None, confidence_scores)) / len(list(filter(None, confidence_scores))) if any(confidence_scores) else 0
        else:
            final_emotion = "No face detected"
            emotion_counts = {}
            avg_confidence = 0

        return final_emotion, emotion_counts, emotions, avg_confidence, analyzed_frames

    def _analyze_frame_deepface(self, frame):
        """Helper function to analyze a single frame with DeepFace."""
        try:
            # Save frame to temporary file for DeepFace
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp:
                cv2.imwrite(temp.name, frame)
                analysis = DeepFace.analyze(temp.name, actions=["emotion"], enforce_detection=False)
            
            # Clean up temp file
            os.unlink(temp.name)
            
            # Analysis may return a list; we assume the first result
            emotion = analysis[0]["dominant_emotion"]
            confidence = analysis[0]["emotion"][emotion]
            
            # Draw the emotion on the frame
            cv2.putText(frame, f"{emotion}: {confidence:.2f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            return emotion, confidence, frame
            
        except Exception as e:
            return "No face detected", None, frame

    def _analyze_frame_mediapipe(self, frame):
        """Helper function to analyze a single frame with MediaPipe and Keras model."""
        try:
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.models["mediapipe_face"].process(rgb_frame)
            
            if not results.detections:
                return "No face detected", None, frame
                
            # Get the first detected face
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            
            # Get image dimensions
            h, w, _ = frame.shape
            
            # Convert relative coordinates to absolute
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            # Ensure coordinates are within image bounds
            x = max(0, x)
            y = max(0, y)
            width = min(width, w - x)
            height = min(height, h - y)
            
            # Extract face ROI
            face_roi = frame[y:y+height, x:x+width]
            if face_roi.size == 0:
                return "No valid face", None, frame
                
            # Preprocess for emotion model
            gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            resized_roi = cv2.resize(gray_roi, (48, 48))
            normalized_roi = resized_roi / 255.0
            reshaped_roi = np.reshape(normalized_roi, (1, 48, 48, 1))
            
            # Predict emotion
            emotion_probs = self.models["face_emotion"].predict(reshaped_roi)[0]
            emotion_idx = np.argmax(emotion_probs)
            emotion = EMOTION_LABELS[emotion_idx]
            confidence = float(emotion_probs[emotion_idx])
            
            # Draw bounding box and emotion
            cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
            cv2.putText(frame, f"{emotion}: {confidence:.2f}", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Process with face mesh for detailed landmarks
            results_mesh = self.models["mediapipe_face_mesh"].process(rgb_frame)
            if results_mesh.multi_face_landmarks:
                for face_landmarks in results_mesh.multi_face_landmarks:
                    mp_drawing = mp.solutions.drawing_utils
                    mp_face_mesh = mp.solutions.face_mesh
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                    )
            
            return emotion, confidence, frame
            
        except Exception as e:
            return "Error", None, frame

    def extract_audio(self, video_path, output_path=None):
        """Extract audio from video using moviepy."""
        if output_path is None:
            output_path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
            
        try:
            video_clip = VideoFileClip(video_path)
            audio_clip = video_clip.audio
            audio_clip.write_audiofile(output_path, verbose=False, logger=None)
            audio_clip.close()
            video_clip.close()
            return output_path
        except Exception as e:
            st.error(f"Error extracting audio: {str(e)}")
            return None

    def analyze_speech_emotions(self, video_path):
        """Analyze emotions from speech using Whisper and DistilBERT."""
        # Extract audio from video
        audio_path = self.extract_audio(video_path)
        if not audio_path:
            return "Audio extraction failed", None, None, None
        
        # Load audio for additional features
        try:
            y, sr = librosa.load(audio_path)
            
            # Extract audio features
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            # Compute average pitch (fundamental frequency)
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch = np.mean(pitches[magnitudes > 0.1]) if np.any(magnitudes > 0.1) else 0
            # Compute loudness
            rms = librosa.feature.rms(y=y)[0]
            loudness = np.mean(rms)
            # Speaking rate approximation (using zero crossings)
            zero_crossings = librosa.feature.zero_crossing_rate(y)[0]
            speaking_rate = np.mean(zero_crossings) * sr
        except Exception as e:
            st.warning(f"Audio feature extraction limited: {str(e)}")
            pitch = 0
            loudness = 0
            speaking_rate = 0
            mfcc = None

        # Transcribe audio and analyze emotions in the text
        try:
            result = self.models["whisper"].transcribe(audio_path)
            text = result["text"]
            emotions = self.models["speech_emotion"](text)
            
            # Clean up
            os.unlink(audio_path)
            
            audio_features = {
                "pitch": float(pitch),
                "loudness": float(loudness),
                "speaking_rate": float(speaking_rate)
            }
            
            return text, emotions[0]["label"], emotions[0]["score"], audio_features
        except Exception as e:
            # Clean up
            if os.path.exists(audio_path):
                os.unlink(audio_path)
            return f"Error in speech analysis: {str(e)}", None, None, None

    def process_audio_chunk(self, audio_chunk):
        """Process an audio chunk for streaming audio analysis."""
        # Save the audio chunk to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp:
            temp_path = temp.name
            with open(temp_path, 'wb') as f:
                f.write(audio_chunk)
        
        # Analyze with Whisper
        try:
            result = self.models["whisper"].transcribe(temp_path)
            text = result["text"]
            if text.strip():  # Only analyze if there's text
                emotions = self.models["speech_emotion"](text)
                emotion = emotions[0]["label"]
                confidence = emotions[0]["score"]
            else:
                emotion = "neutral"
                confidence = 1.0
                
            # Clean up
            os.unlink(temp_path)
            return text, emotion, confidence
        except Exception as e:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            return "", "error", 0.0

    def analyze_frame_realtime(self, frame):
        """Analyze a single frame for real-time processing."""
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        annotated_frame = frame.copy()
        
        # Process with MediaPipe Face Detection
        results = self.models["mediapipe_face"].process(rgb_frame)
        
        emotion = "No face"
        confidence = 0.0
        
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                
                # Get image dimensions
                h, w, _ = frame.shape
                
                # Convert relative coordinates to absolute
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Ensure coordinates are within image bounds
                x = max(0, x)
                y = max(0, y)
                width = min(width, w - x)
                height = min(height, h - y)
                
                # Extract face ROI
                face_roi = frame[y:y+height, x:x+width]
                if face_roi.size == 0:
                    continue
                    
                # Preprocess for emotion model
                gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                resized_roi = cv2.resize(gray_roi, (48, 48))
                normalized_roi = resized_roi / 255.0
                reshaped_roi = np.reshape(normalized_roi, (1, 48, 48, 1))
                
                # Predict emotion
                emotion_probs = self.models["face_emotion"].predict(reshaped_roi)[0]
                emotion_idx = np.argmax(emotion_probs)
                emotion = EMOTION_LABELS[emotion_idx]
                confidence = float(emotion_probs[emotion_idx])
                
                # Draw bounding box and emotion
                cv2.rectangle(annotated_frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"{emotion}: {confidence:.2f}", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Process with face mesh for detailed landmarks
                results_mesh = self.models["mediapipe_face_mesh"].process(rgb_frame)
                if results_mesh.multi_face_landmarks:
                    for face_landmarks in results_mesh.multi_face_landmarks:
                        mp_drawing = mp.solutions.drawing_utils
                        mp_face_mesh = mp.solutions.face_mesh
                        mp_drawing.draw_landmarks(
                            image=annotated_frame,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                        )
                
                # Only process the first face for now
                break
        
        # Add timestamp and detected emotion to history
        timestamp = time.time()
        self.emotion_history.append({
            'timestamp': timestamp,
            'emotion': emotion,
            'confidence': confidence
        })
        
        # Keep only the last 100 entries to limit memory usage
        if len(self.emotion_history) > 100:
            self.emotion_history = self.emotion_history[-100:]
        
        # Convert BGR to RGB for Streamlit
        rgb_annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        return rgb_annotated_frame, emotion, confidence

    def comprehensive_analysis(self, video_path):
        """Perform both facial and speech emotion analysis."""
        # Facial emotion analysis
        facial_emotion, emotion_counts, frame_emotions, face_confidence, analyzed_frames = self.analyze_facial_emotions_deepface(video_path)
        
        # Speech emotion analysis
        transcription, speech_emotion, speech_confidence, audio_features = self.analyze_speech_emotions(video_path)

        # Convert analyzed frames to base64 for display
        encoded_frames = []
        for frame in analyzed_frames:
            # Convert OpenCV BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            pil_img = Image.fromarray(rgb_frame)
            # Save to bytes buffer
            buffer = io.BytesIO()
            pil_img.save(buffer, format="JPEG")
            # Get base64 encoding
            img_str = base64.b64encode(buffer.getvalue()).decode()
            encoded_frames.append(f"data:image/jpeg;base64,{img_str}")

        # Combine results
        analysis_results = {
            "facial_analysis": {
                "dominant_emotion": facial_emotion,
                "confidence": face_confidence,
                "emotion_distribution": emotion_counts,
                "frame_by_frame": frame_emotions,
                "processed_frames": encoded_frames
            },
            "speech_analysis": {
                "transcription": transcription,
                "detected_emotion": speech_emotion,
                "confidence": speech_confidence,
                "audio_features": audio_features
            },
            "combined_analysis": {
                "emotion_match": facial_emotion == speech_emotion,
                "primary_emotion": facial_emotion if face_confidence > speech_confidence else speech_emotion,
                "overall_confidence": max(face_confidence, speech_confidence) if face_confidence and speech_confidence else None
            }
        }
        return analysis_results

    def generate_report(self, analysis_results):
        """Generate a detailed report from analysis results."""
        report = {
            "summary": {
                "primary_emotion": analysis_results["combined_analysis"]["primary_emotion"],
                "facial_emotion": analysis_results["facial_analysis"]["dominant_emotion"],
                "speech_emotion": analysis_results["speech_analysis"]["detected_emotion"],
                "emotion_agreement": analysis_results["combined_analysis"]["emotion_match"]
            },
            "emotion_intensity": {
                "facial": analysis_results["facial_analysis"]["confidence"],
                "speech": analysis_results["speech_analysis"]["confidence"]
            },
            "communication_metrics": {
                "speech_clarity": analysis_results["speech_analysis"]["transcription"] != "Audio extraction failed" and 
                                 analysis_results["speech_analysis"]["transcription"] != "Error in speech analysis",
                "audio_features": analysis_results["speech_analysis"]["audio_features"]
            },
            "recommendations": self._generate_recommendations(analysis_results)
        }
        return report
        
    def _generate_recommendations(self, analysis_results):
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        # Check for emotion mismatch
        if not analysis_results["combined_analysis"]["emotion_match"]:
            recommendations.append(
                "There's a mismatch between facial emotion and speech emotion, " + 
                "which may indicate mixed feelings or incongruent communication."
            )
        
        # Check dominant emotions and provide specific advice
        facial_emotion = analysis_results["facial_analysis"]["dominant_emotion"]
        speech_emotion = analysis_results["speech_analysis"]["detected_emotion"]
        
        if facial_emotion in ["angry", "fear", "sad", "disgust"] or speech_emotion in ["anger", "fear", "sadness", "disgust"]:
            recommendations.append(
                "Negative emotions detected. Consider techniques for emotional regulation " +
                "such as deep breathing or reframing the situation."
            )
            
        if facial_emotion in ["surprise"] or speech_emotion in ["surprise"]:
            recommendations.append(
                "Surprise detected. This could indicate new information or unexpected events."
            )
            
        # Generic recommendation if none specific
        if not recommendations:
            recommendations.append(
                "Continue monitoring emotions over time to establish baseline patterns."
            )
            
        return recommendations

#########################################
# Streamlit App Interface               #
#########################################

def main():
    # Initialize session state attributes if they don't exist
    if 'stream_active' not in st.session_state:
        st.session_state.stream_active = False
    if 'emotion_history' not in st.session_state:
        st.session_state.emotion_history = []
    if 'audio_buffer' not in st.session_state:
        st.session_state.audio_buffer = []
    if 'record_stream' not in st.session_state:
        st.session_state.record_stream = False
    if 'audio_enabled' not in st.session_state:
        st.session_state.audio_enabled = True
    if 'show_landmarks' not in st.session_state:
        st.session_state.show_landmarks = True
    if 'frame_rate' not in st.session_state:
        st.session_state.frame_rate = 15  # Default frame rate

    # Load models once
    models = load_models()
    analyzer = EmotionAnalyzer(models)
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        color: #4A56E2;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .section-header {
        color: #6E7CF3;
        font-weight: 600;
    }
    .stButton > button {
        background-color: #4A56E2;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    .status-box {
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .status-active {
        background-color: rgba(0, 255, 0, 0.1);
        border: 1px solid #00FF00;
    }
    .status-inactive {
        background-color: rgba(255, 0, 0, 0.1);
        border: 1px solid #FF0000;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # App title
    st.markdown('<h1 class="main-header">Comprehensive Emotion Analysis System</h1>', unsafe_allow_html=True)
    
    # Sidebar mode selection
    mode = st.sidebar.radio("Select Mode", [
        "Upload Video Analysis", 
        "Webcam Emotion Detection", 
        "Live Streaming Analysis",
        "Emotion Analytics Dashboard"
    ])
    
    if mode == "Upload Video Analysis":
        run_upload_mode(analyzer)
    elif mode == "Webcam Emotion Detection":
        run_webcam_mode(analyzer)
    elif mode == "Live Streaming Analysis":
        run_streaming_mode(analyzer)
    elif mode == "Emotion Analytics Dashboard":
        run_analytics_dashboard(analyzer)

def run_upload_mode(analyzer):
    st.markdown('<h2 class="section-header">Upload Video for Comprehensive Analysis</h2>', unsafe_allow_html=True)
    
    # File uploader
    video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
    
    # Analysis options
    with st.expander("Analysis Options", expanded=False):
        use_mediapipe = st.checkbox("Use MediaPipe for faster face analysis", value=True)
        generate_report = st.checkbox("Generate detailed analysis report", value=True)
    
    if video_file is not None:
        # Save uploaded video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(video_file.read())
        video_path = tfile.name
        
        # Display video
        st.video(video_file)
        
        # Run analysis
        with st.spinner("Analyzing video. This may take a while..."):
            results = analyzer.comprehensive_analysis(video_path)
        
        # Show results in tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Facial Analysis", "Speech Analysis", "Report"])
        
        with tab1:
            st.markdown('<h3 class="section-header">Analysis Summary</h3>', unsafe_allow_html=True)
            
            # Create two columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Facial Emotion")
                st.markdown(f"**Primary Emotion:** {results['facial_analysis']['dominant_emotion']}")
                st.markdown(f"**Confidence:** {results['facial_analysis']['confidence']:.2f}")
                
                # Display emotion distribution as a pie chart
                if results['facial_analysis']['emotion_distribution']:
                    emotions = list(results['facial_analysis']['emotion_distribution'].keys())
                    counts = list(results['facial_analysis']['emotion_distribution'].values())
                    df = pd.DataFrame({'Emotion': emotions, 'Count': counts})
                    fig = px.pie(df, values='Count', names='Emotion', title='Facial Emotion Distribution')
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### Speech Emotion")
                st.markdown(f"**Detected Emotion:** {results['speech_analysis']['detected_emotion']}")
                st.markdown(f"**Confidence:** {results['speech_analysis']['confidence']:.2f}")
                
                # Display audio features
                if results['speech_analysis']['audio_features']:
                    st.markdown("### Audio Features")
                    audio_features = results['speech_analysis']['audio_features']
                    st.markdown(f"**Average Pitch:** {audio_features['pitch']:.2f} Hz")
                    st.markdown(f"**Loudness:** {audio_features['loudness']:.2f}")
                    st.markdown(f"**Speaking Rate:** {audio_features['speaking_rate']:.2f} crossings/sec")
            
            # Combined analysis
            st.markdown("### Overall Assessment")
            st.markdown(f"**Primary Emotion:** {results['combined_analysis']['primary_emotion']}")
            st.markdown(f"**Emotion Match:** {'Yes' if results['combined_analysis']['emotion_match'] else 'No - Potential emotional incongruence'}")
        
        with tab2:
            st.markdown('<h3 class="section-header">Facial Emotion Analysis</h3>', unsafe_allow_html=True)
            
            # Display processed frames
            st.markdown("### Processed Frames")
            if results['facial_analysis']['processed_frames']:
                images_per_row = 3
                for i in range(0, len(results['facial_analysis']['processed_frames']), images_per_row):
                    cols = st.columns(images_per_row)
                    for j in range(images_per_row):
                        if i + j < len(results['facial_analysis']['processed_frames']):
                            cols[j].image(results['facial_analysis']['processed_frames'][i + j], use_column_width=True)
        
        with tab3:
            st.markdown('<h3 class="section-header">Speech Analysis</h3>', unsafe_allow_html=True)
            
            # Display transcription
            st.markdown("### Transcription")
            st.write(results['speech_analysis']['transcription'])
            
            # Display detected emotion
            st.markdown("### Emotion Analysis")
            st.write(f"**Detected Emotion:** {results['speech_analysis']['detected_emotion']}")
            st.write(f"**Confidence:** {results['speech_analysis']['confidence']:.2f}")
        
        with tab4:
            if generate_report:
                st.markdown('<h3 class="section-header">Detailed Analysis Report</h3>', unsafe_allow_html=True)
                report = analyzer.generate_report(results)
                
                # Summary section
                st.markdown("### Emotion Summary")
                st.markdown(f"**Primary Emotion:** {report['summary']['primary_emotion']}")
                st.markdown(f"**Emotion Agreement:** {'Yes' if report['summary']['emotion_agreement'] else 'No'}")
                
                # Emotion intensity
                st.markdown("### Emotion Intensity")
                intensity_data = pd.DataFrame({
                    'Channel': ['Facial', 'Speech'],
                    'Confidence': [report['emotion_intensity']['facial'], report['emotion_intensity']['speech']]
                })
                fig = px.bar(intensity_data, x='Channel', y='Confidence', 
                             title='Emotion Detection Confidence by Channel')
                st.plotly_chart(fig, use_container_width=True)
                
                # Communication metrics
                st.markdown("### Communication Metrics")
                st.markdown(f"**Speech Clarity:** {'Good' if report['communication_metrics']['speech_clarity'] else 'Poor'}")
                
                # Recommendations
                st.markdown("### Recommendations")
                for rec in report['recommendations']:
                    st.markdown(f"- {rec}")
            else:
                st.info("Enable 'Generate detailed analysis report' in the options to view a complete report.")
        
        # Cleanup temporary files
        os.unlink(video_path)

def run_webcam_mode(analyzer):
    st.markdown('<h2 class="section-header">Webcam Emotion Detection</h2>', unsafe_allow_html=True)
    
    # Two modes: Capture image or continuous stream
    webcam_mode = st.radio("Select capture mode:", ["Single Image", "Continuous Stream (5 seconds)"])
    
    if webcam_mode == "Single Image":
        # Use Streamlit's built-in camera input
        image_file = st.camera_input("Take a picture")
        
        if image_file is not None:
            # Convert uploaded image to a format OpenCV can work with
            file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
            cv2_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Process image for emotion recognition
            processed_img, emotion, confidence = analyzer.analyze_frame_realtime(cv2_img)
            
            # Display results
            st.image(processed_img, caption=f"Detected Emotion: {emotion} ({confidence:.2f})", use_column_width=True)
            
            # Emotional analysis tips
            st.markdown("### Emotion Analysis")
            if emotion in ["Happy", "Surprise"]:
                st.success(f"Detected positive emotion: {emotion}")
            elif emotion in ["Sad", "Angry", "Fear", "Disgust"]:
                st.warning(f"Detected negative emotion: {emotion}")
            elif emotion == "Neutral":
                st.info("Detected neutral expression")
            else:
                st.error("No face detected or unable to recognize emotion")
    
    else:  # Continuous Stream
        st.markdown("Press the button to start a 5-second emotion recognition stream")
        
        if st.button("Start Webcam Stream"):
            # Initialize webcam
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Could not open webcam. Please check your camera connection.")
                return
                
            # Create a placeholder for the stream
            stream_placeholder = st.empty()
            
            # Create a placeholder for emotion trend
            trend_placeholder = st.empty()
            
            # Reset emotion history
            analyzer.emotion_history = []
            
            # Stream for 5 seconds
            start_time = time.time()
            emotions_detected = []
            
            while time.time() - start_time < 5:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture image from webcam")
                    break
                
                # Process frame
                processed_frame, emotion, confidence = analyzer.analyze_frame_realtime(frame)
                emotions_detected.append((emotion, confidence))
                
                # Update the display
                stream_placeholder.image(processed_frame, channels="RGB", use_column_width=True)
                
                # Display real-time emotion trend
                if len(analyzer.emotion_history) > 1:
                    df = pd.DataFrame(analyzer.emotion_history)
                    # Convert timestamp to relative time in seconds
                    df['relative_time'] = df['timestamp'] - df['timestamp'].iloc[0]
                    
                    # Create a line chart of emotions over time
                    emotion_codes = {emotion: i for i, emotion in enumerate(set(df['emotion']))}
                    df['emotion_code'] = df['emotion'].map(emotion_codes)
                    
                    fig = px.line(df, x='relative_time', y='confidence', 
                                 color='emotion', title='Emotion Confidence Over Time')
                    trend_placeholder.plotly_chart(fig, use_container_width=True)
                
                # Short delay to reduce CPU usage
                time.sleep(0.1)
            
            # Release the webcam
            cap.release()
            
            # Final analysis
            emotion_counts = Counter([e[0] for e in emotions_detected])
            primary_emotion = emotion_counts.most_common(1)[0][0] if emotion_counts else "None"
            
            st.markdown(f"### Analysis Complete")
            st.markdown(f"**Primary emotion detected:** {primary_emotion}")
            
            # Display emotion distribution
            emotions = list(emotion_counts.keys())
            counts = list(emotion_counts.values())
            df = pd.DataFrame({'Emotion': emotions, 'Count': counts})
            fig = px.pie(df, values='Count', names='Emotion', title='Emotions Detected During Stream')
            st.plotly_chart(fig, use_container_width=True)

def run_streaming_mode(analyzer):
    st.markdown('<h2 class="section-header">Live Streaming Analysis</h2>', unsafe_allow_html=True)
    
    # Status indicator
    if st.session_state.stream_active:
        st.markdown('<div class="status-box status-active">Stream Active</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-box status-inactive">Stream Inactive</div>', unsafe_allow_html=True)
    
    # Stream controls
    col1, col2 = st.columns(2)
    
    with col1:
        if not st.session_state.stream_active:
            if st.button("Start Stream"):
                st.session_state.stream_active = True
                st.session_state.emotion_history = []
                st.rerun()  # Changed from st.experimental_rerun() to st.rerun()
    
    with col2:
        if st.session_state.stream_active:
            if st.button("Stop Stream"):
                st.session_state.stream_active = False
                st.rerun()  # Changed from st.experimental_rerun() to st.rerun()
    
    # Stream settings (only when inactive)
    if not st.session_state.stream_active:
        with st.expander("Stream Settings", expanded=False):
            st.slider("Video Frame Rate", min_value=1, max_value=30, value=15, key="frame_rate")
            st.checkbox("Enable Audio Analysis", value=True, key="audio_enabled")
            st.checkbox("Show Facial Landmarks", value=True, key="show_landmarks")
            st.checkbox("Record Stream for Later Analysis", value=False, key="record_stream")
    
    # Main streaming interface
    if st.session_state.stream_active:
        # Create layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Video stream placeholder
            video_placeholder = st.empty()
            
            # Initialize webcam
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Could not open webcam. Please check your camera connection.")
                st.session_state.stream_active = False
                st.rerun()  # Changed from st.experimental_rerun() to st.rerun()
                return
            
            # Initialize recording if enabled
            if st.session_state.record_stream:
                # Create a temporary file for recording
                temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                temp_video_path = temp_video.name
                
                # Get video properties
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = st.session_state.frame_rate
                
                # Create VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(temp_video_path, fourcc, fps, (frame_width, frame_height))
            
            # Variables for tracking
            frame_count = 0
            start_time = time.time()
            current_emotion = "Unknown"
            emotion_confidence = 0.0
            
            try:
                while st.session_state.stream_active:
                    # Read frame
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to capture image from webcam")
                        break
                    
                    # Process every nth frame based on frame rate
                    if frame_count % max(1, int(30 / st.session_state.frame_rate)) == 0:
                        # Process frame
                        processed_frame, emotion, confidence = analyzer.analyze_frame_realtime(frame)
                        current_emotion = emotion
                        emotion_confidence = confidence
                    
                    # Record if enabled
                    if st.session_state.record_stream:
                        out.write(frame)
                    
                    # Display the processed frame
                    video_placeholder.image(processed_frame, channels="RGB", use_column_width=True)
                    
                    # Increment frame count
                    frame_count += 1
                    
                    # Short delay to control frame rate
                    elapsed = time.time() - start_time
                    desired_elapsed = frame_count / 30.0  # Assuming camera is 30fps
                    if elapsed < desired_elapsed:
                        time.sleep(desired_elapsed - elapsed)
            except Exception as e:
                st.error(f"Stream error: {str(e)}")
            finally:
                # Clean up
                cap.release()
                if st.session_state.record_stream and 'out' in locals():
                    out.release()
                    
                    # Offer download of recorded video
                    with open(temp_video_path, 'rb') as f:
                        video_bytes = f.read()
                        
                    st.download_button(
                        label="Download Recorded Video",
                        data=video_bytes,
                        file_name="emotion_analysis_recording.mp4",
                        mime="video/mp4"
                    )
        
        with col2:
            # Real-time emotion display
            st.markdown(f"### Current Emotion")
            st.markdown(f"<h3 style='color:#4A56E2'>{current_emotion}</h3>", unsafe_allow_html=True)
            st.progress(emotion_confidence)
            
            # Emotion history chart
            if len(analyzer.emotion_history) > 1:
                st.markdown("### Emotion Trend")
                df = pd.DataFrame(analyzer.emotion_history)
                
                # Convert timestamp to relative time in seconds
                if len(df) > 0 and 'timestamp' in df.columns:
                    df['relative_time'] = df['timestamp'] - df['timestamp'].iloc[0]
                    
                    # Create a line chart of emotions over time
                    emotion_codes = {emotion: i for i, emotion in enumerate(set(df['emotion']))}
                    df['emotion_code'] = df['emotion'].map(emotion_codes)
                    
                    fig = px.line(df, x='relative_time', y='confidence', 
                                 color='emotion', title='Emotion Confidence Over Time')
                    st.plotly_chart(fig, use_container_width=True)
            
            # Audio analysis results (if enabled)
            if st.session_state.audio_enabled and len(st.session_state.audio_buffer) > 0:
                st.markdown("### Speech Analysis")
                # Placeholder for audio analysis results
                st.info("Speech emotion analysis would appear here")

def run_analytics_dashboard(analyzer):
    st.markdown('<h2 class="section-header">Emotion Analytics Dashboard</h2>', unsafe_allow_html=True)
    
    # Check if we have emotion history data
    if not analyzer.emotion_history and not st.session_state.emotion_history:
        st.info("No emotion data available yet. Use the streaming or webcam features to collect data first.")
        
        # Demo data option
        if st.button("Load Demo Data"):
            # Generate some demo data
            emotions = ["Happy", "Neutral", "Sad", "Angry", "Surprise", "Fear"]
            weights = [0.4, 0.3, 0.1, 0.1, 0.05, 0.05]  # More happy and neutral emotions
            
            demo_data = []
            base_time = time.time() - 60  # Start 60 seconds ago
            
            for i in range(100):
                emotion = np.random.choice(emotions, p=weights)
                # Make transitions more realistic
                if i > 0 and np.random.random() < 0.7:  # 70% chance to keep the same emotion
                    emotion = demo_data[-1]['emotion']
                
                confidence = min(1.0, max(0.4, np.random.normal(0.7, 0.15)))
                timestamp = base_time + i * 0.6  # Every 0.6 seconds
                
                demo_data.append({
                    'timestamp': timestamp,
                    'emotion': emotion,
                    'confidence': confidence
                })
            
            st.session_state.emotion_history = demo_data
            st.experimental_rerun()
        
        return
    
    # Use session state or analyzer history, whichever has data
    history_data = st.session_state.emotion_history if st.session_state.emotion_history else analyzer.emotion_history
    
    # Convert to DataFrame
    df = pd.DataFrame(history_data)
    
    # Add relative time for better display
    df['relative_time'] = df['timestamp'] - df['timestamp'].min()
    
    # Dashboard layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Emotion Distribution")
        
        # Count emotions
        emotion_counts = df['emotion'].value_counts().reset_index()
        emotion_counts.columns = ['Emotion', 'Count']
        
        # Create pie chart
        fig = px.pie(emotion_counts, values='Count', names='Emotion', 
                    title='Emotion Distribution',
                    color_discrete_sequence=px.colors.qualitative.Bold)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Emotion Confidence")
        
        # Group by emotion and calculate average confidence
        emotion_confidence = df.groupby('emotion')['confidence'].mean().reset_index()
        emotion_confidence.columns = ['Emotion', 'Average Confidence']
        
        # Create bar chart
        fig = px.bar(emotion_confidence, x='Emotion', y='Average Confidence',
                    title='Average Confidence by Emotion',
                    color='Emotion', color_discrete_sequence=px.colors.qualitative.Bold)
        st.plotly_chart(fig, use_container_width=True)
    
    # Timeline analysis
    st.markdown("### Emotion Timeline")
    
    # Create line chart
    fig = px.line(df, x='relative_time', y='confidence', color='emotion',
                 title='Emotion Confidence Over Time',
                 labels={'relative_time': 'Time (seconds)', 'confidence': 'Confidence'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Emotion transitions
    st.markdown("### Emotion Transitions")
    
    # Create a shifted column to analyze transitions
    df['next_emotion'] = df['emotion'].shift(-1)
    transition_df = df.dropna(subset=['next_emotion']).copy()
    
    # Count transitions
    transitions = transition_df.groupby(['emotion', 'next_emotion']).size().reset_index()
    transitions.columns = ['From', 'To', 'Count']
    
    # Filter to only show significant transitions
    transitions = transitions[transitions['Count'] > 1]
    
    if not transitions.empty:
        # Create a heatmap
        transition_matrix = transitions.pivot(index='From', columns='To', values='Count').fillna(0)
        fig = px.imshow(transition_matrix, text_auto=True,
                       title='Emotion Transition Frequency',
                       color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough emotion transitions to display a heatmap")
    
    # Insights section
    st.markdown("### Emotion Analysis Insights")
    
    # Most frequent emotion
    most_common = df['emotion'].mode()[0]
    st.markdown(f"**Most common emotion:** {most_common}")
    
    # Emotion stability (how often emotions change)
    emotion_changes = (df['emotion'] != df['emotion'].shift(1)).sum()
    stability = 1 - (emotion_changes / len(df))
    st.markdown(f"**Emotional stability:** {stability:.2f} (higher means more stable)")
    
    # Additional insights based on emotions detected
    if 'Happy' in df['emotion'].values or 'Surprise' in df['emotion'].values:
        positive_ratio = df[df['emotion'].isin(['Happy', 'Surprise'])].shape[0] / df.shape[0]
        st.markdown(f"**Positive emotion ratio:** {positive_ratio:.2f}")
    
    if 'Angry' in df['emotion'].values or 'Sad' in df['emotion'].values or 'Fear' in df['emotion'].values:
        negative_ratio = df[df['emotion'].isin(['Angry', 'Sad', 'Fear'])].shape[0] / df.shape[0]
        st.markdown(f"**Negative emotion ratio:** {negative_ratio:.2f}")
    
    # Option to export data
    if st.button("Export Analysis Data"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="emotion_analysis_data.csv",
            mime="text/csv"
        )

if __name__ == '__main__':
    main()