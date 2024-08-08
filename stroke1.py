import streamlit as st
import cv2
import dlib
import numpy as np
import time
import geocoder
import pandas as pd
import speech_recognition as sr
import mediapipe as mp

# Set page to wide layout and custom styling
st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
    .stApp {
        background-color: #fffff0; 
    }
    .title {
        font-size: 48px;
        font-weight: bold;
        color: #041d4f; 
        text-align: center;
    }
    .header {
        font-size: 36px;
        font-weight: bold;
        color: #041d4f;
        text-align: center;
        margin-top: 20px;
    }
    .success {
        color: #041d4f;
        font-weight: bold;
    }
    .error {
        color: #9c191a;
        font-weight: bold;
    }
    .warning {
        color: #041d4f;
        font-weight: bold;
    }
    </style>
    """, 
    unsafe_allow_html=True)

# Initialize dlib's face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize variables for storing test results
face_detected = False
arm_detected = False
speech_detected = False

# Initialize variables for storing results and thresholds
warning_threshold = 100  # Number of consecutive frames showing warning
warning_counter = 0      # Counter for consecutive warnings
baseline_mar = None      # Baseline for face test

# Function to calculate Mouth Aspect Ratio (MAR)
def calculate_mar(shape):
    left_lip = (shape.part(48).x, shape.part(48).y)
    right_lip = (shape.part(54).x, shape.part(54).y)
    top_lip = (shape.part(51).x, shape.part(51).y)
    bottom_lip = (shape.part(57).x, shape.part(57).y)

    mouth_width = np.linalg.norm(np.array(right_lip) - np.array(left_lip))
    mouth_height = np.linalg.norm(np.array(bottom_lip) - np.array(top_lip))

    if mouth_width == 0:
        return 0
    return mouth_height / mouth_width

# Function to perform face test
def perform_face_test():
    global face_detected, baseline_mar, warning_counter

    mar_values = []
    start_time = time.time()

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if len(faces) > 0:
            face = faces[0]
            shape = predictor(gray, face)
            mar = calculate_mar(shape)
            mar_values.append(mar)

            if time.time() - start_time < 20:
                cv2.putText(frame, "Analysing MAR...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (31, 119, 180), 2)
                baseline_mar = np.mean(mar_values)
            else:
                cv2.putText(frame, "Model is ready", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (31, 119, 180), 2)
                if mar < baseline_mar - 0.03:
                    cv2.putText(frame, "Drooping smile detected", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (214, 39, 40), 2)
                    warning_counter += 1
                else:
                    warning_counter = 0

                if warning_counter >= warning_threshold:
                    face_detected = True
                    break

        cv2.imshow('Frame', frame)
        cv2.setWindowProperty('Frame', cv2.WND_PROP_TOPMOST, 1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to perform speech test
def perform_speech_test():
    global speech_detected

    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        st.text("Adjusting for ambient noise, please wait...")
        st.text("Ready to listen...")
        recognizer.adjust_for_ambient_noise(source, duration=2)

        try:
            audio = recognizer.listen(source, timeout=10)
            recognized_speech = recognizer.recognize_google(audio)
            st.success("Speech is normal.")
        except sr.UnknownValueError:
            st.error("Slurred Speech detected.")
            speech_detected = True
        except sr.RequestError as e:
            st.write(f"Could not request results from Google Speech Recognition service; {e}")
        except sr.WaitTimeoutError:
            st.write("Listening timed out while waiting for phrase to start")

# Function to get current location using geocoder
def get_current_location():
    g = geocoder.ip('me')
    return g.latlng

# Function to calculate haversine distance
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c

# Function to get nearest hospitals
def get_nearest_hospitals(current_lat, current_lon, num_hospitals=10):
    df = pd.read_csv('hospitals.csv')
    df['distance'] = haversine(current_lat, current_lon, df['latitude'], df['longitude'])
    nearest_hospitals = df.nsmallest(num_hospitals, 'distance')
    return nearest_hospitals
import cv2
import mediapipe as mp
import numpy as np
import time

# Function to initialize MediaPipe Pose and define arm landmarks and connections
def initialize_pose():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    ARM_LANDMARKS = [
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_ELBOW,
        mp_pose.PoseLandmark.LEFT_WRIST,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_ELBOW,
        mp_pose.PoseLandmark.RIGHT_WRIST
    ]
    ARM_CONNECTIONS = [
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
        (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
        (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST)
    ]
    return pose, ARM_LANDMARKS, ARM_CONNECTIONS

def extract_arm_keypoints(landmarks, mp_pose):
    return {
        'left_shoulder': [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
        'left_elbow': [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y],
        'left_wrist': [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y],
        'right_shoulder': [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y],
        'right_elbow': [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y],
        'right_wrist': [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    }

def check_arm_condition(arm_keypoints):
    left_arm_length = np.linalg.norm(np.array(arm_keypoints['left_shoulder']) - np.array(arm_keypoints['left_wrist']))
    right_arm_length = np.linalg.norm(np.array(arm_keypoints['right_shoulder']) - np.array(arm_keypoints['right_wrist']))
    asymmetry = abs(left_arm_length - right_arm_length) > 0.05
    return asymmetry

def display_message(frame, message, color=(255, 255, 255)):
    cv2.putText(frame, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

def draw_arm_landmarks(frame, landmarks, arm_landmarks, connections):
    for landmark in arm_landmarks:
        idx = landmark.value
        x = int(landmarks[idx].x * frame.shape[1])
        y = int(landmarks[idx].y * frame.shape[0])
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    for connection in connections:
        start_idx = connection[0].value
        end_idx = connection[1].value
        x1, y1 = int(landmarks[start_idx].x * frame.shape[1]), int(landmarks[start_idx].y * frame.shape[0])
        x2, y2 = int(landmarks[end_idx].x * frame.shape[1]), int(landmarks[end_idx].y * frame.shape[0])
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

def analyze_arm_condition():
    pose, ARM_LANDMARKS, ARM_CONNECTIONS = initialize_pose()
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    consecutive_symmetric_frames = 0
    total_symmetric_duration = 0
    total_frames = 0
    start_analysis = False
    asymmetry_detected = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            arm_keypoints = extract_arm_keypoints(results.pose_landmarks.landmark, mp.solutions.pose)
            
            left_arm_raised = arm_keypoints['left_wrist'][1] < arm_keypoints['left_shoulder'][1]
            right_arm_raised = arm_keypoints['right_wrist'][1] < arm_keypoints['right_shoulder'][1]
            
            if left_arm_raised or right_arm_raised:
                if not start_analysis:
                    start_analysis = True
                    start_time = time.time()
                    display_message(frame, 'Arms Detected. Starting analysis...', color=(0, 255, 0))
                else:
                    asymmetry_detected = check_arm_condition(arm_keypoints)
                    
                    if not asymmetry_detected:
                        consecutive_symmetric_frames += 1
                        total_symmetric_duration += 1 / fps
                    else:
                        consecutive_symmetric_frames = 0
                    
                    label = 'Arm Asymmetry Detected' if asymmetry_detected else 'No Asymmetry'
                    color = (0, 0, 255) if asymmetry_detected else (0, 255, 0)
                    display_message(frame, label, color)
            else:
                start_analysis = False
                display_message(frame, 'Please lift your arms.', color=(0, 0, 255))
            
            draw_arm_landmarks(frame, results.pose_landmarks.landmark, ARM_LANDMARKS, ARM_CONNECTIONS)
            total_frames += 1
        
        else:
            start_analysis = False
            display_message(frame, 'Arms not detected. Please raise your hands.', color=(0, 0, 255))
        
        cv2.imshow('Arm Condition Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        if start_analysis and (time.time() - start_time) > 20:
            break

    final_label = 'Arm Asymmetry Detected' if total_symmetric_duration < 14 else 'No Asymmetry'
    

    cap.release()
    cv2.destroyAllWindows()
    return final_label



# Main streamlit app
st.markdown('<h1 class="title">Stroke Detection</h1>', unsafe_allow_html=True)

# Run face test
st.markdown('<h2 class="header">Face Test</h2>', unsafe_allow_html=True)
#st.write("This section detects stroke symptoms based on facial features.")
perform_face_test()
st.markdown('<p class="success">Face test completed.</p>', unsafe_allow_html=True)
if face_detected == True:
    st.error('Drooping Smile detected.')
else:
    st.success('No drooping smile detected.')
#st.write("Results stored.")

# Run arm test
st.markdown('<h2 class="header">Arm Test</h2>', unsafe_allow_html=True)
#st.write("This section detects stroke symptoms based on arm movement.")
arm_test_result = analyze_arm_condition()
st.markdown('<p class="success">Arm test completed.</p>', unsafe_allow_html=True)

if "Asymmetry" in arm_test_result:
    st.error('Arm Asymmetry detected.')
    arm_detected = True
else:
    st.success('No arm asymmetry detected.')


# Run speech test
st.markdown('<h2 class="header">Speech Test</h2>', unsafe_allow_html=True)
#st.write("This section detects stroke symptoms based on speech.")
perform_speech_test()
st.markdown('<p class="success">Speech test completed.</p>', unsafe_allow_html=True)
#st.write("Results stored.")

# Analyze results and display hospitals if stroke is detected
st.markdown('<h2 class="header">Results</h2>', unsafe_allow_html=True)
st.write("Analyzing test results...")

if face_detected and arm_detected and speech_detected:
    st.markdown('<h3 class="warning">Possible stroke detected.</h3>', unsafe_allow_html=True)
    st.write("Fetching nearby hospitals...")

    # Get current location
    current_location = get_current_location()

    if current_location:
        current_lat, current_lon = current_location[0], current_location[1]
        nearest_hospitals = get_nearest_hospitals(current_lat, current_lon)

        # Display hospitals
        st.write("Nearest Hospitals:")
        st.dataframe(nearest_hospitals[['name', 'phone', 'distance']].reset_index(drop=True))
    else:
        st.markdown('<p class="warning">Unable to fetch current location.</p>', unsafe_allow_html=True)
else:
    st.markdown('<h3 class="success">Not all tests passed. Stroke detection inconclusive.</h3>', unsafe_allow_html=True)
