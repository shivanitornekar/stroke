import streamlit as st
import cv2
import mediapipe as mp
import time
import math
import numpy as np

def check_drooping_smile():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
    cap = cv2.VideoCapture(0)

    start_time = time.time()
    detection_duration = 10  # seconds
    
    try:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue
    
            image = cv2.flip(image, 1)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)
    
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    h, w, _ = image.shape
                    right_mouth_corner = face_landmarks.landmark[61]
                    left_mouth_corner = face_landmarks.landmark[291]
    
                    right_y = right_mouth_corner.y * h
                    left_y = left_mouth_corner.y * h
    
                    threshold = 5
                    if abs(left_y - right_y) > threshold:
                        cap.release()
                        cv2.destroyAllWindows()
                        return True
    
            elapsed_time = time.time() - start_time
            if elapsed_time > detection_duration:
                break
    
            if cv2.waitKey(1) & 0xFF == 27:
                break
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return False

def check_arm_balance():
    # Initialize MediaPipe Pose Detection
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    
    # Open the webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Failed to open the video capture.")
        return False
    
    # Define the required angles for straight arms
    STRAIGHT_ARM_ANGLE = 180  # degrees
    ARM_ANGLE_TOLERANCE = 60  # degrees
    
    # Define the shoulder length threshold
    SHOULDER_LENGTH_THRESHOLD = 0.3  # relative to body height
    
    # Delay between frames (in milliseconds)
    FRAME_DELAY = 10
    
    # Monitoring delay and duration
    MONITORING_DELAY = 3  # seconds
    MONITORING_DURATION = 10  # seconds

    for attempt in range(2):  # Give two tries
        print(f"Attempt {attempt + 1}/2: Raise your arms and hold for 10 seconds after 3 seconds.")
        
        start_time = None
        arm_straight_duration = 0
        initial_time = time.time()
        
        while True:
            # Read a frame from the video stream
            ret, frame = cap.read()
    
            # Check if the frame was successfully read
            if not ret:
                print("Failed to read the frame.")
                break
    
            elapsed_time = time.time() - initial_time
            
            if elapsed_time < MONITORING_DELAY:
                # Display the countdown on the frame
                countdown = MONITORING_DELAY - int(elapsed_time)
                cv2.putText(frame, f"Starting in {countdown}...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            else:
                # Convert the frame to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
                # Process the frame with Pose Detection
                results = pose.process(rgb_frame)
        
                # Access the detected pose landmarks
                if results.pose_landmarks:
                    # Get the relevant landmark coordinates
                    landmarks = results.pose_landmarks.landmark
        
                    # Calculate the angles of the arms
                    left_shoulder = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y])
                    left_elbow = np.array([landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y])
                    left_wrist = np.array([landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y])
        
                    right_shoulder = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y])
                    right_elbow = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y])
                    right_wrist = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y])
        
                    left_arm_vec = left_wrist - left_elbow
                    left_arm_vec /= np.linalg.norm(left_arm_vec)
        
                    right_arm_vec = right_wrist - right_elbow
                    right_arm_vec /= np.linalg.norm(right_arm_vec)
        
                    left_arm_angle = math.acos(np.dot(left_arm_vec, left_shoulder - left_elbow) / np.linalg.norm(left_shoulder - left_elbow)) * 180 / math.pi
                    right_arm_angle = math.acos(np.dot(right_arm_vec, right_shoulder - right_elbow) / np.linalg.norm(right_shoulder - right_elbow)) * 180 / math.pi
        
                    # Check if the arms are straight
                    arms_straight = (
                        abs(left_arm_angle - STRAIGHT_ARM_ANGLE) <= ARM_ANGLE_TOLERANCE and
                        abs(right_arm_angle - STRAIGHT_ARM_ANGLE) <= ARM_ANGLE_TOLERANCE
                    )
        
                    # Calculate the shoulder length
                    shoulder_length = abs(
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y -
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
                    )
        
                    # Check if the arms are at shoulder length
                    arms_at_shoulder_length = shoulder_length <= SHOULDER_LENGTH_THRESHOLD
        
                    # If the arms are straight and at shoulder length, start/update the timer
                    if arms_straight and arms_at_shoulder_length:
                        if start_time is None:
                            start_time = time.time()
                        arm_straight_duration = time.time() - start_time
                    else:
                        start_time = None
                        arm_straight_duration = 0
        
                    # Check if the person has held the position for 10 seconds
                    if arm_straight_duration >= MONITORING_DURATION:
                        print("Arms held straight for 10 seconds! TEST PASSED!")
                        cap.release()
                        cv2.destroyAllWindows()
                        return False  # Arms held up for 10 seconds successfully
        
            # Display the frame with optional visualizations
            cv2.imshow('Pose Detection', frame)
    
            # Add a delay between frames
            cv2.waitKey(FRAME_DELAY)
    
            # Exit the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if time.time() - initial_time > MONITORING_DELAY + MONITORING_DURATION:
                break
    
    cap.release()
    cv2.destroyAllWindows()
    return True  # Failed to hold arms up for 10 seconds in both tries

def check_slurred_speech():
    # Placeholder function for slurred speech detection
    return True

st.title('Stroke Detection Web App')

# Step 1: Check for drooping smile
st.header('Step 1: Check for Drooping Smile')
if st.button('Run Drooping Smile Test'):
    smile_test_passed = check_drooping_smile()
    if smile_test_passed:
        st.success('Drooping Smile Test Passed')
    else:
        st.error('Drooping Smile Test Failed')

# Step 2: Check for arm balance
if 'smile_test_passed' in locals() and smile_test_passed:
    st.header('Step 2: Check for Arm Balance')
    if st.button('Run Arm Balance Test'):
        arm_test_passed = check_arm_balance()
        if arm_test_passed:
            st.success('Arm Balance Test Passed')
        else:
            st.error('Arm Balance Test Failed')

# Step 3: Check for slurred speech
if 'arm_test_passed' in locals() and arm_test_passed:
    st.header('Step 3: Check for Slurred Speech')
    if st.button('Run Slurred Speech Test'):
        speech_test_passed = check_slurred_speech()
        if speech_test_passed:
            st.success('Slurred Speech Test Passed')
        else:
            st.error('Slurred Speech Test Failed')

# Final Stroke Detection
if 'speech_test_passed' in locals() and speech_test_passed:
    st.header('Final Step: Stroke Detection')
    st.success('All tests passed. Stroke Detected!')
else:
    st.header('Final Step: Stroke Detection')
    st.warning('Not all tests passed. Stroke detection inconclusive.')
