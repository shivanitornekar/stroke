import streamlit as st
import cv2
import mediapipe as mp
import time
import math
import numpy as np
import speech_recognition as sr

def check_drooping_smile():
    # Initialize MediaPipe Face Mesh.
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

    # Open the webcam.
    cap = cv2.VideoCapture(0)

    smile_start_time = None
    no_smile_start_time = None

    try:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Flip the image horizontally for a later selfie-view display.
            image = cv2.flip(image, 1)
            # Convert the BGR image to RGB.
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Process the image and find face landmarks.
            results = face_mesh.process(image_rgb)

            # Draw face landmarks and detect drooping smile.
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Get coordinates of specific landmarks (left and right corners of the mouth).
                    h, w, _ = image.shape
                    right_mouth_corner = face_landmarks.landmark[61]
                    left_mouth_corner = face_landmarks.landmark[291]

                    right_mouth_corner_coords = (int(right_mouth_corner.x * w), int(right_mouth_corner.y * h))
                    left_mouth_corner_coords = (int(left_mouth_corner.x * w), int(left_mouth_corner.y * h))

                    # Calculate the vertical positions of the mouth corners
                    right_y = right_mouth_corner.y * h
                    left_y = left_mouth_corner.y * h

                    # Check for drooping smile by comparing the y-coordinates of the mouth corners
                    threshold = 5  # Tweak this threshold as necessary
                    if abs(left_y - right_y) > threshold:
                        if smile_start_time is None:
                            smile_start_time = time.time()
                        elif time.time() - smile_start_time >= 5:
                            cap.release()
                            cv2.destroyAllWindows()
                            return True  # Detected drooping smile for 5 seconds

                        cv2.putText(image, 'Drooping Smile Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        no_smile_start_time = None  # Reset no smile timer
                    else:
                        smile_start_time = None
                        if no_smile_start_time is None:
                            no_smile_start_time = time.time()
                        elif time.time() - no_smile_start_time >= 5:
                            cap.release()
                            cv2.destroyAllWindows()
                            st.write("Test failed: No drooping smile detected.")
                            return False

            # Display the image.
            cv2.imshow('Drooping Smile Test', image)

            # Break the loop if the ESC key is pressed
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return False  # No drooping smile detected for 5 seconds


def arm_balance_test():
    # Initialize MediaPipe Pose Detection
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # Open the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Failed to open the video capture.")
        return

    # Define the required angles for straight arms
    STRAIGHT_ARM_ANGLE = 180  # degrees
    ARM_ANGLE_TOLERANCE = 60  # degrees

    # Define the shoulder length threshold
    SHOULDER_LENGTH_THRESHOLD = 0.3  # relative to body height

    # Delay between frames (in milliseconds)
    FRAME_DELAY = 10

    # Initialize trial variables
    trial_count = 0
    trial_durations = [0, 0]
    trial_start_times = [None, None]
    trial_passed = [False, False]

    def check_arms_straight_and_shoulder_length(landmarks):
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

        arms_straight = (
            abs(left_arm_angle - STRAIGHT_ARM_ANGLE) <= ARM_ANGLE_TOLERANCE and
            abs(right_arm_angle - STRAIGHT_ARM_ANGLE) <= ARM_ANGLE_TOLERANCE
        )

        shoulder_length = abs(
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y -
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y
        )

        arms_at_shoulder_length = shoulder_length <= SHOULDER_LENGTH_THRESHOLD

        return arms_straight and arms_at_shoulder_length

    while True:
        # Read a frame from the video stream
        ret, frame = cap.read()

        # Check if the frame was successfully read
        if not ret:
            print("Failed to read the frame.")
            break

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with Pose Detection
        results = pose.process(rgb_frame)

        # Access the detected pose landmarks
        if results.pose_landmarks:
            # Check if arms are straight and at shoulder length
            arms_straight_and_shoulder_length = check_arms_straight_and_shoulder_length(results.pose_landmarks.landmark)

            if arms_straight_and_shoulder_length:
                if trial_start_times[trial_count] is None:
                    trial_start_times[trial_count] = time.time()
                trial_durations[trial_count] = time.time() - trial_start_times[trial_count]
            else:
                if trial_start_times[trial_count] is not None and trial_durations[trial_count] < 10:
                    print(f"Trial {trial_count + 1}: Arms not held straight for 10 seconds! TRIAL PASSED!")
                    trial_passed[trial_count] = True
                    trial_count += 1
                    if trial_count < 2:
                        trial_durations[trial_count] = 0
                        trial_start_times[trial_count] = None
                else:
                    trial_start_times[trial_count] = None
                    trial_durations[trial_count] = 0

            # Check if the person has held the position for 10 seconds in this trial
            if trial_count < 2 and trial_durations[trial_count] >= 10:
                print(f"Trial {trial_count + 1}: Arms held straight for 10 seconds! TRIAL FAILED!")
                trial_count += 1
                if trial_count < 2:
                    trial_durations[trial_count] = 0
                    trial_start_times[trial_count] = None

        # Display the frame with optional visualizations
        cv2.imshow('Pose Detection', frame)

        # Add a delay between frames
        cv2.waitKey(FRAME_DELAY)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Exit the loop if both trials are completed
        if trial_count >= 2:
            break

    # Check the final result
    if all(trial_passed):
        print("TEST PASSED!")
        return True
    else:
        print("TEST FAILED!")
        return False

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def speech_test():
    # Initialize recognizer
    recognizer = sr.Recognizer()

    # Use the microphone as the source
    with sr.Microphone() as source:
        print("Adjusting for ambient noise, please wait...")
        print("Ready to listen...")
        recognizer.adjust_for_ambient_noise(source, duration=2)

        try:
            # Listen for a command
            audio = recognizer.listen(source, timeout=10)
            
            # Recognize the speech using Google Web Speech API
            print(recognizer.recognize_google(audio))
            print("Your speech is not slurred")
            return False  # Speech is not slurred
        except sr.UnknownValueError:
            print("Your speech is slurred")
            return True  # Speech is slurred
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            return False
        except sr.WaitTimeoutError:
            print("Listening timed out while waiting for phrase to start")
            return False
        
    

def run_tests():
    st.title('Stroke Detection Web App')

    # Initialize session_state
    if 'test_results' not in st.session_state:
        st.session_state.test_results = {
            'smile_test_passed': False,
            'arm_test_passed': False,
            'speech_test_passed': False
        }

    # Step 1: Check for drooping smile
    st.header('Step 1: Check for Drooping Smile')
    if st.button('Run Drooping Smile Test'):
        st.session_state.test_results['smile_test_passed'] = check_drooping_smile()
        if st.session_state.test_results['smile_test_passed']:
            st.success('Drooping Smile Test Passed')
        else:
            st.error('Drooping Smile Test Failed')

    # Step 2: Check for arm balance
    st.header('Step 2: Check for Arm Balance')
    if st.button('Run Arm Balance Test'):
        st.session_state.test_results['arm_test_passed'] = arm_balance_test()
        if st.session_state.test_results['arm_test_passed']:
            st.success('Arm Balance Test Passed')
        else:
            st.error('Arm Balance Test Failed')

    # Step 3: Check for slurred speech
    st.header('Step 3: Check for Slurred Speech')
    if st.button('Run Slurred Speech Test'):
        st.session_state.test_results['speech_test_passed'] = speech_test()
        if st.session_state.test_results['speech_test_passed']:
            st.success('Slurred Speech Test Passed')
        else:
            st.error('Slurred Speech Test Failed')

    # Final Stroke Detection
    if all(st.session_state.test_results.values()):
        st.header('Final Step: Stroke Detection')
        st.success('All tests passed. Stroke Detected!')
    else:
        st.header('Final Step: Stroke Detection')
        st.warning('Not all tests passed. Stroke detection inconclusive.')

run_tests()



