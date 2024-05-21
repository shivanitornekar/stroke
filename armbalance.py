import cv2
import mediapipe as mp
import time
import math
import numpy as np

# Initialize MediaPipe Pose Detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Failed to open the video capture.")
    exit()

# Define the required angles for straight arms
STRAIGHT_ARM_ANGLE = 180  # degrees
ARM_ANGLE_TOLERANCE = 60  # degrees

# Define the shoulder length threshold
SHOULDER_LENGTH_THRESHOLD = 0.3  # relative to body height

# Delay between frames (in milliseconds)
FRAME_DELAY = 10

# Initialize the timer
start_time = None
arm_straight_duration = 0

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

     # Check if the frame was successfully read
    if not ret:
        print("Failed to read the frame.")
        break  # or continue (to skip this iteration)

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
        if arm_straight_duration >= 10:
            print("Arms held straight for 10 seconds! TEST PASSED!")
            # Reset the timer
            start_time = None
            arm_straight_duration = 0

    # Display the frame with optional visualizations
    cv2.imshow('Pose Detection', frame)

    # Add a delay between frames
    cv2.waitKey(FRAME_DELAY)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()