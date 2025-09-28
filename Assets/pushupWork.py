import math
import time
import cv2
import mediapipe as mp
import numpy as np

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 200, 0)
RED = (0, 0, 255)
BLUE = (245, 117, 25)

# Threshold for correct push-up angle
ELBOW_ANGLE_MIN = 30  # Minimum angle for correct push-up (30 degrees)
ELBOW_ANGLE_MAX = 178  # Maximum angle for correct push-up (178 degrees)

# Speed calculation
previous_time = 0
pushup_counter = 0
pushup_start_time = 0
fps = 30  # Frames per second

# Constants for speed calculation (estimate of the distance covered during a push-up)
PUSHUP_DISTANCE_METERS = 0.5  # Example fixed distance for a full push-up cycle (in meters)

# Counter for correct and incorrect poses
correct_pose_count = 0
incorrect_pose_count = 0


# Calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / math.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


# Process a single frame to extract pose
def process_frame(frame, pose):
    height, width, _ = frame.shape
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results, height, width


# Extract pose landmarks
def extract_landmarks(results):
    try:
        landmarks = results.pose_landmarks.landmark
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    except:
        pass
    return shoulder, elbow, wrist


# Calculate push-up speed based on time to reach 160 degrees in the upward phase
def calculate_pushup_speed(elbow_angle, current_time):
    global pushup_start_time
    if elbow_angle >= 160:
        # Calculate time taken to reach the "up" position (160 degrees)
        time_diff = current_time - pushup_start_time
        if time_diff > 0:
            speed = PUSHUP_DISTANCE_METERS / time_diff  # meters per second
        else:
            speed = 0
        return speed
    return 0  # Return 0 if no upward movement yet


# Check if the elbow angle is within the valid range
def check_elbow_angle(elbow_angle):
    if ELBOW_ANGLE_MIN <= elbow_angle <= ELBOW_ANGLE_MAX:
        return True
    return False


# Draw rounded rectangle with text inside
def draw_rounded_rectangle(image, position, size, color, text):
    # Draw the rectangle with rounded corners by creating ellipses at corners
    top_left = (position[0], position[1])
    bottom_right = (position[0] + size[0], position[1] + size[1])
    # Top-left corner ellipse
    cv2.ellipse(image, (position[0] + 20, position[1] + 20), (20, 20), 180, 0, 90, color, -1)
    # Top-right corner ellipse
    cv2.ellipse(image, (position[0] + size[0] - 20, position[1] + 20), (20, 20), 270, 0, 90, color, -1)
    # Bottom-left corner ellipse
    cv2.ellipse(image, (position[0] + 20, position[1] + size[1] - 20), (20, 20), 90, 0, 90, color, -1)
    # Bottom-right corner ellipse
    cv2.ellipse(image, (position[0] + size[0] - 20, position[1] + size[1] - 20), (20, 20), 0, 0, 90, color, -1)

    # Draw the rectangle
    cv2.rectangle(image, top_left, bottom_right, color, -1)

    # Remove the middle part where ellipses are overlapping
    cv2.rectangle(image, (position[0] + 20, position[1] + 20),
                  (position[0] + size[0] - 20, position[1] + size[1] - 20), (0, 0, 0), -1)

    # Draw the text inside the rectangle
    cv2.putText(image, text, (position[0] + 10, position[1] + size[1] // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)
    return image


# Render the frame with UI elements
def render_ui(image, counter, stage, elbow_angle, speed, width, height):
    # Render pose status
    pose_status = "Correct Pose" if check_elbow_angle(elbow_angle) else "Incorrect Pose"

    # Display pose status
    color = GREEN if pose_status == "Correct Pose" else RED
    image = draw_rounded_rectangle(image, (width - 350, 10), (320, 80), color, pose_status)

    # Display push-up counter and speed
    image = draw_rounded_rectangle(image, (10, 50), (260, 80), BLUE, f"Reps: {counter} ({stage})")
    image = draw_rounded_rectangle(image, (10, 150), (260, 80), BLUE, f"Speed: {speed:.2f} m/s")

    # Display elbow angle
    image = draw_rounded_rectangle(image, (10, 250), (260, 80), BLUE, f"Elbow Angle: {elbow_angle:.2f} deg")

    return image


# Main function to run pose detection
def run_pose_detection(mp_drawing, mp_pose, filename):
    global pushup_counter, previous_time, pushup_start_time, correct_pose_count, incorrect_pose_count

    cap = cv2.VideoCapture(filename)
    start_time = time.time()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        stage = "up"  # Initialize stage at the start of the loop
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image, results, height, width = process_frame(frame, pose)
            shoulder, elbow, wrist = extract_landmarks(results)

            if shoulder and elbow and wrist:
                elbow_angle = calculate_angle(shoulder, elbow, wrist)

                # Determine the stage of the push-up
                if elbow_angle > 160:
                    stage = "up"
                if elbow_angle < 50 and stage == 'up':
                    stage = "down"
                    pushup_counter += 1
                    pushup_start_time = time.time()  # Reset start time when push-up is complete

                # Calculate push-up speed based on reaching 160 degrees
                current_time = time.time()
                speed = calculate_pushup_speed(elbow_angle, current_time)

                # Count the correct and incorrect poses
                if check_elbow_angle(elbow_angle):
                    correct_pose_count += 1
                else:
                    incorrect_pose_count += 1

                # Render UI
                image = render_ui(image, pushup_counter, stage, elbow_angle, speed, width, height)

                # Draw landmarks
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=5),
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=5))

            # Display image
            cv2.imshow('Push-up Pose Detection', image)

            # Print angle and pose status in terminal
            print(
                f"Elbow Angle: {elbow_angle:.2f}, Pose Status: {'Correct Pose' if check_elbow_angle(elbow_angle) else 'Incorrect Pose'}")

            # Exit on pressing 'c'
            if cv2.waitKey(10) & 0xFF == ord('c'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # After simulation, display total correct and incorrect poses
        print(f"\nTotal Correct Poses: {correct_pose_count}")
        print(f"Total Incorrect Poses: {incorrect_pose_count}")


if __name__ == "__main__":
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    run_pose_detection(mp_drawing, mp_pose, 'assets/pushup.mp4')  # Replace with your video file

# run_pose_detection(mp_drawing, mp_pose, 'assets/pushup.mp4')