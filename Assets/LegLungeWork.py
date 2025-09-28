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

# Thresholds for lunge angles
KNEE_ANGLE_MIN = 90  # Minimum knee angle for correct lunge (90 degrees)
KNEE_ANGLE_MAX = 180  # Maximum knee angle for correct lunge (180 degrees)

# Speed calculation
previous_time = 0
lunge_counter = 0
lunge_start_time = 0
fps = 30  # Frames per second

# Constants for speed calculation (estimate of the distance covered during a lunge)
LUNGE_DISTANCE_METERS = 0.5  # Example fixed distance for a full lunge cycle (in meters)

# Counter for correct and incorrect poses
correct_pose_count = 0
incorrect_pose_count = 0


# Calculate the angle between three points
def calculate_angle(a, b, c):
    """
    Calculate the angle between three points.
    a = hip, b = knee, c = ankle (for knee angle)
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # Vector from b to a and b to c
    ab = a - b
    bc = c - b

    # Calculate the dot product and magnitude of vectors
    dot_product = np.dot(ab, bc)
    mag_ab = np.linalg.norm(ab)
    mag_bc = np.linalg.norm(bc)

    # Calculate the angle in radians and convert to degrees
    cos_angle = dot_product / (mag_ab * mag_bc)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    angle = np.degrees(angle)

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


# Extract pose landmarks for Lunge
def extract_landmarks(results):
    # Initialize landmarks variables
    hip, knee, ankle = None, None, None

    # Check if pose landmarks are detected
    if results.pose_landmarks:
        try:
            landmarks = results.pose_landmarks.landmark
            # Extract the relevant landmarks
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        except Exception as e:
            print(f"Error extracting landmarks: {e}")

    return hip, knee, ankle


# Calculate lunge speed based on time to reach 150 degrees in the upward phase
def calculate_lunge_speed(knee_angle, current_time):
    global lunge_start_time
    if knee_angle <= 150:
        # Calculate time taken to reach the "up" position (150 degrees)
        time_diff = current_time - lunge_start_time
        if time_diff > 0:
            speed = LUNGE_DISTANCE_METERS / time_diff  # meters per second
        else:
            speed = 0
        return speed
    return 0  # Return 0 if no upward movement yet


# Check if the lunge pose is correct (knee angle range)
def check_lunge_pose(knee_angle):
    if KNEE_ANGLE_MIN <= knee_angle <= KNEE_ANGLE_MAX:
        return True
    return False


# Draw rounded rectangle with text inside
def draw_rounded_rectangle(image, position, size, color, text):
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
def render_ui(image, counter, stage, knee_angle, speed, width, height):
    # Render pose status
    pose_status = "Correct Pose" if check_lunge_pose(knee_angle) else "Incorrect Pose"

    # Display pose status
    color = GREEN if pose_status == "Correct Pose" else RED
    image = draw_rounded_rectangle(image, (width - 350, 10), (320, 80), color, pose_status)

    # Display squat counter, stage, and speed
    image = draw_rounded_rectangle(image, (10, 50), (260, 80), BLUE, f"Reps: {counter} ({stage})")
    image = draw_rounded_rectangle(image, (10, 150), (260, 80), BLUE, f"Speed: {speed:.2f} m/s")

    # Display knee angle
    image = draw_rounded_rectangle(image, (10, 250), (260, 80), BLUE, f"Knee: {knee_angle:.2f} deg")

    return image


# Main function to run pose detection
def run_pose_detection(mp_drawing, mp_pose, filename):
    global lunge_counter, lunge_start_time, correct_pose_count, incorrect_pose_count

    cap = cv2.VideoCapture(filename)
    lunge_start_time = time.time()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        stage = "up"  # Initialize stage at the start of the loop
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image, results, height, width = process_frame(frame, pose)
            hip, knee, ankle = extract_landmarks(results)

            if hip and knee and ankle:
                knee_angle = calculate_angle(hip, knee, ankle)

                # Determine the stage of the lunge
                if knee_angle < 110 and stage == "down":
                    stage = "up"
                    lunge_counter += 1
                    lunge_start_time = time.time()  # Reset start time when lunge is complete
                if knee_angle > 150 and stage == 'up':
                    stage = "down"

                # Calculate lunge speed based on reaching 150 degrees
                current_time = time.time()
                speed = calculate_lunge_speed(knee_angle, current_time)

                # Count the correct and incorrect poses
                if check_lunge_pose(knee_angle):
                    correct_pose_count += 1
                else:
                    incorrect_pose_count += 1

                # Render UI
                image = render_ui(image, lunge_counter, stage, knee_angle, speed, width, height)
                # Draw landmarks
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=5),
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=5))

            # Display image
            cv2.imshow('Lunge Pose Detection', image)

            # Print angle and pose status in terminal
            print(
                f"Knee Angle: {knee_angle:.2f}, Pose Status: {'Correct Pose' if check_lunge_pose(knee_angle) else 'Incorrect Pose'}")

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

    run_pose_detection(mp_drawing, mp_pose, 'assets/LegLunge.mp4')  # Replace with your video file
