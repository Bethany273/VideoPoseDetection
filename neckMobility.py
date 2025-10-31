import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break

        # Flip image horizontally for better selfie view
        frame = cv2.flip(frame, 1)

        # Convert to RGB for Mediapipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        # Process pose
        results = pose.process(image_rgb)

        # Convert back to BGR for OpenCV
        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Draw pose landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                right_ear = [landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y]
                left_ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

                height, width = image.shape[:2]

                # Convert to pixel coordinates correctly
                left_ear_pixel = (np.array(left_ear) * [width, height]).astype(int)
                right_ear_pixel = (np.array(right_ear) * [width, height]).astype(int)

                # Calculate angle between ears
                dx = left_ear_pixel[0] - right_ear_pixel[0]
                dy = left_ear_pixel[1] - right_ear_pixel[1]
                angle_rad = np.arctan2(dy, dx)
                angle_deg = np.degrees(angle_rad)
                
                #calculate angle between shoulders
                sdx = left_shoulder[0] - right_shoulder[0]
                sdy = left_shoulder[1] - right_shoulder[1]
                shoulder_angle_rad = np.arctan2(sdy, sdx)
                shoulder_angle_deg = np.degrees(shoulder_angle_rad)

                head_tilt = angle_deg - shoulder_angle_deg
                print(f"Angle between ears: {head_tilt:.2f}°")

                # Optional: draw points for clarity
                cv2.circle(image, tuple(left_ear_pixel), 5, (0, 255, 0), -1)
                cv2.circle(image, tuple(right_ear_pixel), 5, (0, 255, 0), -1)

                cv2.putText(image, f"Angle: {angle_deg:.2f}deg",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 255), 2, cv2.LINE_AA)

            except Exception as e:
                print(f"Error calculating angle: {str(e)}")
                cv2.putText(image, "Error calculating angle",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2, cv2.LINE_AA)

        else:
            cv2.putText(image, "No pose detected",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)

        # Show the result window
        cv2.imshow('Mediapipe Feed', image)

        # Press 'q' to quit
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Cleanup
cap.release()
cv2.destroyAllWindows()
