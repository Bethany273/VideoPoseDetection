import cv2
import mediapipe as mp
import numpy as np
import math

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

# ---- REP PARAMETERS ----
FLEXION_THRESHOLD = 20     # must pass this to start a rep
NEUTRAL_THRESHOLD = 10     # must drop below this to finish the rep
MAX_REPS = 5

rep_count = 0
in_rep = False
current_rep_max = 0
rep_max_angles = []

cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks and rep_count < MAX_REPS:
            landmarks = results.pose_landmarks.landmark

            # LEFT SIDE
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

            raw_angle = calculate_angle(ear, shoulder, hip)
            neck_angle = 180 - raw_angle

            #----- REP LOGIC -----
            if not in_rep:
                # Start rep when angle goes above flexion threshold
                if neck_angle > FLEXION_THRESHOLD:
                    in_rep = True
                    current_rep_max = neck_angle
                # track max while flexed
            else:
                current_rep_max = max(current_rep_max, neck_angle)

                # Finish rep when returned to neutral
                if neck_angle < NEUTRAL_THRESHOLD:
                    rep_count += 1
                    rep_max_angles.append(current_rep_max)
                    in_rep = False

            # DISPLAY
            cv2.putText(image, f"Neck Angle: {int(neck_angle)} deg",
                        (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0), 2)

            cv2.putText(image, f"Reps: {rep_count}/{MAX_REPS}",
                        (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 0), 2)

            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

        # After 5 reps — show average max angle
        if rep_count >= MAX_REPS:
            avg_angle = sum(rep_max_angles) / len(rep_max_angles)
            cv2.putText(image, "DONE!",
                        (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                        (0, 0, 255), 3)
            cv2.putText(image, f"Avg Max Angle: {avg_angle:.1f} deg",
                        (40, 140), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 200, 255), 2)

        cv2.imshow('Neck Flexion/Extension', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
