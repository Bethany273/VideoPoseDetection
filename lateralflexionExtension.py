import cv2
import mediapipe as mp
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# --- REP LOGIC ---
ANGLE_THRESHOLD = 15     # Must tilt at least this much to count as a side
NEUTRAL_THRESHOLD = 8     # Return to neutral before next rep

rep_count = 0
direction = "neutral"     # "neutral", "left", "right"
max_left = 0
max_right = 0
last_rep_left = None
last_rep_right = None
rep_max_left = []
rep_max_right = []

MAX_REPS = 5

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = pose.process(image_rgb)
        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        height, width = image.shape[:2]

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            try:
                lm = results.pose_landmarks.landmark
                
                # Extract landmarks
                LE = np.array([lm[mp_pose.PoseLandmark.LEFT_EAR.value].x * width,
                               lm[mp_pose.PoseLandmark.LEFT_EAR.value].y * height])

                RE = np.array([lm[mp_pose.PoseLandmark.RIGHT_EAR.value].x * width,
                               lm[mp_pose.PoseLandmark.RIGHT_EAR.value].y * height])

                LS = np.array([lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                               lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y])

                RS = np.array([lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                               lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y])

                # Angle between ears
                dx, dy = LE[0] - RE[0], LE[1] - RE[1]
                angle_deg = np.degrees(np.arctan2(dy, dx))

                # Angle between shoulders
                sdx, sdy = LS[0] - RS[0], LS[1] - RS[1]
                shoulder_deg = np.degrees(np.arctan2(sdy, sdx))

                # Final head tilt angle (relative to shoulders)
                head_tilt = angle_deg - shoulder_deg

                # Draw angle
                cv2.putText(image, f"Tilt: {head_tilt:.1f}°", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Draw ear dots
                cv2.circle(image, tuple(LE.astype(int)), 5, (0, 255, 0), -1)
                cv2.circle(image, tuple(RE.astype(int)), 5, (0, 255, 0), -1)

                # --- REP LOGIC ---
                if head_tilt > 0:
                    max_right = max(max_right, head_tilt)
                else:
                    max_left = min(max_left, head_tilt)

                if direction == "neutral":
                    if head_tilt > ANGLE_THRESHOLD:
                        direction = "right"
                    elif head_tilt < -ANGLE_THRESHOLD:
                        direction = "left"

                elif direction == "right":
                    if abs(head_tilt) < NEUTRAL_THRESHOLD:
                        direction = "neutral"

                elif direction == "left":
                    if abs(head_tilt) < NEUTRAL_THRESHOLD:
                        direction = "neutral"

                # Completed one left + one right → rep
                if max_right > ANGLE_THRESHOLD and max_left < -ANGLE_THRESHOLD and direction == "neutral":
                    rep_count += 1

                    # Store rep values
                    rep_max_left.append(max_left)
                    rep_max_right.append(max_right)
                    last_rep_left = max_left
                    last_rep_right = max_right
                    show_rep_until = time.time() + 3 # show for 3 seconds   
                    
                    print(f"Rep {rep_count} complete. L: {max_left:.1f}°  R: {max_right:.1f}°")

                    # Reset for next rep
                    max_left = 0
                    max_right = 0

                # Display rep count
                cv2.putText(image, f"Reps: {rep_count}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

                # STOP after 5 reps
                if rep_count >= MAX_REPS:
                    avg_left = sum(rep_max_left) / len(rep_max_left)
                    avg_right = sum(rep_max_right) / len(rep_max_right)

                    print("\n=== SESSION COMPLETE ===")
                    print(f"Average Left Tilt:  {avg_left:.1f}°")
                    print(f"Average Right Tilt: {avg_right:.1f}°")
                    print("=========================")

                    cv2.putText(image, "SESSION COMPLETE", (20, 140),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(image, f"Avg L: {avg_left:.1f}°", (20, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(image, f"Avg R: {avg_right:.1f}°", (20, 220),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                    cv2.imshow("Mediapipe Feed", image)
                    cv2.waitKey(10000)
                    break
                # Show last rep’s max values for a few seconds
                if last_rep_left is not None and time.time() < show_rep_until:
                    cv2.putText(image, f"Max Left: {last_rep_left:.1f}°", (20, 300),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                    cv2.putText(image, f"Max Right: {last_rep_right:.1f}°", (20, 350),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

            except Exception as e:
                print("Error:", e)

        else:
            cv2.putText(image, "No pose detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
