import cv2
import mediapipe as mp
import numpy as np
import math
import time


CALIBRATION_DURATION = 3.0  # seconds
scaling_extension = 1.8
scaling_flexion = 1.2
calibrating = True
calib_start_time = None
calib_sum = 0.0
calib_count = 0
baseline_angle = 0.0

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_side_view_angle(shoulder, hip, head):
    shoulder = np.array(shoulder)
    hip = np.array(hip)
    head = np.array(head)
    
    # Vector from hip to shoulder (torso vertical)
    torso_vec = shoulder - hip
    
    # Vector from shoulder to head (head direction)
    head_vec = head - shoulder
    
    # Normalize vectors
    torso_unit = torso_vec / np.linalg.norm(torso_vec)
    head_unit = head_vec / np.linalg.norm(head_vec)
    
    # Dot product and angle in degrees
    dot = np.dot(torso_unit, head_unit)
    angle = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
    
    # Determine sign by cross product to get positive for flexion (forward tilt)
    cross = torso_unit[0]*head_unit[1] - torso_unit[1]*head_unit[0]
    if cross < 0:
        angle = -angle
    
    return angle

def categorize_angle(avg_angle):
    """Categorize average angle: RED <35, YELLOW 35-45, GREEN >45"""
    if avg_angle < 35:
        return "RED", (0, 0, 255)  # Blue in BGR
    elif 35 <= avg_angle <= 45:
        return "YELLOW", (0, 255, 255)  # Cyan in BGR
    else:
        return "GREEN", (0, 255, 0)  # Green in BGR

# ---- REP PARAMETERS ----
FLEXION_THRESHOLD = 20     # must pass this to start a rep
NEUTRAL_THRESHOLD = 10     # must drop below this to finish the rep
MAX_REPS = 5

rep_count = 0
in_rep = False
current_rep_max = 0
current_rep_min=0
rep_max_angles = []
rep_min_angles =[]

show_rep_until = 0  # timestamp for showing rep result on-screen

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

            if calib_start_time is None:
                calib_start_time = time.time()
            current_time = time.time()

            # LEFT SIDE
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,
                    landmarks[mp_pose.PoseLandmark.NOSE.value].y]

            raw_angle = calculate_side_view_angle(shoulder, hip, nose)
        
        
            # ----- CALIBRATION PHASE -----
            if calibrating:
                if current_time - calib_start_time <= CALIBRATION_DURATION:
                    # accumulate raw forward-facing angle
                    calib_sum += raw_angle
                    calib_count += 1
                    neck_angle = 0.0  # show 0 during calibration
                    cv2.putText(image, "Calibrating... turn your body to the left",
                                (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 255, 255), 2)
                else:
                    # finish calibration
                    if calib_count > 0:
                        baseline_angle = calib_sum / calib_count
                    calibrating = False
                    print(f"Calibration complete. Baseline angle: {baseline_angle:.2f} deg")
                    neck_angle = 0.0
            else:
                # ----- POST-CALIBRATION: USE CHANGE FROM BASELINE -----
                neck_angle = raw_angle - baseline_angle
                
                if( neck_angle >0):
                    scaling = scaling_flexion
                else:
                    scaling = scaling_extension
                    
                neck_angle = neck_angle* scaling
                

            # ----- REP LOGIC -----
            if not calibrating:
                if not in_rep:
                    if neck_angle > FLEXION_THRESHOLD:
                        in_rep = True
                        current_rep_max = neck_angle
                    elif -neck_angle > FLEXION_THRESHOLD:
                        in_rep = True
                        current_rep_min = neck_angle 
                else:
                    current_rep_max = max(current_rep_max, neck_angle)
                    current_rep_min = min(current_rep_min,neck_angle)
                    if neck_angle < NEUTRAL_THRESHOLD and neck_angle > -NEUTRAL_THRESHOLD:
                        rep_count += 1
                        rep_max_angles.append(current_rep_max)
                        rep_min_angles.append(current_rep_min)
                        show_rep_until = time.time() + 3
                        print(f"Rep {rep_count} complete. Max Angle: {current_rep_max:.1f}°")
                        in_rep = False


            # DISPLAY
            cv2.putText(image, f"Neck Angle: {int(neck_angle)} deg",
                        (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0), 2)

            cv2.putText(image, f"Reps: {rep_count}/{MAX_REPS}",
                        (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 0), 2)
            cv2.putText(image, "Turn to your left",
                        (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 0), 2)

            # Show last rep's max angle for a few seconds after completion
            if time.time() < show_rep_until:
                cv2.putText(image, f"Rep {rep_count} Max: {current_rep_max:.1f}deg",
                            (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 255, 255), 2)
                cv2.putText(image, f"Rep {rep_count} Min: {current_rep_min:.1f}deg",
                            (40, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 255, 255), 2)

            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

        # After 5 reps — show average max angle
        if rep_count >= MAX_REPS:
            avg_max_angle = sum(rep_max_angles) / len(rep_max_angles)
            avg_min_angle = sum(rep_min_angles)/len(rep_min_angles)
            flexion_category, flexion_color = categorize_angle(avg_max_angle)
            extension_category, extension_color = categorize_angle(-avg_min_angle)
            
            cv2.putText(image, "SESSION COMPLETE!",
            (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
            (0, 0, 255), 2)

            # Flexion block (left)
            cv2.putText(image, f"Avg Max Angle: {avg_max_angle:.1f}deg",
                        (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        flexion_color, 2)
            cv2.putText(image, f"Rating: {flexion_category}",
                        (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        flexion_color, 2)

            # Extension block (right)
            cv2.putText(image, f"Avg Min Angle: {avg_min_angle:.1f}deg",
                        (400, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        extension_color, 2)
            cv2.putText(image, f"Rating: {extension_category}",
                        (400, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        extension_color, 2)

            # Legend at bottom
            cv2.putText(image, "Legend: RED <35  YELLOW 35-45  GREEN >45",
                        (40, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (200, 200, 200), 1)


        cv2.imshow('Neck Flexion/Extension', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
