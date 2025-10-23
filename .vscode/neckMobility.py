import cv2
import mediapipe as mp
import numpy as np

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    dot_product = np.dot(ba, bc)
    magnitude_ba = np.linalg.norm(ba)
    magnitude_bc = np.linalg.norm(bc)
    cosine_angle = dot_product / (magnitude_ba * magnitude_bc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip image horizontally for better selfie view and convert to RGB
        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        results = pose.process(image)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Draw pose landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,
                        landmarks[mp_pose.PoseLandmark.NOSE.value].y,
                        landmarks[mp_pose.PoseLandmark.NOSE.value].z]
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z]
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z]
                
                mid_shoulder = [(left_shoulder[0] + right_shoulder[0]) / 2,
                                (left_shoulder[1] + right_shoulder[1]) / 2]
                
                height, width = image.shape[:2]
                
                nose_pixel = tuple(np.multiply(nose[:2], [width, height]).astype(int))
                left_shoulder_pixel = tuple(np.multiply(left_shoulder[:2], [width, height]).astype(int))
                mid_shoulder_pixel = tuple(np.multiply(mid_shoulder, [width, height]).astype(int))
                
                # Calculate angle between vectors
                vector1 = np.array([nose_pixel[0] - mid_shoulder_pixel[0], nose_pixel[1] - mid_shoulder_pixel[1]])
                vector2 = np.array([left_shoulder_pixel[0] - mid_shoulder_pixel[0], left_shoulder_pixel[1] - mid_shoulder_pixel[1]])
                
                angle = np.degrees(np.arctan2(vector1[1], vector1[0]) - np.arctan2(vector2[1], vector2[0]))
                angle = abs(angle)
                if angle > 180:
                    angle = 360 - angle
                
                # Draw lines and angle arc
                cv2.line(image, mid_shoulder_pixel, nose_pixel, (255, 0, 0), 2)
                cv2.line(image, mid_shoulder_pixel, left_shoulder_pixel, (255, 0, 0), 2)
                
                # Arc drawing
                angle_radius = 30
                start_angle = np.degrees(np.arctan2(nose_pixel[1] - mid_shoulder_pixel[1],
                                                   nose_pixel[0] - mid_shoulder_pixel[0]))
                end_angle = np.degrees(np.arctan2(left_shoulder_pixel[1] - mid_shoulder_pixel[1],
                                                 left_shoulder_pixel[0] - mid_shoulder_pixel[0]))
                
                # Adjust angles for ellipse function
                if end_angle < start_angle:
                    start_angle, end_angle = end_angle, start_angle
                
                cv2.ellipse(image, mid_shoulder_pixel,
                            (angle_radius, angle_radius),
                            0, start_angle, end_angle, (255, 0, 0), 2)
                
                # Draw angle text
                angle_text = f"{int(angle)}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                text_size = cv2.getTextSize(angle_text, font, font_scale, thickness)[0]
                text_x = mid_shoulder_pixel[0] - 30
                text_y = mid_shoulder_pixel[1] - 20
                
                cv2.rectangle(image, 
                              (text_x - 5, text_y - text_size[1] - 5),
                              (text_x + text_size[0] + 5, text_y + 5),
                              (0, 0, 0), -1)
                cv2.putText(image, angle_text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
                
                print(f"Neck to left shoulder angle: {angle:.1f}°")
                
            except Exception as e:
                print(f"Error calculating angle: {str(e)}")
                cv2.putText(image, "Error calculating angle",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(image, "No pose detected",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('Mediapipe Feed', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
