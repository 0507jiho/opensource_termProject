import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Load the video file
cap = cv2.VideoCapture('interview.mp4')

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.3) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # Convert the image to RGB for Mediapipe processing
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)


        image.flags.writeable = True
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if results.detections:
            # 6 characteristics: right eye, left eye, tip of the nose, center of the mouth, right ear, left ear
            for detection in results.detections:
                # mp_drawing.draw_detection(image, detection)
                
                # Retrieve the position of a specific feature among the six characteristics.
                keypoints = detection.location_data.relative_keypoints
                right_eye = keypoints[0]
                left_eye = keypoints[1]
                nose_tip = keypoints[2]

                h, w, _ = image.shape
                right_eye = (int(right_eye.x * w), int(right_eye.y * h))
                left_eye = (int(left_eye.x * w), int(left_eye.y * h))
                nose_tip = (int(nose_tip.x * w), int(nose_tip.y * h))

                cv2.circle(image, right_eye, 20, (255, 0, 0), 10, cv2.LINE_AA)
                cv2.circle(image, left_eye, 20, (0, 255, 0), 10, cv2.LINE_AA)
                cv2.circle(image, nose_tip, 40, (0, 255, 255), 10, cv2.LINE_AA)

        # Show the image with face detection
        cv2.imshow('MediaPipe Face Detection', cv2.resize(image, None, fx = 0.5, fy = 0.5))

        # Check for 'a' key press to exit the loop
        if cv2.waitKey(1) == ord('q'):
            break

# Release the video capture and destroy any open CV windows
cap.release()
cv2.destroyAllWindows()
