import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Load the video file
cap = cv2.VideoCapture('face_video.mp4')

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
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
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)

        # Show the image with face detection
        cv2.imshow('MediaPipe Face Detection', cv2.resize(image, None, fx = 0.5, fy = 0.5))

        # Check for 'a' key press to exit the loop
        if cv2.waitKey(1) == ord('q'):
            break

# Release the video capture and destroy any open CV windows
cap.release()
cv2.destroyAllWindows()
