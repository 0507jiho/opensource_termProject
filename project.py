import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection

image_right_eye = cv2.imread('eye.png')
image_right_eye = cv2.resize(image_right_eye, (100, 63))

image_left_eye = cv2.imread('eye.png')
image_left_eye = cv2.resize(image_left_eye, (100, 63))

# Load the video file
cap = cv2.VideoCapture('foreigner_interview.mp4')

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.7) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        
        # Convert the image to RGB for Mediapipe processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image_rgb)

        if results.detections:
            for detection in results.detections:
                # mp_drawing.draw_detection(image, detection)
                
                # Retrieve the position of a specific feature among the 2 characteristics.
                keypoints = detection.location_data.relative_keypoints
                right_eye = keypoints[0]
                left_eye = keypoints[1]

                h, w, _ = image.shape
                right_eye_pos = (int(right_eye.x * w)), (int(right_eye.y * h))
                left_eye_pos = (int(left_eye.x * w)), (int(left_eye.y * h))

                image[right_eye_pos[1]-31:right_eye_pos[1]+32, right_eye_pos[0]-50:right_eye_pos[0]+50] = image_right_eye
                image[left_eye_pos[1]-31:left_eye_pos[1]+32, left_eye_pos[0]-50:left_eye_pos[0]+50] = image_left_eye

        # Show the image with face detection
        cv2.imshow('Face Detection with Eyes', image)

         # Check for 'a' key press to exit the loop
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the video capture and destroy any open CV windows
    cap.release()
    cv2.destroyAllWindows() 
