from ffpyplayer.player import MediaPlayer
import cv2

video_path = 'foreigner_interview.mp4'

# Open the video file
cap = cv2.VideoCapture(video_path)

# Initialize the ffpyplayer MediaPlayer
player = MediaPlayer(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Get the current audio frame
    audio_frame, val = player.get_frame()

    # Check for 'eof' (end of file)
    if val != 'eof' and audio_frame is not None:
        # Process or play the audio frame as needed
        pass

    # Show the video frame with face detection
    cv2.imshow('MediaPipe Face Detection', cv2.resize(frame, None, fx=0.5, fy=0.5))

    # Check for 'q' key press to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture and destroy any open CV windows
cap.release()
cv2.destroyAllWindows()
