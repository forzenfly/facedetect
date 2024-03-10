import cv2

# Load the pre-trained cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the pre-trained cascade classifier for left eye detection
left_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')

# Load the pre-trained cascade classifier for right eye detection
right_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')

# Initialize video capture
cap = cv2.VideoCapture(0)

# Continuously capture frames from the webcam
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Get the region of interest (ROI) for the detected face
        roi_gray = gray[y:y+h, x:x+w]

        # Perform left eye detection within the region of interest
        left_eyes = left_eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in left_eyes:
            # Draw a rectangle around the left eye
            cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)

        # Perform right eye detection within the region of interest
        right_eyes = right_eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in right_eyes:
            # Draw a rectangle around the right eye
            cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)

        # Calculate the coordinates of the left and right sides of the face
        left_side_x = x + left_eyes[0][0] + left_eyes[0][2] if len(left_eyes) > 0 else x
        right_side_x = x + right_eyes[0][0] if len(right_eyes) > 0 else x + w

        # Print the coordinates of the left and right sides of the face
        print("Left side of the face:", (left_side_x, y))
        print("Right side of the face:", (right_side_x, y))

    # Display the resulting frame
    cv2.imshow('Face Detection', frame)

    # Check for 'q' key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
