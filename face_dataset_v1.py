import cv2
import time

# Detect object in video stream using Haarcascade Frontal Face
face_detector= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Start capturing video 
cap = cv2.VideoCapture(0)

time.sleep(2)

# For each Subject, one face id
face_id =  input("Please enter the id_number ")

# Initialize sample face image
count = 0

while(True):
    # Capture video frame
    ret, img = cap.read()
    # Convert frame to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect frames of different sizes, list of faces rectangles
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
    
    if(len(faces)!=0):
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            # Increment sample face image
            count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite("datasets/Subject." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        # Display the video frame, with bounded rectangle on the person's face
        cv2.imshow('frame',img)
        # To stop taking video, press 'q' for at least 100ms
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # If image taken reach 100 , stop taking video
    elif count>=100:
        break
# Stop video    
cap.release()
# Close all started windows
cv2.destroyAllWindows()