import cv2

# Create Local Binary Patterns Histograms for face recognization
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load the trained mode
recognizer.read('train/train.yml')

# Load prebuilt model for Frontal Face
cascadePath = "haarcascade_frontalface_default.xml"

# Create classifier from prebuilt model
faceCascade = cv2.CascadeClassifier(cascadePath)

# Set the font style
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize and start the video frame capture
cam = cv2.VideoCapture(0)

# Loop
while True:
    # Read the video frame
    ret, im =cam.read()

    # Convert the captured frame into grayscale
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    # Get all face from the video frame
    faces = faceCascade.detectMultiScale(gray, 1.2,5)

    # For each face in faces
    for(x,y,w,h) in faces:

        # Create rectangle around the face
        cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)

        # Recognize the face belongs to which ID
        Id,conf = recognizer.predict(gray[y:y+h,x:x+w])

        # Check the ID if exist 
        if(Id == 1):
            Id = "Shivji"
        if(Id == 2):
            Id = "Ganeshji"
        if(Id == 3):
            Id = "Hanumanji"
        if(Id == 4):
            Id = "Krishnaji"
            
        
        # Put text describe who is in the picture
        cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
        cv2.putText(im, str(Id), (x,y-40), font, 2, (255,255,255), 3)

    # Display the video frame with the bounded rectangle
    cv2.imshow('im',im) 

    # If 'q' is pressed, close program
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Stop the camera
cam.release()

# Close all windows
cv2.destroyAllWindows()