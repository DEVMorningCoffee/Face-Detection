import cv2 
import os
from pathlib import Path


# Algo
haar_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

# Capture WebCam Video
cam = cv2.VideoCapture(0)

while True:
    _, frame = cam.read()
    text = "Face Not Detected"

    # detect faces using Haar Cascade 
    grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = haar_cascade.detectMultiScale(grayImg, 1.3, 5)

    for (x, y, w, h) in face:
        text = "Face Detected"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    # display the text on the image
    print(text)
    image = cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)


    # display the output window and press escape key to exit
    cv2.imshow("Face Detection", image)
    key = cv2.waitKey(10)
 
    if key == 27:
        break


cam.release()
cv2.destroyAllWindows()