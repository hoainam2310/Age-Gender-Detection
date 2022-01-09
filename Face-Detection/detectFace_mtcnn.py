from mtcnn import MTCNN
import numpy as np
import cv2

detector = MTCNN()
img = cv2.imread('image\image11.jpg')
faces = detector.detect_faces(img)

    # loop through detections and draw them on transparent overlay image
for face in faces:
  x , y, w, h = face['box'] 
  confidence = face['confidence']
  cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 2)
  cv2.putText(img, str(confidence)[:4] ,(x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
  cv2.imshow('face', img)
  cv2.imwrite('detected(mtcnn)\image11.jpg', img)

cv2.waitKey(0) 
cv2.destroyAllWindows() 