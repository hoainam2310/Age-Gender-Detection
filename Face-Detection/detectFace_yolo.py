import cv2
import numpy as np
net = cv2.dnn.readNetFromDarknet('file_yolo\yolov4-custom.cfg', 'file_yolo\yolov4-custom_last.weights')

classes = []
with open("file_yolo\yolov3.txt", "r") as f:
    classes = f.read().splitlines()
cap = cv2.VideoCapture(0)
while(cap.isOpened()):
    img = cv2.imread('image\image7.jpg', 1)
    #_,img = cap.read()
    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

    if len(indexes)>0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            if round(confidences[i],2) > 0:
                confidence = str(round(confidences[i],2))
                if x < 0:
                    x = 0
                if y < 0:
                    y = 0
                cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
                
                cv2.putText(img, confidence, (x, y+20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 2)
    cv2.imshow('image', img)
    cv2.imwrite('detected(yolo)\image7.jpg', img)
    c = cv2.waitKey(1) 
    if c == 27: 
        break 
cv2.destroyAllWindows()
cap.release()