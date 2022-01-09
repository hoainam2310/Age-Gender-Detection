import os
from mtcnn import MTCNN
import numpy as np
from tensorflow.keras.models import load_model
from skimage.feature import canny
import cv2
import pickle
os.system('cls')

model_gender = load_model('model\gen_model_CNN.h5')
model_age = pickle.load(open('model\gen_model_SVM.sav', 'rb'))

detector = MTCNN()

def features_grid(img):
    features = np.array([], dtype='uint8')
    
    for y in range(0, img.shape[0], 10):
        for x in range(0, img.shape[1], 10):
            section_img = img[y:y+10, x:x+10]         
            section_mean = np.mean(section_img)
            section_std = np.std(section_img)           
            features = np.append(features, [section_mean, section_std])
    
    return features

def extract_canny_edges1(img):

    all_imgs = np.zeros((1, 800), dtype='uint8')
   
    img = canny(img, sigma=0.9)
    img_features = features_grid(img)
    img_features = img_features.reshape(1, img_features.shape[0])
    all_imgs = np.append(all_imgs, img_features, axis=0)   
    all_imgs = all_imgs[1:]

    return all_imgs

cap = cv2.VideoCapture(0)
while(cap.isOpened()):
    #img = cv2.imread('*link to demo image*')
    _, img = cap.read()
    faces = detector.detect_faces(img)
    for face in faces:
        x, y, w, h = face['box']
        if w*h > 20000:
          x = x- 20
          w , h = w+30, h+20
        elif 1000 <= w*h <= 20000:
          x = x - 10
          w, h = w + 10, h + 10
        elif 5000 <= w*h < 10000:
          x = x - 10
          w, h = w + 10, h + 10
        else:
          x = x
          w,h= w,h
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        img_test = img[y:y+h, x:x+w]
        # PREDICT
        images_feature_test = []
        img_test = cv2.cvtColor(img_test, cv2.COLOR_RGB2GRAY)
        img_resized = cv2.resize(img_test, (200,200))

        images_feature_test.append(np.asarray(img_resized))   

        data_gender = np.asarray(images_feature_test)  
        y_pred_gender = model_gender.predict(data_gender.reshape((1,200,200,1)))
        if (y_pred_gender[0][1] == 1):
          re_gender = 'Male'
        elif (y_pred_gender[0][0] == 1):
          re_gender = 'Female'
        else:
          re_gender = 'Unknown'

        data_age = extract_canny_edges1(img_resized)
        predictions = model_age.predict(data_age)
        age = ['1-6', '7-22', '23-26', '27-33', '34-45', '46-61', '62-116']
        print(age[predictions[0]])
        cv2.putText(img, 'Gender: ' + re_gender + ' Age: ' + age[predictions[0]], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    
    cv2.imshow('face', img)
    cv2.imwrite('imgDetected\image10.jpg', img)
    c = cv2.waitKey(1)
    if c == 27:
      break 
cv2.destroyAllWindows()
cap.release()
