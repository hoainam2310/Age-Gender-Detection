import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D
from PIL import Image
import os

df_train = pd.read_csv('C:\\GD\\train.csv')
df_val = pd.read_csv('C:\\GD\\valid.csv')
df_test = pd.read_csv('C:\\GD\\test.csv')

images_feature_train = []
for i in range(20000):
  img = Image.open(df_train['link'][i]).convert('L')
  img_resized = img.resize((128, 128))
  images_feature_train.append(np.asarray(img_resized).ravel())
  print('Processed train', i)
  os.system('cls')

images_feature_val = []
for i in range(4000):
  img = Image.open(df_val['link'][i]).convert('L')
  img_resized = img.resize((128, 128))
  images_feature_val.append(np.asarray(img_resized).ravel())
  print('Processed valid', i)
  os.system('cls')

images_feature_test = []
for i in range(4000):
  img = Image.open(df_test['link'][i]).convert('L')
  img_resized = img.resize((128, 128))
  images_feature_test.append(np.asarray(img_resized).ravel())
  print('Processed test', i)
  os.system('cls')


########## NORMALIZE AND TRANSFORM DATA ##########

y_train = df_train['label'][:20000]
y_train = np.array(y_train)
X_train = np.array(images_feature_train)
X_train = (X_train.astype('float32')) / 255

y_val = df_val['label'][:4000]
y_val = np.array(y_val)
X_val = np.array(images_feature_val)
X_val = (X_val.astype('float32')) / 255

y_test = df_test['label'][:4000]
y_test = np.array(y_test)
X_test = np.array(images_feature_test)
X_test = (X_test.astype('float32')) / 255

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

"""C0 = 0
gamma0 = 0
accuracy0 = 0
kernel0 = ''
print('Find the best parameter...')
for c_para in [1, 5, 9, 10]:
    for gamma_para in [1, 0.1, 0.01, 0.001]:
        for kernel_para in ['linear']:
            model = SVC(C=c_para, gamma=gamma_para, kernel=kernel_para)
            model.fit(X_train, y_train)
            predictions = model.predict(X_val)
            acc = accuracy_score(y_val, predictions)
            if acc > accuracy0:
                C0 = c_para
                gamma0 = gamma_para
                kernel0 = kernel_para
                accuracy0 = acc"""

os.system('cls')
#print('\nSVM with C =', C0, 'gamma = ', gamma0, 'kernel = ', kernel0, '\n')
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# save the model to disk
import pickle
filename = 'model//gen_model_SVM.sav'
pickle.dump(model, open(filename, 'wb'))

predictions = model.predict(X_test)
from sklearn.metrics import accuracy_score
print('* ACCURACY: ', round(accuracy_score(y_test, predictions)*100, 3), '%')