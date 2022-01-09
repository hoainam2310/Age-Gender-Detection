import pandas as pd
import numpy as np
from PIL import Image
import os
from tensorflow.keras.models import load_model

df_test = pd.read_csv('C:\\GD\\test.csv')

images_feature_test = []
for i in range(4000):
  img = Image.open(df_test['link'][i]).convert('L')
  img_resized = img.resize((128, 128))
  images_feature_test.append(np.asarray(img_resized))
  print('Processed test', i)
  os.system('cls')

########## NORMALIZE AND TRANSFORM DATA ##########

y_test = df_test['label'][:4000]
X_test = np.asarray(images_feature_test)

model_loaded = load_model('model\\gen_model_CNN.h5')
y_pred = model_loaded.predict(X_test)
y_pred = y_pred.tolist()

predictions = []
for i in y_pred:
  predictions.append(i.index(max(i)))

from sklearn.metrics import accuracy_score
os.system('cls')
print(round(accuracy_score(y_test, predictions)*100, 3), '%')