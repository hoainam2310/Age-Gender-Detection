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
for i in range(200):
  img = Image.open(df_train['link'][i]).convert('L')
  img_resized = img.resize((128, 128))
  images_feature_train.append(np.asarray(img_resized))
  print('Processed train', i)
  os.system('cls')

images_feature_val = []
for i in range(40):
  img = Image.open(df_val['link'][i]).convert('L')
  img_resized = img.resize((128, 128))
  images_feature_val.append(np.asarray(img_resized))
  print('Processed valid', i)
  os.system('cls')

images_feature_test = []
for i in range(40):
  img = Image.open(df_test['link'][i]).convert('L')
  img_resized = img.resize((128, 128))
  images_feature_test.append(np.asarray(img_resized))
  print('Processed test', i)
  os.system('cls')


########## NORMALIZE AND TRANSFORM DATA ##########

def one_hot(y):
  oh = np.zeros((y.shape[0], y.max()+1))
  oh[np.arange(y.shape[0]), y] = 1
  return oh

y_train = df_train['label'][:200]
y_train = one_hot(np.array(y_train))
X_train = np.array(images_feature_train)
X_train = (X_train.astype('float32')) / 255
X_train = X_train.reshape((X_train.shape[0], 128, 128, 1))
print(X_train.shape)
print(y_train)

y_val = df_val['label'][:40]
y_val = one_hot(np.array(y_val))
X_val = np.array(images_feature_val)
X_val = (X_val.astype('float32')) / 255
X_val = X_val.reshape((X_val.shape[0], 128, 128, 1))

y_test = df_test['label'][:40]
y_test = one_hot(np.array(y_test))
X_test = np.array(images_feature_test)
X_test = (X_test.astype('float32')) / 255
X_test = X_test.reshape((X_test.shape[0], 128, 128, 1))

del images_feature_train
del images_feature_test
del images_feature_val

# create model.
model = Sequential()

model.add(Conv2D(16, kernel_size=(3, 3), input_shape = (128, 128, 1), activation = 'relu',padding='valid'))

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(32, kernel_size=(3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(32, kernel_size=(3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())

model.add(Dense(units = 128, activation = 'relu'))
model.add(Dense(units = 2, activation = 'softmax'))
model.summary()

# set compole parameter optimizer algorthem, loss function and accuracy.
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# fit model on train data.

H = model.fit(X_train, y_train, steps_per_epoch=10, epochs=10, validation_data=(X_val, y_val), validation_steps=20, verbose=2)

model.save('model\\gen_model_CNN.h5')

pd.DataFrame(H.history).plot()

# calculate test accuracy and loss.
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print('Test Accuracy:', round(test_acc*100, 2),"%")
print('Test Loss:', test_loss)

plt.plot(H.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()