import os
import keras
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import matplotlib.pyplot as plt
from scipy import ndimage


file_path = 'A_Z Handwritten Data.csv'

names = ['class']
for id in range(1,785):
    names.append(id)

df = pd.read_csv(file_path,header=None, names=names)
print(df.head())

class_mapping = {}
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
for i in range(len(alphabets)):
    class_mapping[i] = alphabets[i]
class_mapping

df['class'].map(class_mapping).unique()

y_full = df.pop('class')
x_full = df.to_numpy().reshape(-1,28,28, 1)

splitter = StratifiedShuffleSplit(n_splits=3,test_size=0.2)


for train_ids, test_ids in splitter.split(x_full, y_full):
    X_train_full, y_train_full = x_full[train_ids], y_full[train_ids].to_numpy()
    X_test, y_test = x_full[test_ids], y_full[test_ids].to_numpy()

X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.1)

plt.figure(figsize=(15, 8))
for i in range(1, 11):
    id = np.random.randint(len(X_train))
    image, label = tf.squeeze(X_train[id]), class_mapping[int(y_train[id])]

    plt.subplot(2, 5, i)
    plt.imshow(image, cmap='binary')
    plt.title(label)
    plt.axis('off')

plt.tight_layout()
plt.show()

#train

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(26, activation='sigmoid'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


model.fit(X_train, y_train, validation_data=(X_valid,y_valid), epochs=8)


model.save('handwritten.modelbreak270')
model = tf.keras.models.load_model('handwritten.modelbreak270')

#test
#rotation
X_test_rotated = np.zeros_like(X_test)
for i in range(len(X_test)):
    X_test_rotated[i] = ndimage.rotate(X_test[i], angle=90, reshape=False)

loss, accuracy = model.evaluate(X_test_rotated, y_test)
print('test_loss', loss)
print('test_accuracy',accuracy)



#prediction

plt.figure(figsize=(20, 20))
for i in range(1, 101):
    id = np.random.randint(len(X_test_rotated))
    image, label = X_test_rotated[id].reshape(28, 28), class_mapping[int(y_test[id])]
    pred = class_mapping[int(np.argmax(model.predict(image.reshape(-1, 28, 28, 1))))]

    plt.subplot(10, 10, i)
    plt.imshow(image, cmap='binary')
    plt.title(f"Org: {label}, Pred: {pred}")
    plt.axis('off')

plt.tight_layout()
plt.show()



# #LOSS AND ACCURACY (it is also up)
# loss, accuracy = model.evaluate(X_test, y_test)
# print('test_loss', loss)
# print('test_accuracy',accuracy)
