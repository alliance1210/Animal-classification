from tensorflow import keras
from tensorflow.keras import layers
import os, warnings
import matplotlib.pyplot as plt
from matplotlib import gridspec
from PIL import Image, ImageOps
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix,classification_report
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras

from keras.layers import Dropout, Dense, Flatten, BatchNormalization, LeakyReLU, GlobalAveragePooling2D, Conv2D, MaxPool2D
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.applications.vgg16 import VGG16


import cv2

from tqdm import tqdm
import os
import gc

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings("ignore")


images = []
labels = []

main_directory = 'animals/animals'

for animal in tqdm(os.listdir(main_directory)):
    for i in range(len(os.listdir(main_directory + '/' + animal))):
        if i < 300:
            img = cv2.imread(main_directory + '/' + animal + '/' + os.listdir(main_directory + '/' + animal)[i])
            resized_img = cv2.resize(img,(224,224))
            resized_img = resized_img / 255.0
            images.append(resized_img)
            labels.append(animal)

images = np.array(images,dtype = 'float32')
le = preprocessing.LabelEncoder()
le.fit(labels)
class_names = le.classes_
labels = le.transform(labels)

labels = np.array(labels, dtype = 'uint8')
labels = np.resize(labels, (len(labels),1))

train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.33, stratify = labels)

n = np.random.randint(0,np.shape(train_images)[0])

# plt.imshow(train_images[n])
# plt.title(str(le.inverse_transform([train_labels[n]])))
# plt.show()


vgg_model = Sequential()

vgg_base_model = VGG16(include_top = False, weights="imagenet", input_shape = (224,224,3))
print(f'Number of layers in VGG16 : {len(vgg_base_model.layers)}')

vgg_base_model.trainable = False

# for layer in vgg_base_model.layers[:15]:
#     layer.trainable = False

vgg_model.add(vgg_base_model)

vgg_model.add(GlobalAveragePooling2D())

vgg_model.add(Dense(units = 90, activation = 'softmax'))

vgg_model.summary()

early_stopping = EarlyStopping( monitor = 'val_accuracy', mode = 'max', min_delta = 1,patience = 20,restore_best_weights = True,verbose = 0)

# Compile
vgg_model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.01) , loss = 'sparse_categorical_crossentropy' , metrics = ['accuracy'])

# Train
vgg = vgg_model.fit(train_images, train_labels, batch_size = 32, epochs = 5 , callbacks = [early_stopping], validation_split = 0.2)

# read = Image.open('animals/animals/cats/cats_00002.jpg')
# size = (224, 224)
# read = ImageOps.fit(read, size, Image.ANTIALIAS)
# data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
# image_array = np.asarray(read)
# normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
# data[0] = normalized_image_array

# vgg_predictions = vgg_model.predict(data)[0]
# vgg_predictions = np.argmax(vgg_predictions,axis = 0)
# # n = np.random.randint(0,np.shape(preimg)[0])
# print([vgg_predictions])
# plt.imshow(read)
# plt.title(  "VGG16 Model's Predicted Animal : " + str(le.inverse_transform([vgg_predictions])))
# plt.show()


vgg_predictions = vgg_model.predict(test_images)
vgg_predictions = np.argmax(vgg_predictions,axis = 1)

# n = np.random.randint(0,np.shape(test_images)[0])
# plt.imshow(test_images[n])
# plt.title(  "Predicted Animal : " + str(le.inverse_transform([vgg_predictions[n]])))
# plt.show()
# -----------------------------------------------------ALEX----------------------------------------------------------

alex_model = Sequential()

alex_model.add(Conv2D(filters = 96, kernel_size = (11,11), strides = (4,4), activation = 'relu', input_shape = (224,224,3)))
alex_model.add(BatchNormalization())
alex_model.add(MaxPool2D(pool_size = (3,3), strides = (2,2)))

alex_model.add(Conv2D(filters = 256, kernel_size = (5,5), strides = (1,1), activation = 'relu', padding = 'same'))
alex_model.add(BatchNormalization())
alex_model.add(MaxPool2D(pool_size = (3,3), strides = (2,2)))

alex_model.add(Conv2D(filters = 384, kernel_size = (3,3), strides = (1,1), activation = 'relu', padding = 'same'))
alex_model.add(BatchNormalization())

alex_model.add(Conv2D(filters = 384, kernel_size = (3,3), strides = (1,1), activation = 'relu', padding = 'same'))
alex_model.add(BatchNormalization())

alex_model.add(Conv2D(filters = 256, kernel_size = (3,3), strides = (1,1), activation = 'relu', padding = 'same'))
alex_model.add(BatchNormalization())
alex_model.add(MaxPool2D(pool_size = (3,3), strides = (2,2)))

alex_model.add(Flatten())

alex_model.add(Dense(units = 4096, activation = 'relu'))
alex_model.add(Dropout(0.5))

alex_model.add(Dense(units = 4096, activation = 'relu'))
alex_model.add(Dropout(0.5))

alex_model.add(Dense(units = 90, activation = 'softmax'))

early_stopping = EarlyStopping( monitor = 'val_accuracy', mode = 'max', min_delta = 1,patience = 20,restore_best_weights = True,verbose = 0)

# Compile
alex_model.compile(optimizer = "adam" , loss = 'sparse_categorical_crossentropy' , metrics = ['accuracy'])

alex_model.summary()

# Train
alex = alex_model.fit(train_images, train_labels, batch_size = 32, epochs = 20, callbacks = [early_stopping], validation_split = 0.2)
alex_predictions = alex_model.predict(test_images)
alex_predictions = np.argmax(alex_predictions,axis = 1)
# ---------------------------------------------------------------end--------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------end--------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------end--------------------------------------------------------------------------------------------------


n = np.random.randint(0,np.shape(test_images)[0])

plt.imshow(test_images[n])
plt.title( "Alexnet Model's Predicted Animal : " + str(le.inverse_transform([alex_predictions[n]])) + '\n' + "VGG16 Model's Predicted Animal : " + str(le.inverse_transform([vgg_predictions[n]])))
plt.show()


vgg_cm = confusion_matrix(test_labels, vgg_predictions)
alex_cm = confusion_matrix(test_labels, alex_predictions)

plt.figure(figsize = (30,10))
plt.subplot(2,2,2)
sns.heatmap(alex_cm,cmap = 'Blues',annot = True, xticklabels = class_names, yticklabels = class_names)
plt.subplot(2,2,1)
sns.heatmap(vgg_cm,cmap = 'Blues',annot = True, xticklabels = class_names, yticklabels = class_names)

plt.show()


plt.figure(figsize = (20,10))

plt.subplot(2,2,2)
x_train_acc = alex.history['accuracy']
x_val_acc = alex.history['val_accuracy']
x_epoch = [i for i in range(len(x_val_acc))]
plt.plot(x_epoch , x_train_acc , 'go-' , label = 'Training Accuracy')
plt.plot(x_epoch , x_val_acc , 'ro-' , label = 'Validation Accuracy')
plt.title('Training & Validation Accuracy for AlexNet')
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Accuracy")

plt.subplot(2,2,1)
x_train_acc = vgg.history['accuracy']
x_val_acc = vgg.history['val_accuracy']
x_epoch = [i for i in range(len(x_val_acc))]
plt.plot(x_epoch , x_train_acc , 'go-' , label = 'Training Accuracy')
plt.plot(x_epoch , x_val_acc , 'ro-' , label = 'Validation Accuracy')
plt.title('Training & Validation Accuracy for vgg16')
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Accuracy")

plt.show()


def scores(cm):
    p = {}
    r = {}
    f1 = {}
    
    for i in range(len(cm)):
        p[i] = cm[i,i] / sum(cm[:,i])
        r[i] = cm[i,i] / sum(cm[i,:])
        f1[i] = 2 * (cm[i,i] / sum(cm[:,i])) * (cm[i,i]/sum(cm[i,:])) / ((cm[i,i] / sum(cm[:,i])) + (cm[i,i]/sum(cm[i,:])))
    
    return p,r,f1 

v_p,v_r,v_f1 = scores(vgg_cm)
a_p,a_r,a_f1 = scores(alex_cm)
Precision = {
    'VGG16 Precision' : v_p,
    'AlexNet Precision' : a_p
}

Precision = pd.DataFrame(Precision)
print(Precision)

Recall = {
    
    'VGG16 Recall' : v_r,
    'AlexNet Precision' : a_r
}

Recall = pd.DataFrame(Recall)

print(Recall)

F1_score = {
   
    'VGG16 F1_score' : v_f1,'AlexNet Precision' : a_f1
}

F1_score = pd.DataFrame(F1_score)
print(F1_score)