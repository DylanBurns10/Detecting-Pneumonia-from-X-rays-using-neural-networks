# Importing python libraries
import cv2 
import os

import shutil, os
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix


# Creating a method to obtain the data, formatting it (e.g. setting image size) and adding labels to it
labels = ['PNEUMONIA', 'NORMAL']
img_size = 150
def get_training_data(data_dir):
    data = []
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)


# Loading the test, training and validation datasets for the neural network
train = get_training_data('D:/Downloads/Temp/Actual/chest_xray/train') # Obtaining the Pnemonia training data and setting it to the ndarray 'train', by using the method defined above
test = get_training_data('D:/Downloads/Temp/Actual/chest_xray/test') # Obtaining the Pnemonia test data and setting it to the ndarray 'test', by using the method defined above
val = get_training_data('D:/Downloads/Temp/Actual/chest_xray/val') # Obtaining the Pnemonia validation data and setting it to the ndarray 'val', by using the method defined above

#Creating a dataframe for easy visualisation via the graph method below
train_df = pd.DataFrame(train, columns=['image', 'label'])
test_df = pd.DataFrame(test, columns = ['image', 'label'])

# Creating and plotting a graph using the data frames to visualise the amount Pnemonia and normal x-rays data within each of the training and test datasets
# plt.figure(figsize=(18, 8))
# sns.set_style("darkgrid")

# plt.subplot(1,2,1)
# sns.countplot(train_df['label'], palette = 'coolwarm')
# plt.title('Train data')

# plt.subplot(1,2,2)
# sns.countplot(test_df['label'], palette = "hls")
# plt.title('Test data')

# plt.show()

# l = []
# for i in train:
#     if(i[1] == 0):
#         l.append("Pneumonia")
#     else:
#         l.append("Normal")
# sns.set_style('darkgrid')
# sns.countplot(l)

# plt.figure(figsize = (5,5))
# plt.imshow(train[0][0], cmap='gray')
# plt.title(labels[train[0][1]])

# plt.figure(figsize = (5,5))
# plt.imshow(train[-1][0], cmap='gray')
# plt.title(labels[train[-1][1]])

# d = []
# for i in test:
#     if(i[1] == 0):
#         d.append("Pneumonia")
#     else:
#         d.append("Normal")
# sns.set_style('darkgrid')
# sns.countplot(d)

# plt.figure(figsize = (5,5))
# plt.imshow(test[0][0], cmap='gray')
# plt.title(labels[test[0][1]])

# plt.figure(figsize = (5,5))
# plt.imshow(test[-1][0], cmap='gray')
# plt.title(labels[test[-1][1]])

# Creating empty arrays for the training, validation and test datasets
x_train = []
y_train = []

x_test = []
y_test = []

x_val = []
y_val = []

# Using for loops to append 'feature' and 'label' to the end of the training, test and validation arrays 
for feature, label in train:
    x_train.append(feature)
    y_train.append(label)

for feature, label in test:
    x_test.append(feature)
    y_test.append(label)
    
for feature, label in val:
    x_val.append(feature)
    y_val.append(label)

# Performing grayscale normalization, to the training, test and validation arrays, to reduce the effect of illumination's differences in the x-rays. Then converting the arrays in numpy arrays

x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255
x_test = np.array(x_test) / 255

x_train = x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)

x_val = x_val.reshape(-1, img_size, img_size, 1)
y_val = np.array(y_val)

x_test = x_test.reshape(-1, img_size, img_size, 1)
y_test = np.array(y_test)

# Creating a method in which to be bale to randomise some effects of the images to provide a better learning experience to the neural network (e.g. image rotation)
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range = 69,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.5, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True,  # randomly flip images
        vertical_flip=True)  # randomly flip images

datagen.fit(x_train) # Fitting the training data to the above 

# Creating a sequential neural network
model = Sequential()
model.add(Conv2D(32 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (150,150,1)))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(64 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(128 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(256 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Flatten())
model.add(Dense(units = 128 , activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(units = 1 , activation = 'sigmoid'))
model.compile(optimizer = "rmsprop" , loss = 'binary_crossentropy' , metrics = ['accuracy'])
model.summary()

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.3, min_lr=0.000001)

history = model.fit(datagen.flow(x_train,y_train, batch_size = 32) ,epochs = 12 , validation_data = datagen.flow(x_val, y_val) ,callbacks = [learning_rate_reduction])

print("Loss of the model is - " , model.evaluate(x_test,y_test)[0])
print("Accuracy of the model is - " , model.evaluate(x_test,y_test)[1]*100 , "%")

epochs = [i for i in range(12)]
fig , ax = plt.subplots(1,2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
fig.set_size_inches(20,10)

ax[0].plot(epochs , train_acc , 'go-' , label = 'Training Accuracy')
ax[0].plot(epochs , val_acc , 'ro-' , label = 'Validation Accuracy')
ax[0].set_title('Training & Validation Accuracy')
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")

ax[1].plot(epochs , train_loss , 'g-o' , label = 'Training Loss')
ax[1].plot(epochs , val_loss , 'r-o' , label = 'Validation Loss')
ax[1].set_title('Testing Accuracy & Loss')
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Training & Validation Loss")
plt.show()

#predictions = model.predict(x_test).astype('int32')
predictions = (model.predict(x_test) > 0.5).astype("int32")
predictions = predictions.reshape(1,-1)[0]



print(classification_report(y_test, predictions, target_names = ['Pneumonia (Class 0)','Normal (Class 1)']))

cm = confusion_matrix(y_test,predictions)

cm = pd.DataFrame(cm , index = ['0','1'] , columns = ['0','1'])

plt.figure(figsize = (10,10))
sns.heatmap(cm,cmap= "Blues", linecolor = 'black' , linewidth = 1 , annot = True, fmt='',xticklabels = labels,yticklabels = labels)
plt.show()

correct = np.nonzero(predictions == y_test)[0]
incorrect = np.nonzero(predictions != y_test)[0]

new_xtest = []
new_ytest = []
new_xtest = np.array(new_xtest) / 255

for k in incorrect[:]:
    if predictions[k] != y_test[k]:
        new_xtest = np.append(new_xtest ,x_test[k])
        new_ytest = np.append(new_ytest, y_train[k])

new_xtest = np.append(new_xtest, x_train)
new_ytest = np.append(new_ytest, y_train)
new_xtest = new_xtest.reshape(-1, img_size, img_size, 1)
new_ytest = np.array(new_ytest)  

model.compile(optimizer = "rmsprop" , loss = 'binary_crossentropy' , metrics = ['accuracy'])

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.3, min_lr=0.0000001)

history2 = model.fit(datagen.flow(new_xtest,new_ytest, batch_size = 32) ,epochs = 12 , validation_data = datagen.flow(x_val, y_val) ,callbacks = [learning_rate_reduction])

print("Loss of the model is - " , model.evaluate(x_test, y_test)[0])
print("Accuracy of the model is - " , model.evaluate(x_test, y_test)[1]*100 , "%")

epochs = [i for i in range(12)]
fig , ax = plt.subplots(1,2)
train_acc = history2.history['accuracy']
train_loss = history2.history['loss']
val_acc = history2.history['val_accuracy']
val_loss = history2.history['val_loss']
fig.set_size_inches(20,10)

ax[0].plot(epochs , train_acc , 'go-' , label = 'Training Accuracy')
ax[0].plot(epochs , val_acc , 'ro-' , label = 'Validation Accuracy')
ax[0].set_title('Training & Validation Accuracy')
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")

ax[1].plot(epochs , train_loss , 'g-o' , label = 'Training Loss')
ax[1].plot(epochs , val_loss , 'r-o' , label = 'Validation Loss')
ax[1].set_title('Testing Accuracy & Loss')
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Training & Validation Loss")
plt.show()


i = 0
for c in correct[:6]:                                                                           
    plt.subplot(3,2,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[c].reshape(150,150), cmap="gray", interpolation='none')
    plt.title("Predicted Class {},Actual Class {}".format(predictions[c], y_test[c]))
    plt.tight_layout()
    i += 1
plt.show()


i = 0
for c in incorrect[:6]:
    plt.subplot(3,2,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[c].reshape(150,150), cmap="gray", interpolation='none')
    plt.title("Predicted Class {},Actual Class {}".format(predictions[c], y_test[c]))
    plt.tight_layout()
    i += 1
plt.show()


# source = r'D:\Downloads\Temp\Actual\chest_xray\test\NORMAL'
# destination = r'D:\Downloads\Temp\Actual\chest_xray\Copy'
# filearray = [f for f in listdir(source) if isfile(join(source, f))]
# if not os.path.exists(destination):
#     os.mkdir(destination)
# for f in filearray:
#     if filearray.index(f) == np.nonzero(predictions == y_test).index(f):
#         shutil.copy(source + "\\" + f, destination)
