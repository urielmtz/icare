# -*- coding: utf-8 -*-
"""
ICARE 0.1

Created on Thursday 29th June 2017

@author: Uriel Martinez-Hernandez
"""

# First try using data from walking activities generated in MATLAB

import scipy.io as sio
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm
import numpy.ma as ma

from PIL import Image                                                            
import glob


np.random.seed(7)


'''
function name: getNamePrediction
purpose: map from classes to names
parameters:
    listOfNames: list of labels/names of training classes
    classPredicted: classes predicted by the CNN model
output: return the name of a recognised class
'''
def getNamePrediction(listOfNames, classPredicted):
    outputPredicted = []
    
    for i in range(len(classPredicted)):
        #print(listOfNames[classPredicted[i]])
        outputPredicted.append(listOfNames[classPredicted[i]])

    return outputPredicted
    


namesList = ['uriel', 'fernando']

imageFolderPath = r'C:\Users\Uriel Martinez\Desktop\Photos'
imageFolderTrainingPath = imageFolderPath + r'\train'
imageFolderTestingPath = imageFolderPath + r'\validation'
imageTrainingPath = []
imageTestingPath = []

for i in range(len(namesList)):
    trainingLoad = imageFolderTrainingPath + '\\' + namesList[i] + '\*.JPG'
    testingLoad = imageFolderTestingPath + '\\' + namesList[i] + '\*.JPG'
    imageTrainingPath = imageTrainingPath + glob.glob(trainingLoad)
    imageTestingPath = imageTestingPath + glob.glob(testingLoad)
    
#
print(len(imageTrainingPath))
print(len(imageTestingPath))

updateImageSize = [128, 128]


tempImg = Image.open(imageTrainingPath[0]).convert('L')
tempImg.thumbnail(updateImageSize, Image.ANTIALIAS)
[imWidth, imHeight] = tempImg.size

x_train = np.zeros((len(imageTrainingPath), imHeight, imWidth, 1))
x_test = np.zeros((len(imageTestingPath), imHeight, imWidth, 1))

for i in range(len(x_train)):
    tempImg = Image.open(imageTrainingPath[i]).convert('L')
    tempImg.thumbnail(updateImageSize, Image.ANTIALIAS)
    x_train[i, :, :, 0] = np.array(tempImg, 'f')
    #x_train[i, :, :, 0] = np.array((Image.open(imageTrainingPath[i]).convert('L')).thumbnail(resizeImage, Image.ANTIALIAS), 'f')
    
for i in range(len(x_test)):
    tempImg = Image.open(imageTestingPath[i]).convert('L')
    tempImg.thumbnail(updateImageSize, Image.ANTIALIAS)
    x_test[i, :, :, 0] = np.array(tempImg, 'f')
    #x_test[i, :, :, 0] = np.array((Image.open(imageTestingPath[i]).convert('L')).thumbnail(resizeImage, Image.ANTIALIAS), 'f')


y_train = np.zeros((len(x_train),));
y_test = np.zeros((len(x_test),));


countPos = 0
for i in range(0, len(namesList)):
    for j in range(0, round(len(imageTrainingPath)/len(namesList))):
        y_train[countPos,] = i
        countPos = countPos + 1
    
countPos = 0
for i in range(0, len(namesList)):
    for j in range(0, round(len(imageTestingPath)/len(namesList))):
        y_test[countPos,] = i
        countPos = countPos + 1
        
y_train = keras.utils.to_categorical(y_train, len(namesList));
y_test = keras.utils.to_categorical(y_test, len(namesList));
        

model = Sequential();
model.add(Conv2D(32, (5, 5), activation='sigmoid', input_shape=(imHeight, imWidth, 1)));
#model.add(Conv2D(32, (3, 3), activation='sigmoid'));
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(16, (3, 3), activation='sigmoid'))
#model.add(Conv2D(16, (3, 3), activation='sigmoid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64, activation='sigmoid'))
model.add(Dropout(0.1))
model.add(Dense(2, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='sgd')
#model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

model.fit(x_train, y_train, batch_size=4, epochs=50)
score = model.evaluate(x_test, y_test, batch_size=4)
print("\n%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
output = model.predict(x_test, batch_size=1)


print(output)
print(np.argmax(output, axis=1))

# call function to map from classes to names
personRecognized = getNamePrediction(namesList, np.argmax(output, axis=1))
print(personRecognized)

model.predict_proba(x_test, batch_size=1, verbose=1)


print('OK')