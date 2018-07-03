import cv2
import sys
import numpy as np
import os
from numpy import *
from keras.models import load_model

img_rows, img_cols = 224, 224
channel = 1
num_classes = 2
model = load_model('simple_model_simpAug.h5')
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarCascade_eye.xml') #Put name of your haar Cascade classifier
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray,1.5,5)
    counter = 0
    for (x,y,w,h) in face:
        if(counter<2):
            roi = frame[y:y+h, x:x+w]
            roir = cv2.resize(roi,(224,224))
            roir1 = cv2.cvtColor(roir, cv2.COLOR_BGR2GRAY, 0)
            arr = array(roir1).flatten()
            x_test = np.array(arr)
            x_test = x_test.reshape(1, img_rows, img_cols, channel)
            x_test = x_test.astype('float32')
            x_test /=255
            y_pred = model.predict_classes(x_test)
            if(y_pred[0]==0):
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 3)
                cv2.putText(frame, "Eyes Closed!", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            else:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
                cv2.putText(frame, "Eyes Open!", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow('img', frame)
            print(y_pred)
        else:
            break
        counter=counter+1
    if cv2.waitKey(1) & 0xFF == ord('q'):
    	break
cv2.destroyAllWindows()
cap.release()
