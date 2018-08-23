import numpy as np
import cv2
import matplotlib.pyplot as mpl
import os
import sys

folder_name = 'validationSet'
folder_dest = 'datasetDect'
eye_c = cv2.CascadeClassifier('haarCascade_eye.xml')
counter = 0
print('Running conversion...')
print('<', end='')
for filename in os.listdir(folder_name):
    img = cv2.imread(os.path.join(folder_name, filename))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eye = eye_c.detectMultiScale(gray, 1.5, 5)
    for (x,y,w,h) in eye:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,0), 3)
    cv2.imwrite(os.path.join(folder_dest, str(counter)+'.jpg'), img)
    print("#", end="")
    counter = counter+1
print(">")
print('converted ', counter, ' images')
print('Conversion Successfully completed!')
