import numpy
import cv2
import sys
import os
folder_name = 'test'
folder_dest = 'eye_script'
eye_c = cv2.CascadeClassifier('haarCascade_eye.xml')
face_c = cv2.CascadeClassifier('haarCascadeFaceF.xml')
counter = 0
count = 0
exit_flag = 0
for file in os.listdir(folder_name):
    count += 1
    cap = cv2.VideoCapture(folder_name+'\\'+file)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face = face_c.detectMultiScale(gray, 1.5, 5)
            for (a,b,c,d) in face:
                cv2.rectangle(frame, (a,b), (a+c,b+d), (255,0,0), 2)
                roi_f = gray[b:b+d, a:a+c]
                eye = eye_c.detectMultiScale(roi_f, 1.5, 5)
                for(x,y,w,h) in eye:
                    cv2.rectangle(frame, (a+x,b+y), (a+x+w,b+y+h), (0,0,255), 2)
                    roi = roi_f[y:y+h, x:x+w]
                    roi = cv2.resize(roi, (224,224))
                    cv2.imwrite(os.path.join(folder_dest, str(counter)+'.jpg'), roi)
                    counter += 1
                    print('#', end='')
                    cv2.imshow('eye', roi)
            cv2.imshow('video', frame)
            if cv2.waitKey(1) & 0xFF == ord('n'):
                break
        else:
            print('')
            print('video '+str(count)+' is processed')
            break
    cap.release()
print("")
print('End of process with '+str(counter)+' images')
cv2.destroyAllWindows()
