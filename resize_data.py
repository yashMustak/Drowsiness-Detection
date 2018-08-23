import cv2
import sys
import os
folder_name = 'bmpv'
folder_name_dest = 'xvalidation'
counter = 192
#counter1 = 192
for filename in os.listdir(folder_name):
    img = cv2.imread(os.path.join(folder_name, filename))
    #counter=counter+1
    if img is not None:
        #if counter>199:
        imgr = cv2.resize(img,(224,224))
        gray = cv2.cvtColor(imgr, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(folder_name_dest, str(counter)+'.jpg'), gray)
        print("#", end="")
        counter = counter+1
print("")
print("converted ", counter, "images")
print("End of program")
