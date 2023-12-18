import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from keras.models import load_model
import numpy as np


    
video=cv2.VideoCapture('VideoName.mp4')
facedetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model=load_model('path\to\your\model.h5')
count=0
font=cv2.FONT_HERSHEY_COMPLEX
while True:
    res,frame=video.read()
    faces=facedetect.detectMultiScale(frame,1.3,5)
    for x,y,w,h in faces:
        # print(faces)
        count+=1
        imgg=frame[y:y+h,x:x+w]
        imgg = cv2.resize(imgg,(28,28))
        test_x=[]
        test_x.append(imgg)
        test_x = np.array(test_x)/255.0
        test_x = test_x.reshape(-1,28,28,1)
        prediction = model.predict(test_x)
        prediction=model.predict(test_x)
        y_classes = prediction.argmax(axis=-1)

        prob=np.amax(prediction)
        xx=prediction[0]
        tt=max(xx)
        if tt==xx[0]:
            man='tony'
        elif tt==xx[1]:
            man='captain'
        else:
            man='tony'
        print(man)

        image = cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
        

    cv2.imshow("windowFrame ",frame)
    cv2.waitKey(1)
    if count>5000:
        break
video.release()
cv2.destroyAllWindows()

