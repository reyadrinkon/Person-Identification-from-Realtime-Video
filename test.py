import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from keras.models import load_model
import numpy as np
font = cv2.FONT_HERSHEY_SIMPLEX
facedetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap=cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
front=cv2.FONT_HERSHEY_COMPLEX
model=load_model('D:\ML\ML_AI_Project_4th-year\models\model.h5')

def getclassName(classNo):
    if classNo==0:
        return 'Habiba'
    else:
        return 'Rinkon'
while True:
    success,imgOriginal=cap.read()
    faces=facedetect.detectMultiscale(imgOriginal,1.3,5)
    for x,y,w,h in faces:
        crop_img=imgOriginal[y:y+h,x:x+w]
        img=cv2.resize(crop_img,(28,28))
        prediction=model.predict(img)
        classIndex=model.predict_classes(img)
        probabilityValue=np.amax(prediction)
        cv2.rectangle(imgOriginal,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.rectangle(imgOriginal,(x,y-40),(x+w,y),(0,255,0),-2)
        cv2.putText(imgOriginal, str(getclassName(classIndex)), (x,y-10), font, 0.75,(255,0,0))
        
        # if classIndex==0:
        #     cv2.rectangle(imgOriginal,(x,y),(x+w,y+h),(0,255,0),2)
        #     cv2.rectangle(imgOriginal,(x,y-40),(x+w,y),(0,255,0),-2)
        #     cv2.putText(imgOriginal,str(getclassName(classIndex)),(x,y-10), font ,0.75,(255,0,0))
        # else:
        #     cv2.rectangle(imgOriginal,(x,y),(x+w,y+h),(0,255,0),2)
        #     cv2.rectangle(imgOriginal,(x,y-40),(x+w,y),(0,255,0),-2)
        #     cv2.putText(imgOriginal,str(getclassName(classIndex)),(x,y-10), font ,0.75,(255,0,0))
            