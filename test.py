import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from keras.models import load_model
import numpy as np
# def process(imgg):
#     imgg = cv2.resize(imgg,(28,28))
#     test_x=[]
#     test_x.append(imgg)
#     test_x = np.array(test_x)/255.0
#     test_x = test_x.reshape(-1,28,28,1)
#     prediction = model.predict(test_x)
#     prob=np.amax(prediction)
#     print(prob*100)
#     print(prediction)
#     x=prediction[1]
#     tt=max(x)
#     if tt==x[0]:
#         man='captain'
#     else:
#         man='tony'
#     print(man)
#     return prob,man

    
video=cv2.VideoCapture('tony_captain_final.mp4')
facedetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model=load_model('D:\ML\ML_project_final\model_movie\model_movie.h5')
count=0
font=cv2.FONT_HERSHEY_COMPLEX
while True:
    res,frame=video.read()
    faces=facedetect.detectMultiScale(frame,1.3,5)
    for x,y,w,h in faces:
        # print(faces)
        count+=1
        # name="./images/"+"/"+str(count)+'.jpg'
        # print("Creating images ............"+name)
        imgg=frame[y:y+h,x:x+w]
        # print("got a face")
        imgg = cv2.resize(imgg,(28,28))
        test_x=[]
        test_x.append(imgg)
        test_x = np.array(test_x)/255.0
        test_x = test_x.reshape(-1,28,28,1)
        prediction = model.predict(test_x)
        prediction=model.predict(test_x)
        y_classes = prediction.argmax(axis=-1)
        # print(y_classes)

        # print(y_classes[0])
        # print(y_classes[1])
        # print(y_classes[2])


        # print(prediction)
        prob=np.amax(prediction)
        # print(prob*100)
        # print(prediction)
        xx=prediction[0]
        tt=max(xx)
        if tt==xx[0]:
            man='tony'
        elif tt==xx[1]:
            man='captain'
        else:
            man='tony'
        print(man)
        # print(x)
        # print(y)
        # print(w)
        # print(h)
        image = cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
        # cv2.imshow(window_name, image)

        # cv2.imwrite(man,frame[y:y+h,x:x+w])
        # cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        # cv2.rectangle(imgg,(x,y),(x+w,y+h),(0,255,0),2)
        # cv2.rectangle(imgg,(x,y-40),(x+w,y),(0,255,0),-2)
        # cv2.putText(imgg, str(man), (x,y-10), font, 0.75,(255,0,0))
        # img=cv2.resize(frame[y:y+h,x:x+w],(28,28,1))
        # prediction=model.predict(img)
        # print(prediction)
        # probabilityValue=np.amax(prediction)
        # cv2.imwrite(name,frame[y:y+h,x:x+w])
        # cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        

    cv2.imshow("windowFrame ",frame)
    cv2.waitKey(1)
    if count>5000:
        break
video.release()
cv2.destroyAllWindows()

