import cv2
import os
video=cv2.VideoCapture(0)
facedetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
count=0
nameID=str(input("Enter your name: "))

path='images/'+nameID
isExist=os.path.exists(path)
if isExist:
    print("Name Already exists")
    nameID=str(input("Enter your name: "))
else:
    os.makedirs(path)
    
while True:
    res,frame=video.read()
    faces=facedetect.detectMultiScale(frame,1.3,5)
    for x,y,w,h in faces:
        count+=1
        name="./images/"+nameID+"/"+str(count)+'.jpg'
        print("Creating images ............"+name)
        cv2.imwrite(name,frame[y:y+h,x:x+w])
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
    cv2.imshow("windowFrame ",frame)
    cv2.waitKey(1)
    if count>500:
        break
video.release()
cv2.destroyAllWindows()