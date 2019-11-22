import cv2
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
df=pd.read_csv('C:\\Users\\tusha\\Desktop\\DATA.csv',index_col='Reg.Num')
print(df)

det=cv2.CascadeClassifier('C:\\Users\\tusha\\Desktop\\Haar_Cascade_images(2)\\haarcascade_frontalface_default.xml')
cam=cv2.VideoCapture(0)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('C:\\Users\\tusha\\Desktop\\face_det\\files\\train_data.yml')
id=0

while True:
    _,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face=det.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in face:
        f=gray[y:y+h,x:x+w]
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        id,conf=recognizer.predict(gray[y:y+h,x:x+w])
        cv2.putText(img,str(id),(x+w-94,y-5),cv2.FONT_HERSHEY_COMPLEX,0.62,(0,0,255),2)
        cv2.putText(img,df.loc[id][0],(x+w-94,y+20),cv2.FONT_HERSHEY_COMPLEX,0.62,(0,0,255),2)
        cv2.putText(img,str(df.loc[id][1]),(x+w-94,y+38),cv2.FONT_HERSHEY_COMPLEX,0.62,(0,0,255),2)
        cv2.putText(img,df.loc[id][2],(x+w-94,y+55),cv2.FONT_HERSHEY_COMPLEX,0.62,(0,0,255),2)
    cv2.imshow('image',img)
    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
