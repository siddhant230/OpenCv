import cv2
import numpy as np

cap=cv2.VideoCapture(0)

##your path to haarcascade of openCv for eye
eye=r'/home/parmeet/PycharmProjects/project1/venv/lib/python3.7/site-packages/cv2/data/haarcascade_eye.xml' 

##your path to haarcascade of openCv for frontal face
face=r'/home/parmeet/PycharmProjects/project1/venv/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml'

det=cv2.CascadeClassifier(eye)
det2=cv2.CascadeClassifier(face)
while 1:
    _,img=cap.read()
    gr=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ngr=cv2.equalizeHist(gr)
    eye=det.detectMultiScale(ngr,1.3,5)
    face=det2.detectMultiScale(ngr,1.3,5)

    for (x,y,h,w),(m,n,o,p) in zip(face,eye):
        cv2.rectangle(img,(x,y),(x+h,y+w),(0,0,255),5)
        cv2.rectangle(img,(m,n),(m+o,n+p),(0,255,0),5)
        #cv2.rectangle(img,(m,n),(m+o,n+p),(0,255,0),5)
        cv2.putText(img,'face',((x+h)-20,y-7),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,0,0),2)
        cv2.putText(img,'eye',((m+o)-5,n-3),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,0,0),2)

        cv2.imshow('image',img)
    if cv2.waitKey(1)==ord('q'):
        break
