import cv2,os
import numpy as np
from tkinter import messagebox
from tkinter import *
Tk().withdraw()
import time
import warnings
warnings.filterwarnings('ignore')


det=cv2.CascadeClassifier('C:\\Users\\tusha\\Desktop\\Haar_Cascade_images(2)\\haarcascade_frontalface_default.xml')
cam=cv2.VideoCapture(0)

import pandas as pd
df=pd.read_csv('C:\\Users\\tusha\\Desktop\\ladua.csv')

reg_num=list(df['Reg.Num'])
name=list(df['Name'])
age=list(df['Age'])
gender=list(df['Gender'])

id=input('Enter your Registration id : ')
n=input('Input your name : ')
a=input('Input your Age : ')
g=input('Input your Gender : ')
reg_num.append(id)
name.append(n)
age.append(a)
gender.append(g)
df={'Reg.Num':reg_num,'Name':name,'Age':age,'Gender':gender}
df=pd.DataFrame(df)
df.to_csv('C:\\Users\\tusha\\Desktop\\ladua.csv',index=False)
time.sleep(2)

sample_num=0
max_num=40
while True:
    _,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face=det.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in face:
        sample_num+=1
        f=gray[y:y+h,x:x+w]
        cv2.imwrite('C:\\Users\\tusha\Desktop\\face_det\\images\\{}_{}.png'.format(id,sample_num),f)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.waitKey(100)
    cv2.imshow('image',img)
    cv2.waitKey(2)
    if sample_num>=max_num:
        choice=messagebox.askquestion('INFO','Do you want to add more faces?')
        if choice=='yes':
            sample_num=0
            cam.release()
            print('#######################################################')
            id=input('Enter your Registration id : ')
            n=input('Input your name : ')
            a=input('Input your Age : ')
            g=input('Input your Gender : ')
            reg_num.append(id)
            name.append(n)
            age.append(a)
            gender.append(g)
            df={'Reg.Num':reg_num,'Name':name,'Age':age,'Gender':gender}
            df=pd.DataFrame(df)
            df.to_csv('C:\\Users\\tusha\\Desktop\\ladua.csv',index=False)

            cam=cv2.VideoCapture(0)
            continue
        else:
            break

print('saved...')
