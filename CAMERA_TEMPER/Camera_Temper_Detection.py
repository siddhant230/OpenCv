import cv2
import numpy as np
import time
import smtplib

cap0=cv2.VideoCapture(0)

def check_all_pixels(img,factor):
    count=0
    threshold_intensity=80
    for i in img:
        val=np.sum(i)/factor
        if val<=threshold_intensity:
            count+=1
    return (count)


def blur_detection(img):
    text='BLURRED'
    thresh=100.0
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    laplacian_kernel_value=cv2.Laplacian(gray,cv2.CV_64F).var()

    if laplacian_kernel_value<thresh:
        text='BLURRED'

    return text,laplacian_kernel_value

delay=[]
def check_movement(img,i,factor):
    count=0
    change=30
    refresh_rate=151
    global delay
    if delay==[]:
        delay=img
    if i%refresh_rate==0:
        delay=img
    for i,d in zip(img,delay):
        sum_i=np.sum(i)/factor
        sum_d=np.sum(d)/factor
        if abs(sum_i-sum_d)>change:
            count+=1
    return count

i=0
counting0=False
duration0=0
start=0.0
sent0=False

counting1=False
duration1=0
sent1=False
while True:
    i+=1
    _,img0=cap0.read()
    imgarr=[img0]

    copy0=img0.copy()

    (w0,h0,d0)=img0.shape

    count0=check_all_pixels(img0,h0*d0)
    result0,lkv0=blur_detection(copy0)
    moved0=check_movement(img0,i,h0*d0)

    if (count0/w0)*100 >70:
        if counting0==False:
            counting0=True
            start=time.time()
        text0='Camera covered'
        duration0=time.time()-start
        if duration0>5 and sent0==False:
            sent0=True
            s = smtplib.SMTP('smtp.gmail.com', 587)
            s.starttls()
            s.login("rsiddhant73@gmail.com", "************")
            message = "Your camera number 1 has been covered for more than 5 seconds"
            s.sendmail("rsiddhant73@gmail.com", "sidlovesml@gmail.com", message)
            s.quit()
            print('sent email')

    else:
        text0=''
        sent0=False
        counting0=False


    cv2.putText(img0,text0,(10,40),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,2,246),1)
    cv2.putText(img0,result0+' : '+str(lkv0),(10,60),cv2.FONT_HERSHEY_COMPLEX,0.7,(100,2,246),1)
    if lkv0<125:
        cv2.putText(img0,'TOO MUCH BLURRY',(10,90),cv2.FONT_HERSHEY_SIMPLEX,0.7,(100,2,246),1)
    if moved0>60:
        cv2.putText(img0,'MOVED',(10,130),cv2.FONT_HERSHEY_SIMPLEX,0.7,(100,2,246),1)
    cv2.imshow('black',img0)

    if cv2.waitKey(1)==ord('q'):
        break
