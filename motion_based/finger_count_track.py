import cv2,os
import numpy as np
import warnings,math,imutils

warnings.filterwarnings('ignore')
cv2.namedWindow('panel')
def fun(x):
    pass
cv2.createTrackbar('l_h','panel',0,255,fun)
cv2.createTrackbar('l_s','panel',40,255,fun)
cv2.createTrackbar('l_v','panel',0,255,fun)
cv2.createTrackbar('h_h','panel',176,255,fun)
cv2.createTrackbar('h_s','panel',255,255,fun)
cv2.createTrackbar('h_v','panel',255,255,fun)
cam=cv2.VideoCapture(0)
kernel=np.ones((5,5))
while True:
    _,img=cam.read()
    l_h=cv2.getTrackbarPos('l_h','panel')
    l_s=cv2.getTrackbarPos('l_s','panel')
    l_v=cv2.getTrackbarPos('l_v','panel')
    h_h=cv2.getTrackbarPos('h_h','panel')
    h_s=cv2.getTrackbarPos('h_s','panel')
    h_v=cv2.getTrackbarPos('h_v','panel')
    lower,higher=(l_h,l_s,l_v),(h_h,h_s,h_v)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.rectangle(img,(70,70),(320,320),(0,0,255),1)
    hand=img[70:320,70:320]
    blur=cv2.GaussianBlur(hand,(3,3),0)
    hsv=cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,lower,higher)

    dilate=cv2.dilate(mask,kernel,iterations=1)
    erode=cv2.erode(dilate,kernel,iterations=1)

    filter=cv2.GaussianBlur(erode,(3,3),0)
    ret,thresh=cv2.threshold(filter,127,255,0)

    cnts=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnts=imutils.grab_contours(cnts)

    if len(cnts)>0:
        contour=max(cnts, key = lambda m :cv2.contourArea(m))
        (x,y,w,h)=cv2.boundingRect(contour)
        cv2.rectangle(hand,(x,y),(x+w,y+h),(0,0,255),0)
        hull=cv2.convexHull(contour)
        hull=cv2.convexHull(contour,returnPoints=False)
        defects=cv2.convexityDefects(contour,hull)

        count_defect=0

        for i in range(defects.shape[0]):
            s,e,f,d=defects[i,0]
            start=tuple(contour[s][0])
            end=tuple(contour[e][0])
            far=tuple(contour[f][0])

            a=math.sqrt((end[0]-start[0])**2 +(end[1]-start[1])**2)
            b=math.sqrt((far[0]-start[0])**2 +(far[1]-start[1])**2)
            c=math.sqrt((end[0]-far[0])**2 +(end[1]-far[1])**2)

            angle=(math.acos((b**2 +c**2 -a**2)/(2*b*c))*180)/3.14

            if angle<=90:
                count_defect+=1
        print(count_defect)
        if count_defect==-1:
            cv2.putText(img,'mukka',(x+w-94,y-5),cv2.FONT_HERSHEY_COMPLEX,0.62,(0,0,255),2)
        elif count_defect==0:
            cv2.putText(img,'one',(x+w-94,y-5),cv2.FONT_HERSHEY_COMPLEX,0.62,(0,0,255),2)
        elif count_defect==1:
            cv2.putText(img,'two',(x+w-94,y-5),cv2.FONT_HERSHEY_COMPLEX,0.62,(0,0,255),2)
        elif count_defect==2:
            cv2.putText(img,'three',(x+w-94,y-5),cv2.FONT_HERSHEY_COMPLEX,0.62,(0,0,255),2)
        elif count_defect==3:
            cv2.putText(img,'four',(x+w-94,y-5),cv2.FONT_HERSHEY_COMPLEX,0.62,(0,0,255),2)
        elif count_defect==4:
            cv2.putText(img,'five',(x+w-94,y-5),cv2.FONT_HERSHEY_COMPLEX,0.62,(0,0,255),2)
        else:
            pass
    else:
        pass
    cv2.imshow('image',img)
    cv2.imshow('hand',thresh)

    if cv2.waitKey(1)==ord('q'):
        break
