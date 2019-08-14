import cv2
import matplotlib.pyplot as plt
import numpy as np
import imutils
from imutils.perspective import four_point_transform
# return the warped image
img='/home/parmeet/Downloads/omr.png'
img=cv2.imread(img)
key={0:1,1:4,2:0,3:2,4:3}
org=img.copy()
##image to gray
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur=cv2.GaussianBlur(gray,(5,5),0)
edge=cv2.Canny(gray,75,200)
cnts=cv2.findContours(edge.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts=imutils.grab_contours(cnts)
final=None
if len(cnts)>0:
    cnts=sorted(cnts,key=cv2.contourArea,reverse=True)
    for c in cnts:
        peri=cv2.arcLength(c,True)
        approx=cv2.approxPolyDP(c,0.02*peri,True)
        if len(approx)==4:
            final=approx
            break
paper=four_point_transform(org,final.reshape(4,2))
warp=four_point_transform(gray,final.reshape(4,2))
thresh=cv2.threshold(warp,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]
cnts=cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts=imutils.grab_contours(cnts)
points=[]
for c in cnts:
    (x,y,w,h)=cv2.boundingRect(c)
    ar=w/float(h)
    if w>20 and h>=20 and ar>=0.9 and ar<=1.1:
        points.append(c)

from imutils import contours
points=contours.sort_contours(points,method="top-to-bottom")[0]
correct=0
for (q,i) in enumerate(np.arange(0,len(points),5)):
    cnts=contours.sort_contours(points[i:i+5])[0]
    bubbled=None
    for (j,c) in enumerate(cnts):
        mask=np.zeros(thresh.shape,dtype='uint8')
        cv2.drawContours(mask,[c],-1,255,-1)
        mask=cv2.bitwise_and(thresh,thresh,mask=mask)
        total=cv2.countNonZero(mask)
        if bubbled is None or total>bubbled[0]:
            bubbled=(total,j)
    color=(0,0,255)
    k=key[q]
    if k==bubbled[1]:
        color=(0,255,0)
        correct+=1
    v=cnts[k]
    cv2.drawContours(paper,[v],-1,color,3)
marks="your percentage is {}%".format((correct/len(key))*100)
cv2.putText(paper,marks,(10,40),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,2,246),1)
cv2.imshow('final',paper)
cv2.waitKey(0)
plt.imshow(paper)
plt.savefig('/home/parmeet/Downloads/marks_of_omr.png')
