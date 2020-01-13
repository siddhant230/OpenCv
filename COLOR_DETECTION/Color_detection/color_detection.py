import pandas as pd
import cv2
import numpy as np
from scipy.spatial import distance

cols=['color','color_name','hex','rgb']
df=pd.read_csv('colors.csv',header=None)

def color_finder(color):
    minimum_dist=10**5
    index_color=None
    for i in range(len(df)):
        pos=[int(df.loc[i,len(df.columns)-3]),int(df.loc[i,len(df.columns)-2]),int(df.loc[i,len(df.columns)-1])]
        dist=distance.euclidean(color,pos)
        if dist<minimum_dist:
            minimum_dist=dist
            index_color=i
    color_found=df.loc[index_color,1]
    return color_found

mouseX=0
mouseY=0
def draw_circle(event,x,y,flags,param):
    global mouseX,mouseY
    if cv2.EVENT_LBUTTONDOWN:
        mouseX,mouseY = x,y

cap=cv2.VideoCapture(0)
while True:
    _,img=cap.read()
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',draw_circle)
    center=(mouseX,mouseY)
    x,y=center
    if (y<480 and x>0) and (y<640 and y>0):
        col=img[y,x]
        found=color_finder(col)
        cv2.putText(img,found,(10,40),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,2,246),1)
    cv2.imshow('image',img)
    if cv2.waitKey(1)==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
