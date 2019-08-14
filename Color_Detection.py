import cv2
import numpy as np
def color_guess(color=None):
    color_dict={"black":[0, 0, 0],
    "green":[0, 128, 0],
    "silver":[192, 192, 192],
    "lime" :[0, 255, 0],
    "gray":[128, 0, 128],
    "olive" :[128, 128, 0],
    "white" :[255, 255, 255],
    "yellow" :[225, 225, 100],
    "maroon" :[128, 0, 0],
    "navy" :[0, 0, 128],
    "red"  :[255, 0, 0],
    "blue" :[0, 0, 255],
    "purple" :[128, 0, 128],
    "teal"  :[0, 128, 128],
    "fuchsia" :[255, 0, 255],
    "aqua" :[0, 255, 255]}
    r,g,b=color
    shade=['red','green','blue']

    if abs(r-g)>100 or abs(r-b)>100 or abs(b-g)>100:
        a=color.index(max(color))
        return shade[a]

    closest=''
    best=999**9
    for k,v in color_dict.items():
        total=0
        for i in range(3):
            total+=abs(color[i]-v[i])
        if total<best:
            best=total
            closest=k
    return closest

def check_color(image):
    pix=[0,0,0]
    for i in image:
        for j in i:
            pix[0]+=j[0]
            pix[1]+=j[1]
            pix[2]+=j[2]
    for i in range(len(pix)):
        pix[i]=pix[i]//400
    pix.reverse()
    col=color_guess(pix)
    return col

mouseX=0
mouseY=0
def draw_circle(event,x,y,flags,param):
    global mouseX,mouseY
    if cv2.EVENT_LBUTTONDOWN:
        mouseX,mouseY = x,y

cap=cv2.VideoCapture(0)

while(1):
    _,img=cap.read()
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',draw_circle)
    center=(mouseX,mouseY)
    x,y=center
    extract=img[y:y+20,x:x+20]
    #extract=cv2.cvtColor(extract,cv2.COLOR_BGR2HSV)
    col=check_color(extract)
    cv2.putText(img,col,(10,40),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,2,246),1)
    cv2.imshow('image',img)
    cv2.imshow('extract',extract)
    if cv2.waitKey(1)==ord('q'):
        break
