import pygame,sys,time
from pygame import *
from tkinter import messagebox
import tkinter as tk
import cv2,imutils
from time import sleep
root=tk.Tk()
root.withdraw()

pygame.init()
cap=cv2.VideoCapture(0)
pygame.display.set_caption('STATUS :  ')
width,height=600,600
b_w=50
b_h=50
screen=pygame.display.set_mode((width,height),0,32)
screen.fill((255,255,255))
center=[0,0]
class rect:
    def __init__(self,x,y):
        self.x=x
        self.y=y
        self.color=(0,255,0)
        self.width=50
        self.height=50
        self.status=True

    def plot(self,col=(0,0,0)):
        pygame.draw.rect(screen,self.color,(self.x,self.y,self.width,self.height))

class car:
    def __init__(self,x=0,y=0):
        self.x=track[0].x+track[0].width//2
        self.y=track[0].y+track[0].height//2
        self.color=(0,0,245)
        self.r=25

    def move(self):
        pygame.draw.circle(screen,self.color,(c.x,c.y),self.r)

rectangles=[]
l_h=84
l_s=116
l_v=19
h_h=176
h_s=255
h_v=255
def mover(img):
    global center,cap
    start=False
    pts=[]
    _,img=cap.read()
    lower,higher=(l_h,l_s,l_v),(h_h,h_s,h_v)
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,lower,higher)
    mask=cv2.erode(mask,None,iterations=3)
    mask=cv2.dilate(mask,None,iterations=3)
    cnts=cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts=imutils.grab_contours(cnts)

    if len(cnts)>0:
        c=max(cnts,key=cv2.contourArea)
        ((x,y),radius)=cv2.minEnclosingCircle(c)
        M=cv2.moments(c)
        center=(int(M['m10']/M['m00']),int(M['m01']/M['m00']))

        if radius>10:
            cv2.circle(img,(int(x),int(y)),int(radius),(0,255,255),2)
            cv2.circle(img,center,5,(0,0,255),-1)
    pts.append(center)
    return img

def mover_util(px,py):
    if abs(center[0]-px)>abs(center[1]-py):
        if center[0]-px>0:
            return 'l'
        elif center[0]-px<0:
            return 'r'
        else:
            return 'no move'
    else:
        if center[1]-py>0:
            return 'd'
        elif center[1]-py<0:
            return 'u'
        else:
            return 'no move'

for i in range(0,width,b_w):
        for j in range(0,height,b_h):
            rectangles.append(rect(i,j))
for r in rectangles:
        r.plot()
start=False
bound=False
track=[]
start_work=False
cx,cy=0,0
up=down=left=right=pressed=False
pin=0
count=0
update=0
thresh=15
while True:
    for e in pygame.event.get():
        if e.type==KEYDOWN:
            pressed=True
            if e.key==K_s:
                start_work=True
                c=car()
                x,y=track[0].x+track[0].width//2,track[0].y+track[0].height//2
                px,py=x,y
                track[0].color=(255,0,255)
            if e.key==K_q:
                pygame.quit()
                sys.exit()
            if e.key==K_r:
                rectangles=[]
                cx,cy=0,0
                for i in range(0,width,b_w):
                        for j in range(0,height,b_h):
                            rectangles.append(rect(i,j))
                for r in rectangles:
                        r.plot()
                start=False
                start_work=False
                track=[]

        ##########################CHANGE PART############################
        if start_work==False:
            if e.type==MOUSEBUTTONDOWN:
                start=True
            if e.type==MOUSEBUTTONUP:
                start=False
            if start==True:
                x,y=pygame.mouse.get_pos()
                for i in range(len(rectangles)):
                    if (x>rectangles[i].x and x<rectangles[i].x+rectangles[i].height) and (y>rectangles[i].y and y<rectangles[i].y+rectangles[i].width):
                        rectangles[i].color=(255,255,255)
                        rectangles[i].status=False
                        if rectangles[i] not in track:
                            track.append(rectangles[i])
    if start_work==True:
        #print(center,(px,py))
        _,img=cap.read()
        img=mover(img)
        cv2.imshow('img',img)
        if cv2.waitKey(1)==ord('q'):
            break
        val=mover_util(px,py)
        pressed=True
        if val=='l':
            left=True
            right=up=down=False
        elif val=='r':
            right=True
            left=up=down=False
        elif val=='u':
            up=True
            left=right=down=False
        elif val=='d':
            down=True
            left=right=up=False
        else:
            continue
        if update%thresh==0:
            px=center[0]
            py=center[1]
    update+=1
    for r in rectangles:
        r.plot()

    if pressed==True:
        bound=False
        if up==True:
            cy=-2*c.r
        if down==True:
            cy=+2*c.r
        if left==True:
            cx=-2*c.r
        if right==True:
            cx=+2*c.r
        for i in range(len(track)):
            count+=1
            if (c.x+cx>=track[i].x and c.x+cx<=track[i].x+track[i].width) and (c.y+cy>=track[i].y and c.y+cy<=track[i].y+track[i].height):
                bound=True
                break
            else:
                bound=False
        if start_work==True and bound==True:
            if up==True or down==True and left==False and right==False:
                c.y+=cy
            if left==True or right==True and up==False and down==False:
                c.x+=cx
    if start_work==True:
        c.move()
    cx,cy=0,0
    pressed=False
    up=down=left=right=False
    bound=False

    if count>1:
        pin=1
    count=0
    pygame.display.set_caption('STATUS : MOVING')
    pygame.display.update()
