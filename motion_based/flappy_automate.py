import pygame
from pygame.locals import *
import sys
import random,cv2
import time,math,imutils
import numpy as np

bird_img='C:\\Users\\tusha\\Downloads\\bird.png'
#back_img='/home/parmeet/Downloads/flappy_back.jpeg'
pygame.init()
pygame.display.set_caption('FLAPPY BIRD')

screen=pygame.display.set_mode((600,370),0,32)
'''back=pygame.image.load(back_img)
screen.blit(back,(0,0))
'''
w,h=600,370
bird=pygame.image.load(bird_img).convert_alpha()
y_bird=165
x_bird=70
y=0
move=0
color=(0,255,0)

##pillar banao
class pipe:

    def __init__(self):
        self.top=random.randint(0,h//2)
        self.bottom=random.randint(0,h//2)
        self.x=w
        self.width_of_pole=45
        self.speed=8
        self.score=0

    def show(self):
        rectT=(self.x,0,self.width_of_pole,self.top)
        rectB=(self.x,self.top+170,self.width_of_pole,h)
        pygame.draw.rect(screen,color,rectT)
        pygame.draw.rect(screen,color,rectB)

    def update(self):
        self.x-=self.speed

    def collision(self,bx,by):
        if bx>self.x and bx<self.x+self.width_of_pole:
            if by<=self.top or by>=self.top+170:
                color=(255,0,0)
                rect_newT=(self.x,0,self.width_of_pole,self.top)
                rect_newB=(self.x,self.top+170,self.width_of_pole,h)
                pygame.draw.rect(screen,color,rect_newT)
                pygame.draw.rect(screen,color,rect_newB)
                return 1
            return 0

cam=cv2.VideoCapture(0)
cv2.namedWindow('panel')
def fun(x):
    pass
cv2.createTrackbar('l_h','panel',0,255,fun)
cv2.createTrackbar('l_s','panel',40,255,fun)
cv2.createTrackbar('l_v','panel',0,255,fun)
cv2.createTrackbar('h_h','panel',176,255,fun)
cv2.createTrackbar('h_s','panel',255,255,fun)
cv2.createTrackbar('h_v','panel',255,255,fun)
def tell_jump(img):
    kernel=np.ones((5,5))
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

        if count_defect>2:
            return True
        else:
            return False

def bird_move_up():
    gravity=-9.9
    return gravity

def bird_move_down():
    gravity=9.7
    return gravity

pillar=[]
pillar.append(pipe())

x_start=550
press=0
up=None
start=False
score=0
while True:

    for e in pygame.event.get():
        if e.type==KEYDOWN:

            if e.key==K_q:
                pygame.quit()
                sys.exit()

            if e.key==K_SPACE:
                press=time.time()
                up=True
                y=bird_move_up()

        if e.type==KEYUP:
            if e.key==K_SPACE:
                y=bird_move_down()
                up=False
    screen.fill((0,0,0))
    ##let the bird jump

    y_bird+=y
    print(y_bird)
    if y_bird>340:
        y_bird=340
    if y_bird<25:
        y_bird=25
    screen.blit(bird,(x_bird,y_bird))
    ##making the pillars
    if e.type==KEYDOWN:
        if e.key==K_c:
            start=True

    if start:
        _,img=cam.read()
        jump=tell_jump(img)
        cv2.imshow('image',img)
        if cv2.waitKey(1)==ord('q'):
            break
        if jump==True:
            up=True
            y =bird_move_up()
        else:
            up=False
            y=bird_move_down()
        if move%60==0:
            pillar.append(pipe())
        for i in range(len(pillar)):
            pillar[i].show()
            if move%2==0:
                pillar[i].update()
                response=pillar[i].collision(x_bird,y_bird)
                if response==1:
                    score-=1
                if response==0:
                    score+=1
        if len(pillar)>10:
            pillar.pop(0)
    move+=1

    myfont = pygame.font.SysFont("Comic Sans MS", 30)
    label = myfont.render("SCORE : "+str(score//7), 3, (255,0,0))
    screen.blit(label,(10,20))
    lab2=myfont.render("Press c to Start", 10, (255,0,0))
    if start==False:
        screen.blit(lab2,(250,300))

    pygame.display.update()
