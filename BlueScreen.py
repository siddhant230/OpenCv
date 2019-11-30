import cv2
import numpy as np

def fun(val):
    pass
cv2.namedWindow('panel')

##adjustment panel for setting your hsv values as per your lighting conditions.
cv2.createTrackbar('l_h','panel',35,255,fun)
cv2.createTrackbar('l_s','panel',142,255,fun)
cv2.createTrackbar('l_v','panel',0,255,fun)
cv2.createTrackbar('h_h','panel',211,255,fun)
cv2.createTrackbar('h_s','panel',255,255,fun)
cv2.createTrackbar('h_v','panel',255,255,fun)
cv2.namedWindow('panel2')
cv2.createTrackbar('r_s','panel2',0,655,fun)
cv2.createTrackbar('r_e','panel2',400,655,fun)
cv2.createTrackbar('c_s','panel2',0,655,fun)
cv2.createTrackbar('c_e','panel2',400,655,fun)
cap=cv2.VideoCapture(0)
mov=cv2.VideoCapture('C:\\Users\\tusha\\Desktop\\Predestination (2014) - Polygon Movies.mp4')
st=1
while True:
    _,img=cap.read()
    _,back=mov.read()
    if st==1:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('C:\\Users\\tusha\\Desktop\\video.avi', fourcc, 20.0, (img.shape[1], img.shape[0]))
    st=9999
    rs=cv2.getTrackbarPos('r_s','panel2')
    re=cv2.getTrackbarPos('r_e','panel2')
    cs=cv2.getTrackbarPos('c_s','panel2')
    ce=cv2.getTrackbarPos('c_e','panel2')
    img=img[rs:re,cs:ce]
    new_back=back[rs:re,cs:ce]
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    l_h=cv2.getTrackbarPos('l_h','panel')
    l_s=cv2.getTrackbarPos('l_s','panel')
    l_v=cv2.getTrackbarPos('l_v','panel')
    h_h=cv2.getTrackbarPos('h_h','panel')
    h_s=cv2.getTrackbarPos('h_s','panel')
    h_v=cv2.getTrackbarPos('h_v','panel')

    lower=(l_h,l_s,l_v)
    upper=(h_h,h_s,h_v)
    mask=cv2.inRange(hsv,lower,upper)
    mask_inv=cv2.bitwise_not(mask)

    bg=cv2.bitwise_and(img,img,mask=mask)

    back=cv2.bitwise_and(new_back,new_back,mask=mask)
    fg=cv2.bitwise_and(img,img,mask=mask_inv)

    new_opt=cv2.add(back,fg)
    opt=cv2.resize(new_opt,(ce,re))
    out.write(opt)
    cv2.imshow('opt',new_opt)
    if cv2.waitKey(1)==ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()
