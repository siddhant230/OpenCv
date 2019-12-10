import cv2,imutils
from sklearn.externals import joblib
import numpy as np
from skimage.feature import hog

def fun(x):
    pass
cv2.namedWindow('panel')
cv2.createTrackbar('l_h','panel',35,255,fun)
cv2.createTrackbar('l_s','panel',142,255,fun)
cv2.createTrackbar('l_v','panel',0,255,fun)
cv2.createTrackbar('h_h','panel',211,255,fun)
cv2.createTrackbar('h_s','panel',255,255,fun)
cv2.createTrackbar('h_v','panel',255,255,fun)

cv2.namedWindow('panel2')
cv2.createTrackbar('x','panel2',109,255,fun)
cv2.createTrackbar('y','panel2',255,255,fun)
cv2.createTrackbar('z','panel2',19,255,fun)

clf=joblib.load('C:\\Users\\tusha\Desktop\ocrclf1.pkl')

def guess(img):
    x=cv2.getTrackbarPos('x','panel2')
    y=cv2.getTrackbarPos('y','panel2')
    z=cv2.getTrackbarPos('z','panel2')
    im_gr=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    im_gr=cv2.GaussianBlur(im_gr,(5,5),2)
    _,im_th=cv2.threshold(im_gr,x,y,cv2.THRESH_BINARY_INV)
    cnts=cv2.findContours(im_th.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts=imutils.grab_contours(cnts)
    rectangles=[cv2.boundingRect(c) for c in cnts]
    rectangles=sorted(rectangles)

    for rect in rectangles:
        cv2.rectangle(img,(rect[0],rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),(0,255,0),3)
        roi=im_th[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
        if roi.shape[0]>z:
            roi=cv2.GaussianBlur(roi,(3,3),2)
            roi=cv2.resize(roi,(28,28),interpolation=cv2.INTER_AREA)
            roi=cv2.dilate(roi,(3,3))
            roi_hog_fd=hog(roi,orientations=9,pixels_per_cell=(14,14),cells_per_block=(1,1),visualize=False)
            nbr=clf.predict(np.array([roi_hog_fd],'float64'))
            cv2.putText(img,str(int(nbr[0])), (rect[0],rect[1]),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
    return img,im_th

if __name__=='__main__':
    cap=cv2.VideoCapture(0)
    while True:
        _,img=cap.read()
        img,im_th=guess(img)
        cv2.imshow('image',img)
        cv2.imshow('im_th',im_th)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
