import dlib
from scipy.spatial import distance as dist
import numpy as np
import playsound
from imutils import face_utils
import cv2

def sound_alarm(path):
    playsound.playsound(path)

alarm_path='/home/parmeet/Documents/Fairy-tail-theme 1.mp3'
det_path='/home/parmeet/Desktop/Siddhant Rai/impi/drowsiness-detection/shape_predictor_68_face_landmarks.dat'
eye_thresh=0.3
frame_allowed=48
count=0
alarm_state=False

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    print(shape.part)
    for i in range(0, 68):
        print(i)
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def eye_aspect_ratio(eye):
    #vertical landmark
    a=dist.euclidean(eye[1],eye[5])
    b=dist.euclidean(eye[2],eye[4])

    #horizontal landmark
    c=dist.euclidean(eye[0],eye[3])

    ear=(a+b)/(2.0*c)
    return ear

detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor(det_path)

(lstart,lend)=face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rstart,rend)=face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

cap=cv2.VideoCapture(0)
#cap=VideoStream(src=0)
while True:
    _,img=cap.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    rects=detector(gray,0)
    for r in rects:
        shape=predictor(gray,r)
        shape=face_utils.shape_to_np(shape)
        lefteye=shape[lstart:lend]
        righteye=shape[rstart:rend]

        leftear=eye_aspect_ratio(lefteye)
        rightear=eye_aspect_ratio(righteye)

        total_ear=(leftear+rightear)/2

        lefthull=cv2.convexHull(lefteye)
        righthull=cv2.convexHull(righteye)
        cv2.drawContours(img,[lefthull],-1,(0,255,0),1)
        cv2.drawContours(img,[righthull],-1,(0,255,0),1)
        print(total_ear,eye_thresh)
        if total_ear<eye_thresh:
            count+=1
            cv2.putText(img,'WAKE UP!!',(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),3)
            if count>=frame_allowed:
                if alarm_state==False:
                    alarm_state=True
                    if alarm_path!='':
                            sound_alarm(alarm_path)
        else:
            count=0
            t=0
            alarm_state=False

    cv2.imshow('check it',img)

    if cv2.waitKey(1)==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
