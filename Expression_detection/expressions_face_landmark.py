import dlib
from scipy.spatial import distance as dist
import numpy as np
import playsound
from imutils import face_utils
import cv2

def sound_alarm(path):
    playsound.playsound(path)

alarm_path='theme_1.mp3'
det_path='/home/parmeet/Downloads/facial-landmarks//shape_predictor_68_face_landmarks.dat'
eye_thresh=0.3
frame_allowed=48
count=0
alarm_state=False

def eye_aspect_ratio(eye):
    #vertical landmark
    a=dist.euclidean(eye[1],eye[5])
    b=dist.euclidean(eye[2],eye[4])

    #horizontal landmark
    c=dist.euclidean(eye[0],eye[3])
    ear=(a+b)/(2.0*c)
    return ear*100

def mouth_aspect_ratio(mouth):
    a=dist.euclidean(mouth[0],mouth[6])
    b=dist.euclidean(mouth[3],mouth[9])
    return a,b

def eyebrow_eye_dist(brow,eye):
    a=dist.euclidean(brow[0],eye[0])
    b=dist.euclidean(brow[4],eye[3])
    aspect=(a+b)/2
    return aspect

thresh_total_ear=29
thresh_mouth_left_right_dist=100
thresh_mouth_to_bottom_dist=60
thresh_dist_eyebrow=195
thresh_total_aspect_bw_eye_brow=61

def check_angry(dist_eyebrow,mouth_left_right_dist,total_ear,mouth_to_bottom_dist):
    if (dist_eyebrow<thresh_dist_eyebrow and mouth_left_right_dist<thresh_mouth_left_right_dist and total_ear<thresh_total_ear and mouth_to_bottom_dist<thresh_mouth_to_bottom_dist):
        return True
    else:
        return False

def check_surprise(total_aspect_bw_eye_brow,mouth_to_bottom_dist):
    if total_aspect_bw_eye_brow>thresh_total_aspect_bw_eye_brow and mouth_to_bottom_dist>thresh_mouth_to_bottom_dist:
        return True
    else:
        return False

def check_happy(mouth_left_right_dist):
    if mouth_left_right_dist>thresh_mouth_left_right_dist:
        return True
    else:
        return False

def check_sad(total_ear):
    if total_ear<thresh_total_ear:
        return True
    else:
        return False

def check_fear(total_ear):
    if total_ear>thresh_total_ear:
        return True
    else:
        return False

detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor(det_path)
###face features###
(leyestart,leyeend)=face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(reyestart,reyeend)=face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
inner_mouth=face_utils.FACIAL_LANDMARKS_IDXS['inner_mouth']
outer_mouth=face_utils.FACIAL_LANDMARKS_IDXS['mouth']
(leyebrowstart,leyebroend)=face_utils.FACIAL_LANDMARKS_IDXS['left_eyebrow']
(reyebrowstart,reyebrowend)=face_utils.FACIAL_LANDMARKS_IDXS['right_eyebrow']

cap=cv2.VideoCapture(0)
move=1

while True:
    _,img=cap.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    rects=detector(gray,0)

    for r in rects:
        shape=predictor(gray,r)
        shape=face_utils.shape_to_np(shape)

        lefteye=shape[leyestart:leyeend]
        righteye=shape[reyestart:reyeend]

        ##eye
        leftear=eye_aspect_ratio(lefteye)
        rightear=eye_aspect_ratio(righteye)
        total_ear=(leftear+rightear)/2

        ##mouth
        mouth_in=shape[inner_mouth[0]:inner_mouth[1]]
        mouth_out=shape[outer_mouth[0]:outer_mouth[1]]
        mouth_left_right_dist,mouth_to_bottom_dist=mouth_aspect_ratio(mouth_out)

        ##eyebrow
        eye_bro_left=shape[leyebrowstart:leyebroend]
        eye_bro_right=shape[reyebrowstart:reyebrowend]
        dist_eyebrow=dist.euclidean(eye_bro_left[-1],eye_bro_right[0])

        ##eyebrow and eye
        dist_leyebrow_leye=eyebrow_eye_dist(eye_bro_left,lefteye)
        dist_reyebrow_reye=eyebrow_eye_dist(eye_bro_right,righteye)
        total_aspect_bw_eye_brow=(dist_leyebrow_leye+dist_reyebrow_reye)

        lefthull=cv2.convexHull(lefteye)
        righthull=cv2.convexHull(righteye)

        cv2.drawContours(img,[lefthull],-1,(0,255,0),1)
        cv2.drawContours(img,[righthull],-1,(0,255,0),1)
        cv2.drawContours(img,[cv2.convexHull(eye_bro_left)],-1,(0,255,0),1)
        cv2.drawContours(img,[cv2.convexHull(eye_bro_right)],-1,(0,255,0),1)
        cv2.drawContours(img,[cv2.convexHull(mouth_in)],-1,(0,255,0),1)
        cv2.drawContours(img,[cv2.convexHull(mouth_out)],-1,(0,255,0),1)

    emotion=None
    if check_angry(dist_eyebrow,mouth_left_right_dist,total_ear,mouth_to_bottom_dist):
        emotion='Angry'
    elif check_surprise(total_aspect_bw_eye_brow,mouth_to_bottom_dist):
        emotion='Surprise'
    elif check_happy(mouth_left_right_dist):
        emotion='Happy'
    elif check_sad(total_ear):
        emotion='Sad'
    elif check_fear(total_ear):
        emotion='Fear'
    else:
        emotion='O Paaji! Expression to do'

    print(dist_eyebrow,mouth_left_right_dist,total_ear)
    cv2.putText(img,emotion,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),3)
    cv2.imshow('check it',img)

    if cv2.waitKey(1)==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
