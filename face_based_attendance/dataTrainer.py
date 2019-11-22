import cv2
import numpy as np
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from collections import defaultdict

recognizer = cv2.face.LBPHFaceRecognizer_create()
path='C:\\Users\\tusha\\Desktop\\face_det\\images\\'
file_path='C:\\Users\\tusha\\Desktop\\face_det\\'

def getImageAndPath(path):
    image_path=os.listdir(path)
    data=defaultdict(list)
    faces=[]
    id=[]
    for di in tqdm(range(len(image_path))):
        d=image_path[di]
        reg_id,img_cur=d.split('_')
        img_cur=cv2.imread(path+d)
        img_cur=cv2.cvtColor(img_cur,cv2.COLOR_BGR2GRAY)
        data[reg_id].append(img_cur)
        faces.append(img_cur)
        id.append(int(reg_id))
        cv2.imshow("img",img_cur)
        cv2.waitKey(1)
    return data,id,faces

data,id,faces=getImageAndPath(path)
id=np.array(id)
recognizer.train(faces,id)
recognizer.save(file_path+'\\files\\train_data.yml')
cv2.destroyAllWindows()
