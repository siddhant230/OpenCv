import tkinter as tk
import os
from tkinter import *
from tkinter import messagebox
import sys
import cv2,win32clipboard
import time
import random
import numpy as np
from tkinter.filedialog import askdirectory
from tqdm import tqdm
from functools import partial

canvas=tk.Tk()
canvas.title('HAAR TRAINER')
canvas.geometry("2000x1000")
canvas.configure(background='#ffccbb')

def trainer(data,target,image_size=64,iterations=10,train_size=80):
    canvas.wm_state('iconic')
    data_path=data
    target_path=target
    print('CHOSEN PATH FOR POSITIVE SAMPLES : {}'.format(data_path))
    print('CHOSEN PATH FOR NEGATIVE SAMPLES : {}'.format(target_path))
    ##POSITIVE DATA
    print('LOADING POSITIVE DATA...')
    time.sleep(0.3)
    p_image_name=os.listdir(data_path)
    p_image_name.sort()
    p,p_target=[],[0 for i in range(len(p_image_name))]
    for i in tqdm(range(len(p_image_name))):
        complete_path=data_path+'//'+p_image_name[i]
        img=cv2.imread(complete_path)
        if img.shape!=(image_size):
            img=cv2.resize(img,(image_size,image_size))
            p.append(img)
    time.sleep(0.3)
    ##NEGATIVE DATA
    print('LOADING NEGATIVE DATA...')
    time.sleep(0.3)
    n_image_name=os.listdir(target_path)
    n_image_name.sort()
    n,n_target=[],[1 for i in range(len(n_image_name))]
    for i in tqdm(range(len(n_image_name))):
        complete_path=target_path+'//'+n_image_name[i]
        img=cv2.imread(complete_path)
        if img.shape!=(image_size):
            img=cv2.resize(img,(image_size,image_size))
            n.append(img)
    time.sleep(0.3)
    ##combining sample
    p.extend(n)
    p_target.extend(n_target)
    X=p[:]
    Y=p_target[:]
    ##data loading done

    l=0
    h=len(X)
    xtr,ytr,xte,yte=[],[],[],[]
    data = list(range(l, h))
    random.shuffle(data)
    print('SPLITTING AND RESHAPING DATA...')
    def train_test_split(n):
        d=data[:n]
        for i in d:
            xtr.append(X[i])
            ytr.append(Y[i])
        rem=data[n:]
        for j in rem:
            xte.append(X[j])
            yte.append(Y[j])
        return (xtr,ytr),(xte,yte)
    n_size=int((train_size/100)*len(X))
    (xtr,ytr),(xte,yte)=train_test_split(n_size)

    ##data splitting done###
    xtr=np.array(xtr)
    ytr=np.array(ytr)
    xte=np.array(xte)
    yte=np.array(yte)

    xtr=xtr.reshape(n_size,image_size,image_size,3)
    xte=xte.reshape(len(X)-n_size,image_size,image_size,3)
    xtr=xtr/255.0
    xte=xte/255.0
    from keras.utils import to_categorical
    ytr=to_categorical(ytr)
    yte=to_categorical(yte)
    ##data correction and resizing done
    print('STARTING TRAINING THE MODEL...')
    from keras.models import Sequential
    from keras.layers import Conv2D,Dense,Flatten,MaxPool2D

    model=Sequential()

    model.add(Conv2D(16,input_shape=(image_size,image_size,3),kernel_size=1,activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2),strides=1))

    model.add(Conv2D(32,kernel_size=3,activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2),strides=2))

    model.add(Conv2D(64,kernel_size=3,activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2),strides=2))

    model.add(Conv2D(128,kernel_size=3,activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2),strides=1))

    model.add(Conv2D(256,kernel_size=1,activation='tanh'))             ##new layer

    model.add(Flatten())
    model.add(Dense(2,activation='relu'))
    model.add(Dense(10,activation='relu'))
    model.add(Dense(10,activation='relu'))
    model.add(Dense(10,activation='relu'))
    model.add(Dense(2,activation='softmax'))

    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.fit(xtr,ytr,epochs=iterations,validation_data=(xte,yte))

    ##model building
    from keras.models import load_model
    if not os.path.exists('/home/parmeet/Desktop/HAAR_trainer'):
        os.makedirs('/home/parmeet/Desktop/HAAR_trainer')
    facial_json=model.to_json()
    with open('/home/parmeet/Desktop/HAAR_trainer//trainer.json','w') as jas:
        jas.write(facial_json)
    model.save_weights('/home/parmeet/Desktop/HAAR_trainer//model.h5')
    print('model saved at {}'.format('/home/parmeet/Desktop//HAAR_trainer//model.h5'))
    ##model saved

def validator():
    if data_entry['text']!='' and target_entry['text']!='' and entry_img.get()!='' and entry_iter.get()!='' and train_size.get()!='':
        status=True
    else:
        status=False
    if status:
        messagebox.showinfo('title','here bruhh')
        trainer(data_entry['text'],target_entry['text'],int(entry_img.get()),int(entry_iter.get()),int(train_size.get()))
    else:
        messagebox.showinfo('ALERT','FILL ALL PLACEHOLDERS')
##functions
def data_caller():
    global canvas
    path = askdirectory(title='Select Folder') # shows dialog box and return the path
    data_entry['text']=path

def target_caller():
    global canvas
    files=askdirectory(title='Select Folder') # shows dialog box and return the path
    print(files)
    target_entry['text']=files

def exiter():
    sys.exit()

def copyit(h):
    global val
    r="import cv2\nfrom keras.models import model_from_json\n\nmodel=model_from_json(open('your_file.json','r').read())\nmodel.load_weights('weight.h5')\n#for prediction\nprediction=model.predict(image)"
    win32clipboard.OpenClipboard()
    win32clipboard.EmptyClipboard()
    win32clipboard.SetClipboardText(r,win32clipboard.CF_TEXT)
    win32clipboard.CloseClipboard()
    val=messagebox.showinfo('INFO','TEXT COPIED!')
    if val=='ok':
        h.destroy()

def helper():
    global val
    r="import cv2\nfrom keras.models import model_from_json\n\nmodel=model_from_json(open('your_file.json','r').read())\nmodel.load_weights('weight.h5')\n#for prediction\nprediction=model.predict(image)"
    h=tk.Tk()
    h.geometry("600x400")
    Label(h,text=r,bg="#ffccbb",fg='black',anchor='n',justify=LEFT,width=50,height=20,font=("Times",15,"bold")).pack()
    Button(h,text="COPY",command=partial(copyit,h)).place(relx=0.87,rely=0.04)

title=tk.Label(text='HAAR-CASCADE TRAINER',font={'verdana','100','italic bold'},bg='red',fg='white',bd=8,width=70)
title.pack(fill=Y,side=TOP)

tk.Button(text='EXIT',width=10,bd=5, height=2,bg="white",font=("Times",8,"bold"),command=exiter).place(relx=0.93,rely=0.0)

####data######
data=Label(text="POSITIVE  : ",bg='#ffccbb',fg='black',font=("Times", 12, "bold"),anchor='w')
data.place(relx=0.095,rely=0.17,bordermode=OUTSIDE)
data_entry=Label(text='',font=("Times", 15),width=35,height=1)
data_entry.place(relx=0.159,rely=0.17)
but1=tk.Button(text='+',width=3,bd=5, height=1,bg="white",font=("verdana",11,"bold"),command=data_caller)
but1.place(relx=0.418,rely=0.165)

####target#####
target=Label(text="NEGATIVE  : ",bg='#ffccbb',fg='black',font=("Times", 12, "bold"),anchor='w')
target.place(relx=0.515,rely=0.17,bordermode=OUTSIDE)
target_entry=Label(text='',font=("Times", 15),width=35,height=1)
target_entry.place(relx=0.58,rely=0.17)
but2=tk.Button(text='+',width=3,bd=5, height=1,bg="white",font=("verdana",11,"bold"),command=target_caller)
but2.place(relx=0.840,rely=0.165)

####PHOTO IMAGE SIZE####
data=Label(text="PHOTO SIZE  : ",bg='#ffccbb',fg='black',font=("Times", 12, "bold"),anchor='w')
data.place(relx=0.065,rely=0.37,bordermode=OUTSIDE)
entry_img=Entry(font=("Times"),width=3)
entry_img.place(relx=0.14,rely=0.37)
Label(text='*(use standard formats like 32,64 or 128 for good results)',bg='#ffccbb',fg='red').place(relx=0.17,rely=0.37)

##Iterations##
data=Label(text="ITERATIONS  : ",bg='#ffccbb',fg='black',font=("Times", 12, "bold"),anchor='w')
data.place(relx=0.495,rely=0.37,bordermode=OUTSIDE)
entry_iter=Entry(font=("Times"),width=6)
entry_iter.place(relx=0.579,rely=0.37)
Label(text='*(atleast 10)',bg='#ffccbb',fg='red').place(relx=0.63,rely=0.37)

##how to load trained model for prdiction
Label(text="If you need help regarding loading the model\nPRESS on button below",font=("Times", 12, "bold"),height=7,anchor='n').place(relx=0.515,rely=0.52)
buthelp=tk.Button(text='MODEL LOADING?',width=16,bd=5, height=1,bg="white",font=("verdana",11,"bold"),command=helper)
buthelp.place(relx=0.55,rely=0.625)

##train size###
data=Label(text="TRAIN_SIZE  : ",bg='#ffccbb',fg='black',font=("Times", 12, "bold"),anchor='w')
data.place(relx=0.065,rely=0.57,bordermode=OUTSIDE)
train_size=Entry(font=("Times"),width=6)
train_size.place(relx=0.141,rely=0.57)
Label(text='*(atleast keep 70-80% data [in %])',bg='#ffccbb',fg='red').place(relx=0.2,rely=0.56)

###train button###
but2=tk.Button(text='TRAIN',width=12,bd=5, height=1,bg="white",font=("verdana",11,"bold"),command=validator)
but2.place(relx=0.435,rely=0.725)

canvas.mainloop()
