import matplotlib.pyplot as plt
import cv2
import pickle
import random
import numpy as np

##data creation
width,height=64,64
X=[]
Y=[]
d_path='/home/parmeet/Desktop/combined_for_cnn/x/'
src='/home/parmeet/Desktop/combined_for_cnn/y.txt'
with open (src, 'rb') as fp:
    itemlist = pickle.load(fp)
name=[d_path+str(i)+'.jpg' for i in range(0,2183)]
for i in range(2182):
    if i==1465  or i==1500:
        continue
    try:
        img=cv2.imread(name[i])
        if img.shape[0]>32:
            img=cv2.resize(img,(width,height))
    except:
        continue
    else:
        print(i)
        #img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        Y.append(itemlist[i])
        X.append(img)

##data splitting
l=0
h=len(X)
xtr,ytr,xte,yte=[],[],[],[]
data = list(range(l, h))
random.shuffle(data)

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
n_size=2000
(xtr,ytr),(xte,yte)=train_test_split(n_size)

##data shape correction
xtr=np.array(xtr)
ytr=np.array(ytr)
xte=np.array(xte)
yte=np.array(yte)

xtr=xtr.reshape(n_size,width,height,3)
xte=xte.reshape(len(X)-n_size,width,height,3)

xtr=xtr/255.0
xte=xte/255.0

from keras.utils import to_categorical

ytr=to_categorical(ytr)
yte=to_categorical(yte)
print(ytr)

##model buildig
from keras.models import Sequential
from keras.layers import Conv2D,Dense,Flatten,MaxPool2D

model=Sequential()

####found something relevant....train acc=>82%-85% vaidation acc=>85% on 8 iterations
model.add(Conv2D(16,input_shape=(width,height,3),kernel_size=1,activation='relu'))
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

#sgd=optimizers.SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(xtr,ytr,epochs=8,validation_data=(xte,yte))
from keras.models import load_model
#model.save('/home/parmeet/Desktop/new_haar/fire.h5')

#obj=load_model('/home/parmeet/Desktop/new_haar/fire.h5')
obj=model

##predicting data
prediction=[]
n=100
pred=obj.predict(xte[:n])
yte=yte[:n]
print(pred)
for ans in pred:
  ans=list(ans)
  prediction.append(ans.index(max(ans)))

print('#########################')
c=0
for pred,exp in zip(prediction,yte):
  if exp[pred]==1:
    c+=1
  print(pred,list(exp).index(1))
print((c/n)*100)
