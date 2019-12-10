from sklearn import datasets
import numpy as np
from sklearn.svm import LinearSVC
from skimage.feature import hog
from sklearn.externals import joblib
from collections import Counter

dataset=datasets.fetch_mldata('MNIST Original')
features=np.array(dataset.data,'int16')
labels=np.array(dataset.target,'int')

list_hog_fd=[]
for feature in features:
    fd=hog(feature.reshape((28,28)),orientations=9,pixels_per_cell=(14,14),cells_per_block=(1,1))
    list_hog_fd.append(fd)
hog_features=np.array(list_hog_fd,'float64')
print('done with count',Counter(labels))

for i in range(1,2,1):
    clf=LinearSVC(C=3.0,max_iter=5000,random_state=1,tol=1e-5)
    clf.fit(hog_features,labels)

    joblib.dump(clf,'C:\\Users\\tusha\Desktop\ocrclf{}.pkl'.format(i),compress=3)
    print('{} done'.format(i))
