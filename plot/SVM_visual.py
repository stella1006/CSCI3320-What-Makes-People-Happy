from sklearn import svm
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

filename_train = '../data/train.csv'
df = pd.read_csv(filename_train,header=0)
df = df[['Income','EducationLevel','Happy']]
df = df.dropna()

h = .02

for column in df:
    if column!='Happy':
        index=[]
        for val in df[column].unique():
            if isinstance(val, str) or isinstance(val, int):
                index.append(val)
        format={x:int('%d' % i) for i,x in enumerate(index)}
        df[column]=df[column].map(format)

x_min, x_max = df[['Income']].min() - 1, df[['Income']].max() + 1
y_min, y_max = df[['EducationLevel']].min() - 1, df[['EducationLevel']].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

clf = svm.SVC(kernel='rbf', gamma=0.7, C = 1.0)
clf.fit(df[['Income','EducationLevel']],df['Happy'])
marker = ['r','b']
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
plt.scatter(df['Income'],df['EducationLevel'], color = marker)
plt.xlabel('Income')
plt.ylabel('EducationLevel')
plt.title('SVM')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.savefig('./SVM.png')
plt.show()
