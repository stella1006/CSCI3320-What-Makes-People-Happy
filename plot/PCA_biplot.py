import pandas as pd
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn import preprocessing

filename_train = '../data/train.csv'
df = pd.read_csv(filename_train,header=0)
df = df.dropna()

for column in df:
    if column!='YOB' and column!='UserID' and column!='Happy' and column!='votes':
        index=[]
        for val in df[column].unique():
            if isinstance(val, str) or isinstance(val, int):
                index.append(val)
        format={x:int('%d' % i) for i,x in enumerate(index)}
        df[column]=df[column].map(format)
X = df

scaler = preprocessing.StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

n = len(df.columns)
pca = PCA(n_components = n)
pca.fit(X)

xvector = pca.components_[0]  # see 'prcomp(my_data)$rotation' in R
yvector = pca.components_[1]

xs = pca.transform(X)[:,0]  # see 'prcomp(my_data)$x' in R
ys = pca.transform(X)[:,1]


for i in range(len(xvector)):
    # arrows project features (ie columns from csv) as vectors onto PC axes
    plt.arrow(0, 0, xvector[i]*max(xs), yvector[i]*max(ys),
              color='r', width=0.0005, head_width=0.0025)
    if (list(df.columns.values)[i] == 'Happy'):
        plt.text(xvector[i]*max(xs)*1.2, yvector[i]*max(ys)*1.2,list(df.columns.values)[i], color='black')
    else:
        plt.text(xvector[i]*max(xs)*1.2, yvector[i]*max(ys)*1.2,list(df.columns.values)[i], color='r')

for i in range(len(xs)):
    # circles project documents (ie rows from csv) as points onto PC axes
    plt.plot(xs[i], ys[i], 'bo')

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA_Biplot')
plt.xlim(-3,3)
plt.ylim(-3,3)
plt.savefig('./PCA_Biplot.png')
plt.show()
