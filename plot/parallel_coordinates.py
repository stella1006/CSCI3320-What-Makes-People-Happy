from pandas.tools.plotting import parallel_coordinates
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


filename_train = '../data/train.csv'
df = pd.read_csv(filename_train,header=0)
df = df[['Gender', 'Income', 'HouseholdStatus', 'EducationLevel', 'Party', 'Happy']]
df=df.dropna()

for column in df:
    if column!='Happy':
        index=[]
        for val in df[column].unique():
            if isinstance(val, str) or isinstance(val, int):
                index.append(val)
        format={x:int('%d' % i) for i,x in enumerate(index)}
        df[column]=df[column].map(format)

colors = ['r','b']

parallel_coordinates(df, 'Happy', color = colors)
plt.title('parallel_coordinates')
plt.savefig('./para.png')
plt.show()
