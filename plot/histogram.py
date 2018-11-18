import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

filename = '../data/train.csv'
df = pd.read_csv(filename,header=0)
df = df['YOB']
df = df.dropna()
df = df.as_matrix()

plt.figure()

# X=np.arange(1, df.shape[0])
plt.hist(df, 200, normed=1,)
plt.xlabel('YOB')
plt.ylabel('Probability Density')
plt.title('Distribution of YOB')
plt.savefig('./histogram.png')
plt.show()
