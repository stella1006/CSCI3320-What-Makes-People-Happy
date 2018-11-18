import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

filename = '../data/train.csv'
df = pd.read_csv(filename,header=0)
df = df[['YOB', 'Income']]
df = df.dropna()
index=[]
indexs=['under $25,000', '$37,500', '$62,500', '$87,500', '$125,000', 'over $150,000']
df=df.replace(['under $25,000','$25,001 - $50,000','$50,000 - $74,999','$75,000 - $100,000','$100,001 - $150,000','over $150,000'],[0,1,2,3,4,5])
# new_arr=[]
# for val in df['Income']:
#     if val=='under $25,000':
#         new_arr.append(0)
#         # df.set_value(index,'Income',0)
#     elif val=='$25,001 - $50,000':
#         new_arr.append(0)
#         # df.set_value(index,'Income',1)
#     elif val=='$50,000 - $74,999':
#         new_arr.append(0)
#         # df.set_value(index,'Income',2)
#     elif val=='$75,000 - $100,000':
#         new_arr.append(0)
#         # df.set_value(index,'Income',3)
#     elif val=='$100,001 - $150,000':
#         new_arr.append(0)
#         # df.set_value(index,'Income',4)
#     elif val=='over $150,000':
#         new_arr.append(0)
#         # df.set_value(index,'Income',5)

# for val in df['Income'].unique():
#     if isinstance(val, str) or isinstance(val, int):
#         if (val=='$25,001 - $50,000'):
#              vall='$37,500'
#         elif (val=='$75,000 - $100,000'):
#              vall='$87,500'
#         elif (val=='$50,000 - $74,999'):
#              vall='$62,500'
#         elif (val=='$100,001 - $150,000'):
#              vall='$125,000'
#         else:
#             vall=val
#         indexs.append(vall)
#         index.append(val)
# format={x:int('%d' % i)+1 for i,x in enumerate(index)}
# #print (dict)
# df['Income']=df['Income'].map(format)
# print (index)
# print (indexs)

df = df.as_matrix()
print (df.shape)

plt.figure(1, figsize=(15,4))
plt.yticks([0,1,2,3,4,5,6],indexs)
plt.scatter(df[:,0],df[:,1],alpha=.3)
plt.xlabel('YOB')
plt.ylabel('Income')
plt.title('Relationship between YOB and income')
plt.savefig('./scatter.png')
plt.show()
