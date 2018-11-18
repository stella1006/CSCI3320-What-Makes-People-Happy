import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

filename = '../data/train.csv'
df = pd.read_csv(filename,header=0)
df = df[['Happy', 'Income']]
df = df.dropna()

y_happy=[None]*6
y_unhappy=[None]*6
indexs=['under $25,000', '$25,001 - $50,000', '$50,000 - $74,999', '$75,000 - $100,000', '$100,001 - $150,000', 'over $150,000']

grouped=df.groupby('Income')
for name, group in grouped:
    print (name)
    happy=(group['Happy']==1).sum()
    unhappy=(group['Happy']==0).sum()
    if name=='under $25,000':
        y_happy[0]=happy*1.0/(happy+unhappy)
        y_unhappy[0]=unhappy*1.0/(happy+unhappy)
    elif name=='$25,001 - $50,000':
        y_happy[1]=happy*1.0/(happy+unhappy)
        y_unhappy[1]=unhappy*1.0/(happy+unhappy)
    elif name=='$50,000 - $74,999':
        y_happy[2]=happy*1.0/(happy+unhappy)
        y_unhappy[2]=unhappy*1.0/(happy+unhappy)
    elif name=='$75,000 - $100,000':
        y_happy[3]=happy*1.0/(happy+unhappy)
        y_unhappy[3]=unhappy*1.0/(happy+unhappy)
    elif name=='$100,001 - $150,000':
        y_happy[4]=happy*1.0/(happy+unhappy)
        y_unhappy[4]=unhappy*1.0/(happy+unhappy)
    elif name=='over $150,000':
        y_happy[5]=happy*1.0/(happy+unhappy)
        y_unhappy[5]=unhappy*1.0/(happy+unhappy)


bar_width=0.35
opacity=0.8
x=np.arange(6)
fig, ax=plt.subplots(1,figsize=(15,4))
# plt.figure(1, figsize=(15,4))
rects1 = plt.bar(x, y_happy, bar_width,alpha=opacity,color='b',label='Happy')
rects1 = plt.bar(x+bar_width, y_unhappy, bar_width,alpha=opacity,color='r',label='Unhappy')

plt.xticks([0,1,2,3,4,5],indexs)

# plt.bar(x,y_happy,width)
plt.xlabel('Income')
plt.ylabel('Fraction of Happy/Unhappy')
plt.title('Relationship between YOB and income')
plt.legend()
plt.tight_layout()
plt.savefig('./barchart.png')
plt.show()
