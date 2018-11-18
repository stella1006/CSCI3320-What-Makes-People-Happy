import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

filename = '../data/train.csv'
df = pd.read_csv(filename,header=0)
df = df[['Gender', 'Happy']]
df = df.dropna()
women_happy=0
women_unhappy=0
man_happy=0
man_unhappy=0

grouped=df.groupby('Gender')
for name, group in grouped:
    print (name)
    # print (group)
    # print (group.shape)
    if (name=='Female'):
        women_happy=(group['Happy']==1).sum()
        women_unhappy=group.shape[0]-women_happy
        print (women_happy, women_unhappy)
        # print ("{0:.2f}%".format(women_happy*100))
    if (name=='Male'):
        man_happy=(group['Happy']==1).sum()
        man_unhappy=group.shape[0]-man_happy
        print (man_happy, man_unhappy)
        # print ("{0:.2f}%".format(man_happy*100))

labels = 'Happy', 'Unhappy'
sizes1 = [man_happy, man_unhappy]
sizes2 = [women_happy, women_unhappy]

# colors = ['yellowgreen', 'lightskyblue']
# explode = (0.1, 0)  # explode 1st slice

# Plot
the_grid = GridSpec(1, 2)
plt.figure(1, figsize=(6.5,3))

plt.subplot(the_grid[0, 0])
patches1,text,autotexts = plt.pie(sizes1, autopct='%1.1f%%', startangle=90)
plt.legend(patches1, labels, loc="best")
plt.title('Male')

plt.subplot(the_grid[0, 1])
patches2,text,autotexts = plt.pie(sizes2, autopct='%1.1f%%', startangle=90)
plt.legend(patches2, labels, loc="best")
plt.title('Female')

# plt.axis('equal')
plt.savefig('./piechart.png')
plt.show()
