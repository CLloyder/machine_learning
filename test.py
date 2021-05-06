import math
import random

import matplotlib.pyplot as plt
import numpy as np
import numpy as py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
print(iris.data)
print(iris.target)

#4.3-7.9
#X = [x[0] for x in DA]
#print(min(X))
#print(max(X))
##2.0-4.4
#Y = [x[1] for x in DA]
#print(min(Y))
#print(max(Y))
##1.0-6.9
#Z = [x[2] for x in DA]
#print(min(Z))
#print(max(Z))
##0.1-2.5
#R = [x[3] for x in DA]
#print(min(R))
#print(max(R))

#随机数
#i = [round(random.uniform(4.3, 7.9), 1), round(random.uniform(2, 4.4), 1), round(random.uniform(1, 6.9), 1), round(random.uniform(0.1, 2.5), 1)]
#print(i)

#数据集
DA = iris.data
TG = iris.target

X = iris.data
Y = iris.target
X_train,X_test, Y_train, Y_test =train_test_split(X,Y,test_size=0.2, random_state=0,stratify=Y)

#获取第6个元素
def takeSixth(elem):
    return elem[5]

#改变维度
T = []
for i in range(0, len(TG)):
    T.append([TG[i]])
#整合数据
DD = np.concatenate((DA, T), axis = 1)
print(DD)


#KNN算法实现
#随机元素
i = random.choice(DA)
print(i)
KNN = []
for x in DD:
    d = math.sqrt((i[0]-x[0]) ** 2 + (i[1]-x[1]) ** 2 + (i[2]-x[2]) ** 2 + (i[3]-x[3]) ** 2)
    KNN.append([x[0], x[1], x[2], x[3], int(x[4]), round(d, 2)])

KNN.sort(key=takeSixth)
print(KNN)
#随机元素自身
b = KNN[0]

KNN = KNN[1:6]
print(KNN)

#检验1
#labels = {"山鸢尾":0, "变色鸢尾":0, "维吉尼亚鸢尾":0}
#print(labels)
#for x in KNN:
#    if x[4] == 0:
#        labels["山鸢尾"] += 1
#    elif x[4] == 1:
#        labels["变色鸢尾"] += 1
#    else:
#        labels["维吉尼亚鸢尾"] += 1
#print(max(zip(labels.values(), labels.keys())))

#检验2
labels = [0, 0, 0]
for x in KNN:
    labels[x[4]] += 1
if labels[0] == max(labels):
    print("推断是山鸢尾")
    a = 0
elif labels[1] == max(labels):
    print("推断是变色鸢尾")
    a = 1
else:
    print("推断是维吉尼亚鸢尾")
    a = 2

#检验结果
if(a==b[4]):
    print("推断正确")
else:
    print("推断错误")





