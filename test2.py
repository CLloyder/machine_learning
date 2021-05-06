import math
import random

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

#获取第6个元素
def takeSixth(elem):
    return elem[5]

def KNN(X, Y):
    #改变维度
    T = []
    for i in range(0, len(Y)):
        T.append([Y[i]])
    #整合数据
    DD = np.concatenate((X, T), axis = 1)
    print(DD)


    #KNN算法实现
    #随机元素
    i = random.choice(X)
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

def main():
    iris = load_iris()
    X = iris.data
    Y = iris.target
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0, stratify=Y)

    KNN(X_train,Y_train)

if __name__ == '__main__':
    main()

