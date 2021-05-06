import math
import random

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class KNNClassifier:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.K = 11

    def fit(self, X, Y, K):
        self.X = X
        self.Y = Y
        self.K = K
        return self

    #获取第6个元素
    def takeSixth(self, elem):
        return elem[5]

    #KNN算法实现
    def KNN(self):
        X = self.X
        Y = self.Y
        K = self.K

        #改变维度
        T = []
        for i in range(0, len(Y)):
            T.append([Y[i]])
        #整合数据
        DD = np.concatenate((X, T), axis = 1)
        #print(DD)

        #选取元素
        #正确数
        sum = 0
        for i in X:
            KNN = []
            for x in DD:
                d = math.sqrt((i[0]-x[0]) ** 2 + (i[1]-x[1]) ** 2 + (i[2]-x[2]) ** 2 + (i[3]-x[3]) ** 2)
                KNN.append([x[0], x[1], x[2], x[3], int(x[4]), round(d, 2)])

            KNN.sort(key=self.takeSixth)
            #print(KNN)
            #元素自身
            b = KNN[0]
            
            KNN = KNN[1:K]
            #print(KNN)
            
            #检验
            labels = [0, 0, 0]
            for x in KNN:
                labels[x[4]] += 1
            if labels[0] == max(labels):
                #print("推断是山鸢尾")
                a = 0
            elif labels[1] == max(labels):
                #print("推断是变色鸢尾")
                a = 1
            else:
                #print("推断是维吉尼亚鸢尾")
                a = 2
            
            #检验结果
            if(a==b[4]):
                #print("推断正确")
                sum += 1
            #else:
                #print("推断错误")
        return sum

    def score(self):
        sum = self.KNN()
        print("score:%.4f" %(sum/len(self.X)))


def main():
    iris = load_iris()
    X = iris.data
    Y = iris.target
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, stratify=Y)

    model = KNNClassifier(X_train, Y_train)
    print("train ", end="")
    model.score()
    model.fit(X_test, Y_test, 11)
    print("test ", end="")
    model.score()

if __name__ == '__main__':
    main()