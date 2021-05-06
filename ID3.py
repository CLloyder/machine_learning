from math import log

import numpy as np
from sklearn.datasets import load_iris

#参考https://www.cnblogs.com/red-code/p/9856438.html
from sklearn.model_selection import train_test_split


def calcShannon(data):
    # 熵
    shannonMean = 0
    # 数据总量
    sumDataNum = len(data)
    # 数据集的所有分类情况
    classify = [man[-1] for man in data]
    # 循环每一种分类结果，计算该分类结果的熵，并求期望
    for resClassify in set(classify):
        # 该分类结果的“发生”概率
        p = classify.count(resClassify) / sumDataNum
        # 计算该分类结果的熵
        shannon = -log(p, 2)
        # 求期望
        shannonMean += p * shannon
    return shannonMean

def statisticsMostClassify(classify):
    map = {}
    for resClassify in classify:
        value = map.get(resClassify)
        if value:
            map[resClassify] = value + 1
        else:
            map[resClassify] = 1
    mostClassify = sorted(map.items(), key=lambda item: item[1])
    if mostClassify[-1][0] == 0:
        return "推断是山鸢尾"
    elif mostClassify[-1][0] == 1:
        return "推断是变色鸢尾"
    else:
        return "推断是维吉尼亚鸢尾"


def categorizationOfData(index, value, data):
    resData = []
    # 循环每一个样本，如果第index个特征的值符合指定特征，就把该特征删除后保存
    for x in data:
        if x[index] == value:
            tmp = x[:]
            resData.append(np.delete(tmp, index))
    return resData


def createTree(data, labels):
    classify = [x[-1] for x in data]
    # 如果当前数据集数据都是同一分类结果则不用继续分类
    if len(set(classify)) == 1:
        return statisticsMostClassify(classify)

    # 如果数据集中的样本没有特征了，则返回数据集中出现最多的分类结果
    if labels is None:
        return statisticsMostClassify(classify)

    # 最优特征（index下标）
    bestFeatureIndex = 0
    # 按照最优特征分类后的结果{分类结果值：分类后的数据集}
    bestClassifyData = {}
    originalShannon = calcShannon(data)
    diffShannon = 0
    for i in range(len(labels)):
        tmpClassifyData = {}
        # 取出第i个特征的所有可能值
        valueAll = [x[i] for x in data]
        valueSetAll = set(valueAll)
        classifyShannonMean = 0
        for value in valueSetAll:
            resData = categorizationOfData(i, value, data)  # 按该特征值分类后结果
            classifyShannon = calcShannon(resData)  # 分类后的熵
            classifyShannonMean += (valueAll.count(value) / len(valueAll)) * classifyShannon
            tmpClassifyData.update({value: resData})
        diff = originalShannon - classifyShannonMean
        if diff >= diffShannon:
            bestFeatureIndex = i
            bestClassifyData = tmpClassifyData
            diffShannon = diff
        bestFeature = labels[bestFeatureIndex]  # 获取最优特征的中文描述
        tmpLabels = labels[:]
        tmpLabels.pop(bestFeatureIndex)
        node = {bestFeature: {}}  # 当前节点
        for key in bestClassifyData:
            resClassify = createTree(bestClassifyData[key], tmpLabels)
            node[bestFeature].update({key: resClassify})
        return node

if __name__ == '__main__':
    iris = load_iris()
    X = iris.data
    Y = iris.target
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, stratify=Y)
    labels = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]

    T = []
    for i in range(0, len(Y_test)):
        T.append([Y_test[i]])
    # 整合数据
    DD = np.concatenate((X_test, T), axis=1)
    tree = createTree(DD, labels)
    print(tree)



