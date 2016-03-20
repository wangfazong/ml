#!/usr/bin/python
#-*-coding:utf-8-*-

from numpy import *
import matplotlib.pyplot as plt

import operator

class kNN:

    def classify(inX, dataSet, labels, k):
        dataSize = dataSet.shape[0]
        diffMat = tile(inX, (dataSize,1)) - dataSet
        sqDiffMat = diffMat ** 2
        sqDistance = sqDiffMat.sum(axis = 1)
        dis = sqDistance ** 0.5
        sortedDistIndices = dis.argsort()
        classCount = {}
        for i in range(k):
            votelabel = labels[sortedDistIndices[i]]
            classCount[votelabel] = classCount.get(votelabel, 0) + 1
        sortedClassCount = sorted(classCount.iteritems(),
                key = operator.itemgetter(1), reverse = True)
        return sortedClassCount

    def normalize(dataSet):
        minVals = dataSet.min(0)
        maxVals = dataSet.max(0)
        ranges = maxVals - minVals
        normDataSet = zeros(shape(dataSet))
        m = dataSet.shape[0]
        normDataSet = dataSet - tile(minVals, (m, 1))
        normDataSet = normDataSet / tile(ranges, (m, 1))
        return normDataSet, ranges, minVals

    def drawPic(dataSet):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(dataSet[:,1],dataSet[:,2])
        plt.show()

