# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 15:43:16 2020

@author: Administrator
"""
import numpy as np

def addcol(matrix1,matrix2):   #将两个行数相同的矩阵按列合并并返回
        [m1,n1] = np.shape(matrix1)
        [m2,n2] = np.shape(matrix2)
        if m1 != m2:
            print('diffrent rows,can not merge matrix')
            return
        mergMat = np.zeros((m1,n1 + n2))
        mergMat[:,0:n1] = matrix1[:,0:n1]
        mergMat[:,n1:(n1+n2)] = matrix2[:,0:n2]
        return mergMat

def create(n,top):    #生成随机点，n为个数，top为坐标最大值
    m = top/2.0
    data = top * np.random.random(size=(n,2))
    kind = np.ones((n,1))
    for i in range(len(kind)):
        if data[i,0]>m and data[i,1]>m:
            kind[i,0] = 0
        elif data[i,0]<m and data[i,1]<m:
            kind[i,0] = 0
        else:
            kind[i,0] = 1
    d = addcol(data,kind)
    return d


def createTest(n,top):
    m = top/2.0
    data = top * np.random.random(size=(n,2))
    kind = np.ones((n,1))
    for i in range(len(kind)):
        if data[i,0]>m and data[i,1]>m:
            kind[i,0] = 0
        elif data[i,0]<m and data[i,1]<m:
            kind[i,0] = 0
        else:
            kind[i,0] = 1
    return data,kind.T
    