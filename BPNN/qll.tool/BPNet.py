# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 15:02:58 2020

@author: Administrator
"""

import numpy as np
import sys

sys.path.append('qll.tool')

import CreateSet as cs

class BPNet(object):
    def __init__(self):    #构造函数:初始化BP网络的基本参数
        #主动设置的参数
        self.eb = 0.01   #误差容限：当误差小于此值则认为算法已经收敛
        self.iterator = 0 #算法收敛时的迭代次数，即实际迭代次数
        self.eta = 0.1  #步长（学习率）
        self.mc = 0.3   #动量因子：引入的一个调优参数
        self.maxiter = 2000000  #最大迭代次数
        self.nHidden = 4  #隐含层神经元个数
        self.nOut = 1   #输出层神经元个数
        
        
        #以下属性由系统自动生成
        self.errlist = []  #误差列表用于评估收敛
        self.dataMat = 0  #训练集
        self.classLabels = 0  #分类标签集
        self.nSampNum = 0   #训练集行数
        self.nSampDim = 0  #训练集列数
        
    def addcol(self,matrix1,matrix2):   #将两个行数相同的矩阵按列合并并返回
        [m1,n1] = np.shape(matrix1)
        [m2,n2] = np.shape(matrix2)
        if m1 != m2:
            print('diffrent rows,can not merge matrix')
            return
        mergMat = np.zeros((m1,n1 + n2))
        mergMat[:,0:n1] = matrix1[:,0:n1]
        mergMat[:,n1:(n1+n2)] = matrix2[:,0:n2]
        return mergMat
        
    def init_hiddenWB(self):     #初始化隐含层权重
        self.hi_w = 2.0*(np.random.rand(self.nHidden,self.nSampDim) - 0.5)
        self.hi_b = 2.0*(np.random.rand(self.nHidden,1) - 0.5)
        self.hi_wb = np.mat(self.addcol(self.hi_w,self.hi_b))
        
        
    def init_OutputWB(self):     #初始化输出层权重
        self.out_w = 2.0*(np.random.rand(self.nOut,self.nHidden) - 0.5)
        self.out_b = 2.0*(np.random.rand(self.nOut,1) - 0.5)
        self.out_wb = np.mat(self.addcol(self.out_w,self.out_b))
    
    
    def loadDataSet(self,n,top):    #生成并初始化训练集
        self.dataMat = []
        self.classLabels = []
        data = cs.create(n,top)   #随机生成二维点数据
        for k in range(len(data)):  #循环取出点坐标和类别标签
            self.dataMat.append([data[k,0],data[k,1],1.0])
            self.classLabels.append(int(data[k,2]))
        self.dataMat = np.mat(self.dataMat)
        m1,n1 = np.shape(self.dataMat)
        self.nSampNum = m1     
        self.nSampDim = n1 - 1
        
        
    def drawClassScatter(self,plt):   #按照不同类别绘制形状不同的散点图
        i = 0
        for mydata in self.dataMat:
            if self.classLabels[i] == 0:
                plt.scatter(mydata[0,0],mydata[0,1],c='blue',marker='o')
            else:
                plt.scatter(mydata[0,0],mydata[0,1],c='red',marker='s')
            i += 1
            
    def Logistic(self,net):    #激活函数：Logistic
        return 1.0/(1.0 + np.exp(-net))
    
    
    def dLogistic(self,net):    #激活函数Logistic的导函数
        return np.multiply(net,(1.0 - net))
    
    def errorfunc(self,inX):       #全局误差函数
        return np.sum(np.power(inX,2)) * 0.5
    
    def bpTrain(self):       #bp网络主函数：训练数据集
        SampIn = self.dataMat.T
        expected = np.mat(self.classLabels)
        self.init_hiddenWB()
        self.init_OutputWB()
        dout_wbold = 0.0 ; dhi_wbold = 0.0  #t-1次迭代默认权值
        for i in range(self.maxiter):
            self.iterator += 1
            
            
            #1.  工作信号正向传播
            
            #1.1  从输入层到隐含层：
            hi_input = self.hi_wb * SampIn
            hi_output = self.Logistic(hi_input)
            
            hi2out = self.addcol(hi_output.T,np.ones((self.nSampNum,1))).T
            
            #1.2  从隐含层到输出层
            out_input = self.out_wb * hi2out
            out_output = self.Logistic(out_input)
            
            
            
            #2  误差计算
            err = expected - out_output
            sse = self.errorfunc(err)
            self.errlist.append(sse)
            if sse <= self.eb:
                self.iterator = i
                break
            
            
            
            #3  误差信号反向传播
            
            DELTA = np.multiply(err,self.dLogistic(out_output))   #DELTA为输出层梯度
            delta = np.multiply(self.out_wb[:,:-1].T*DELTA,self.dLogistic(hi_output))  #delta为隐含层梯度
            
            dout_wb = DELTA * hi2out.T  #输出层微分
            dhi_wb = delta * SampIn.T   #隐含层微分
            
            if i==0:
                self.out_wb += self.eta * dout_wb
                self.hi_wb += self.eta * dhi_wb
            else:
                self.out_wb += (1.0 - self.mc)*self.eta*dout_wb + self.mc*dout_wbold
                self.hi_wb += (1.0 - self.mc)*self.eta*dhi_wb + self.mc*dhi_wbold
            dout_wbold = dout_wb;  dhi_wbold = dhi_wb
                
    def BPClassfier(self,start,end,steps=30):    #分类器函数
        x = np.linspace(start,end,steps)    #生成均匀的二维点
        xx = np.mat(np.ones((steps,steps)))
        xx[:,0:steps] = x
        yy = xx.T
        z = np.ones((len(xx),len(yy)))
        for i in range(len(xx)):  #使用训练好的模型对生成的点进行分类
            for j in range(len(yy)):
                xi = [] ; tauex = [] ; tautemp = []
                np.mat(xi.append([xx[i,j],yy[i,j],1]))
                hi_input = self.hi_wb * (np.mat(xi).T)
                hi_out = self.Logistic(hi_input)
                
                taumrow,taucol = np.shape(hi_out)
                tauex = np.mat(np.ones((1,taumrow + 1)))
                tauex[:,0:taumrow] = (hi_out.T)[:,0:taumrow]
                
                out_input = self.out_wb * (np.mat(tauex).T)
                out_output = self.Logistic(out_input)
                
                z[i,j] = out_output
        
        return x,z
    
    
    def calssify(self,testMat,kinds):
        
        repects = []
        
        row,col = np.shape(testMat)
        
        hi_input = self.hi_wb * np.mat(self.addcol(testMat,kinds.T)).T
        hi_out = self.Logistic(hi_input)
        
        hi2out = self.addcol(hi_out.T,np.ones((row,1)))
        
        out_input = self.out_wb * hi2out.T
        out_output = self.Logistic(out_input)
        
        for i in range(row):
            if out_output[0,i] > 0.5:
                repects.append(1)
            else:
                repects.append(0)
        
        num = 0
        for j in range(row):
            if repects[j]==kinds[0,j]:
                num += 1
        print(float(num/row))
    
    
    def classfyLine(self,plt,x,z):      #绘制分类线：使用等高线图
        plt.contour(x,x,z,1,colors='black')
            