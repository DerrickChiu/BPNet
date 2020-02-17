# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import sys
sys.path.append('../qll.tool')

import BPNet as bp
import CreateSet as cs

bpnet = bp.BPNet()

top = 3.0   #坐标上限

#生成数据
bpnet.loadDataSet(120,top)

#绘制训练集散点图
bpnet.drawClassScatter(plt)

#训练数据集
bpnet.bpTrain()


#绘制分类曲线
x,z = bpnet.BPClassfier(0,top)
bpnet.classfyLine(plt,x,z)

plt.show()


print(bpnet.iterator)