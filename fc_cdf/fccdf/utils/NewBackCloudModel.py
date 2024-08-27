import numpy as np
import math


# 改进的逆向云发生器的算法
# 利用pow(a, b)函数即可。需要开a的r次方则pow(a, 1/r)
'''新的一维正态云逆向生成器
   input：x
   output:云模型的三参数
   （1）Ex估计值为输入x的均值
   （2）计算一阶样本绝对中心距和方差
   （3）En = np.sqrt(math.pi / 2) * center_distance
   （4）He = np.sqrt(s * s - En * En)
'''
def new_backcloud_model( x ):
    if len(x) == 0:
        return 0, 0, 0
    if len(x) == 1:
        return x[0], 0, 0
    Ex = np.mean(x) # 计算数据的均值为Ex
    center_distance = np.mean(np.abs(x - Ex)) # 一阶样本绝对中心距
    s = np.sum((x - Ex) * (x - Ex)) / (len(x) - 1)
    En = np.sqrt(math.pi / 2) * center_distance
    # print(s, En)
    He = np.sqrt(abs(s * s - En * En))
    return Ex, En, He
