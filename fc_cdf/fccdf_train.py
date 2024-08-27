import numpy as np
import torch

from fccdf import fccdf

def main():

    # 读取得分矩阵score data.txt; Q矩阵 qqq.txt
    # TestData testData2 : qqq.txt
    # FrcSub : q.txt
    # Math2 : q.txt
    # real_data
    # MOOPer
    path = "MOOPer"
    score = np.loadtxt("math2015/"+path+"/data.txt")
    score = np.array(score)
    q = np.loadtxt("math2015/"+path+"/qqq.txt")

    # 设置超参数：认知结果生成时，He和En的系数
    weight_he = 0.1
    weight_en = 0.2

    R_lowerlimit, R_upperlimit, skills = fccdf.cloud_cdf(score, q, weight_he, weight_en)


    np.set_printoptions(threshold=np.inf)


    print("============R_lowerlimit===============\n",R_lowerlimit)
    print("============R_upperlimit===============\n",R_upperlimit)

    R_meanlimit = (R_lowerlimit + R_upperlimit) / 2
    print("============R_meanlimit===============\n", R_meanlimit)
    torch.save(torch.tensor(R_meanlimit),"fc_cdf.pth")

    # 取中间值，存成.pth torch.tensor.save()
    # 低：1，3 高 4，8  取值，1，4取中间值=2.5， 3，8取中间值=5 【2.5，5】

    # #输出最终结果
    # for i in skills:
    #     for k in i:
    #         print("%.4f"%k)




    # for i in R_upperlimit:
    #     for k in i:
    #         print("%.4f"%k)

if __name__ == '__main__':
    main()
