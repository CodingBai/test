import numpy as np
from fccdf.utils import NewBackCloudModel
from fccdf.utils import reconstruct_R
import MCMC_DINA


def cloud_cdf(score, q, lr_cloud_r, lr_cloud_En):

    k_num = len(q[0])
    # score是二维矩阵，行是学生个数，列是习题个数，取值就是学生在习题上的得分情况
    s_num = len(score)
    e_num = len(score[0])

    ''' 生成认知云 '''
    # 初始化学生对知识点的认知矩阵 R = 1 * s_num * k_num * 0
    # 其中的0应指该知识点对应的答题记录
    R = [[[] for i in range(k_num)] for j in range(s_num)]
    R_upperlimit, R_lowerlimit = np.zeros([s_num, k_num]), np.zeros([s_num, k_num])

    global_upperlimit = np.zeros(s_num)
    global_lowerlimit = np.zeros(s_num)

    for i in range(s_num):
        # 输入每个学生的得分记录 score[i]
        ex, en, he = NewBackCloudModel.new_backcloud_model(score[i])
        global_lowerlimit[i] = ex - lr_cloud_En * en - lr_cloud_r * he
        global_upperlimit[i] = ex + lr_cloud_En * en + lr_cloud_r * he
        print("C_global ==ex,en,he=== ", ex, en, he)

        R = reconstruct_R.get_R(score, q) # s * k * e
        # R = np.array(R)                   # 重构后的R是list类型，先转为array

        skills = MCMC_DINA.CD_by_MCMC_DINA(score, q)
        # print(skills)

        for i in range(s_num):
            for j in range(k_num):
                if len(R[i][j]) == 2:
                    R_lowerlimit[i][j] = np.min(R[i][j])   # 考虑
                    R_upperlimit[i][j] = np.max(R[i][j])
                elif len(R[i][j]) < 3:
                    R_lowerlimit[i][j] = global_lowerlimit[i] # 用该学生在所有题上的答题记录生成全局认知云
                    R_upperlimit[i][j] = global_upperlimit[i]

                else:  # 输入学生i在知识点j上的所有得分记录
                    ex, en, he = NewBackCloudModel.new_backcloud_model(R[i][j])
                    R_lowerlimit[i][j] = ex - lr_cloud_En * en - lr_cloud_r * he
                    R_upperlimit[i][j] = ex + lr_cloud_En * en + lr_cloud_r * he
                    print("Cik ==ex,en,he=== ",ex,en,he)

        # 对区间数为[0,1]的知识点，调用 MCMC-DINA
        # for i in range(len(R_lowerlimit)):
        #     for j in range(len(R_lowerlimit[0])):
        #         if R_lowerlimit[i][j] < 0 and R_upperlimit[i][j] > 1:
        #             R_lowerlimit[i][j] = skills[i][j] - np.random.uniform(0, 0.138)
        #             R_upperlimit[i][j] = skills[i][j] + np.random.uniform(0, 0.138)

        R_lowerlimit[R_lowerlimit < 0] = 0
        R_upperlimit[R_upperlimit > 1] = 1

        return R_lowerlimit, R_upperlimit, skills
