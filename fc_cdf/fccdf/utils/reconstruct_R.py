import numpy as np

def get_R(score, q):

    s_num = len(score)
    e_num = len(score[0])
    # q[0] 是指有多少个知识点，是一个二维矩阵，第一维是有多少个习题
    k_num = len(q[0])
    # R是三维矩阵，第一维是有多少个学生，s_num,第二维是知识点个数，k_num,第三维空的
    R = [[[] for j in range(k_num)] for i in range(s_num)]

    trans_q = np.transpose(q)  # q: k*e
    print("重建R的Q矩阵",q.shape[0],q.shape[1])
    k_to_e = []

    # 对每一个 k 来说，对应考察该知识点的试题标号是哪些 t_q[i][要得到的题号]
    # np.where(trans_q[i])：矩阵t_q[i]中值为1的对应位置的index
    # 这里应该是知识点对应的所有试题
    for i in range(k_num):
        k_to_e.append(np.where(trans_q[i]==1)[0])

    # 将学生在对应知识点上做了哪些题更新到 R 上
    for i in range(s_num):
        for k in range(k_num):
            e_id = k_to_e[k] # e_id: numpy.ndarray；k_to_e：list
            R[i][k] = score[i,e_id]

    return R