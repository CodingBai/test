# NK = 130
import pandas as pd
import numpy as np

# 读取文件
e_s_EFR_NK_130 = pd.read_csv('new_relations_original_data/e_s_EFR_NK_130.csv')
k_s_MLKC = pd.read_csv('new_relations_original_data/k_s_MLKC.csv')
k_s_PKC = pd.read_csv('new_relations_original_data/k_s_PKC.csv')
df = pd.read_csv('MOOPer/inter_over_50_with_new_score.csv')

# 读取qqq.txt文件
qqq_path = 'MOOPer/qqq.txt'
with open(qqq_path, 'r') as file:
    qqq = np.array([list(map(int, line.split())) for line in file.readlines()])

# 获取学生编号，试题编号并建立试题编号对应字典
students = df.iloc[:, 1].unique()
excercises = df.iloc[:, 2].unique()
exercises_to_idx = {item: idx for idx, item in enumerate(excercises)}

# 计算余弦相似度的函数
def cosine_similarity_numpy(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

for student_id in students:
    # 提取学生对所有知识点的掌握程度并求积
    mlkc_data = k_s_MLKC[k_s_MLKC.iloc[:, 0] == student_id].iloc[:, 2].values
    MLKC_product = np.prod(mlkc_data)

    # 提取学生做下一题时所有知识点出现的概率
    pkc_data = k_s_PKC[k_s_PKC.iloc[:, 0] == student_id].iloc[:, 2].values

    # 初始化结果存储结构
    results = []

    # 遍历每个习题，计算推荐得分
    for excercise_id in excercises:
        item_idx = exercises_to_idx[excercise_id]
        q_matrix = qqq[item_idx]
        # 计算 cos_similarity
        cos_similarity = cosine_similarity_numpy(q_matrix, pkc_data)
        item_cos_similarity = np.square(cos_similarity)

        # 获取学生对习题的遗忘率EFR
        forgetting_rate_row = e_s_EFR_NK_130[
            (e_s_EFR_NK_130.iloc[:, 0] == student_id) &
            (e_s_EFR_NK_130.iloc[:, 1] == excercise_id)
            ]
        EFR = forgetting_rate_row.iloc[0, 2]

        # 计算推荐得分
        item_MLKC = np.square(0.7 - MLKC_product)
        item_EFR = np.square(0.7 - EFR)
        score = np.sqrt(item_MLKC + item_cos_similarity + item_EFR)
        results.append([student_id, excercise_id, score])

    # 转换为DataFrame
    results_df = pd.DataFrame(results, columns=['student', 'excercise', 'recommend'])
    output_file_path = f'all_e_s_recommend_NK_130/e_s_recommend_{student_id}.csv'
    results_df.to_csv(output_file_path, index=False)