import pandas as pd

# 读取学生历史学习互动数据
train_set_path = 'data/sorted_train_updated.csv'
train_set = pd.read_csv(train_set_path)

# 读取学生对知识点的遗忘率数据
frck_path = 'relations_original_data/k_s_frkc.csv'
frck_data = pd.read_csv(frck_path)

# 读取习题与知识点对应关系数据
knowledge_points_path = 'data/e_k_mapping.csv'
knowledge_points = pd.read_csv(knowledge_points_path)

# 初始化存储习题和知识点的对应关系
problem_to_knowledge = {}
for _, row in knowledge_points.iterrows():
    problem_id = int(row[0])
    knowledge_ids = list(map(int, row[1].split(',')))
    problem_to_knowledge[problem_id] = knowledge_ids

#获取学生编号
students = train_set.iloc[:, 1].unique()
for student_id in students:
    # 读取学生对知识点的遗忘率并转换为字典
    student_forgetting_rates = frck_data[frck_data.iloc[:, 0] == student_id]
    knowledge_forgetting_dict = dict(zip(student_forgetting_rates.iloc[:, 1], student_forgetting_rates.iloc[:, 2]))
    # 计算学生对每道习题的遗忘率
    results = []
    for problem_id, knowledge_ids in problem_to_knowledge.items():
        total_forgetting_rate = 0
        # NK = len(knowledge_ids)  # NK取该习题包含的知识点的数量
        NK = 130  # NK取知识点总数130
        for knowledge_id in knowledge_ids:
            forgetting_rate = knowledge_forgetting_dict.get(knowledge_id, 1)
            total_forgetting_rate += forgetting_rate
        EFR = (total_forgetting_rate / NK)
        results.append([student_id, problem_id, EFR])

    # 转换为DataFrame并保存为CSV文件
    results_df = pd.DataFrame(results, columns=['student', 'excercise', 'EFR'])
    # output_file_path = f'all_e_s_EFR/e_s_EFR_{student_id}.csv'
    output_file_path = f'all_e_s_EFR_NK_130/e_s_EFR_{student_id}.csv'
    results_df.to_csv(output_file_path, index=False)