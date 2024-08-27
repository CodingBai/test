import pandas as pd
import numpy as np
import math

# 读取学生历史学习互动数据
train_set_path = 'data/sorted_train_updated.csv'
train_set = pd.read_csv(train_set_path)

# 读取习题和知识点对应关系数据
knowledge_points_path = 'data/e_k_mapping.csv'
knowledge_points = pd.read_csv(knowledge_points_path)

# 初始化存储习题和知识点的对应关系
problem_to_knowledge = {}
for _, row in knowledge_points.iterrows():
    problem_id = int(row[0])
    knowledge_ids = list(map(int, row[1].split(',')))
    problem_to_knowledge[problem_id] = knowledge_ids

# 获取所有知识点的集合
all_knowledge_ids = set()
for knowledge_list in problem_to_knowledge.values():
    all_knowledge_ids.update(knowledge_list)

students = train_set.iloc[:, 1].unique()
for student_id in students:
    # 获取学生的历史答题编号
    student_data = train_set[train_set.iloc[:, 1] == student_id]
    # 计算遗忘率
    results = []
    calculated_knowledge = set()
    exercise_count = len(student_data)
    problem_ids = student_data.iloc[:, 2].values  # 按时间顺序获得的学生历史答题编号

    for t in range(exercise_count, 0, -1):
        problem_id = problem_ids[t - 1]
        if problem_id in problem_to_knowledge:
            for knowledge_id in problem_to_knowledge[problem_id]:
                if knowledge_id in calculated_knowledge:
                    continue
                old_t = t
                forgetting_rate = 1 - math.exp(-(exercise_count + 1 - old_t) / 0.41)
                results.append([student_id, knowledge_id, forgetting_rate])
                calculated_knowledge.add(knowledge_id)

    # 对没有做过的知识点，遗忘率设为1
    for knowledge_id in all_knowledge_ids:
        if knowledge_id not in calculated_knowledge:
            old_t = 0
            forgetting_rate = 1 - math.exp(-(exercise_count + 1 - old_t) / 0.41)
            results.append([student_id, knowledge_id, forgetting_rate])

    # 转换为DataFrame并保存为CSV文件
    results_df = pd.DataFrame(results, columns=['student_id', 'knowledge_id', 'forgetting_rate'])
    output_file_path = f'all_s_frkc/k_s_frkc_{student_id}.csv'
    results_df.to_csv(output_file_path, index=False)
