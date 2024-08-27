# 计算novelty指标
import csv
import math

import pandas as pd
import numpy as np
from sklearn.metrics import jaccard_score


# 读取exercise_detail.csv文件
def load_exercise_details(file_path):
    exercise_detail = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过表头
        for line in reader:
            u_id = int(line[0])
            e_id = int(line[1])
            Dqjk = float(line[2])
            if u_id not in exercise_detail:
                exercise_detail[u_id] = {}
            exercise_detail[u_id][e_id] = Dqjk
    return exercise_detail

# 读取student_recommend.csv文件
def load_student_recommendations(file_path):
    student_recommend = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过表头
        for line in reader:
            student_id = int(line[0])
            exercise_id = int(line[1])

            if student_id not in student_recommend:
                student_recommend[student_id] = []

            student_recommend[student_id].append(exercise_id)

    return student_recommend

exercise_detail = load_exercise_details('../data/exercise_detail.csv')
recommendations = load_student_recommendations('recommendations_converted.csv')



def load_data():
    # 读取only_k.txt文件
    only_k_data = {}
    with open('../data/only_k.txt', 'r') as file:
        for line in file:
            k, idx = line.strip().split(',')
            only_k_data[int(k)] = int(idx)

    # 读取e_k_mapping.csv文件
    e_k_mapping = pd.read_csv('../data/e_k_mapping.csv')
    e_k_dict = {}
    for _, row in e_k_mapping.iterrows():
        e_id = int(row['e_id'])
        k_ids = [int(k) for k in row['k_ids'].split(",")]
        e_k_dict[e_id] = k_ids

    # 读取u_answer_e.txt文件
    u_answer_e = pd.read_csv('../data/u_answer_e.txt', sep='\t', header=None, names=['student_id', 'e_id', 'attempts', 'score'])

    return only_k_data, e_k_dict, u_answer_e


def create_130_dim_vector(only_k_data):
    vector = [0] * 130
    for k, idx in only_k_data.items():
        vector[idx] = 0
    return vector


def create_exercise_vectors(e_k_dict, only_k_data):
    exercise_vectors = {}
    for e_id, k_ids in e_k_dict.items():
        vector = create_130_dim_vector(only_k_data)
        for k in k_ids:
            if k in only_k_data:
                vector[only_k_data[k]] = 1
        exercise_vectors[e_id] = vector
    return exercise_vectors


def create_student_vector(u_answer_e, exercise_vectors, only_k_data):
    student_vectors = {}
    for student_id in u_answer_e['student_id'].unique():
        vector = create_130_dim_vector(only_k_data)
        student_exercises = u_answer_e[u_answer_e['student_id'] == student_id]
        for _, row in student_exercises.iterrows():
            e_id = row['e_id']
            score = row['score']
            if score == 1.0 and e_id in exercise_vectors:
                e_vector = exercise_vectors[e_id]
                for i in range(130):
                    if e_vector[i] == 1:
                        vector[i] = 1
        student_vectors[student_id] = vector
    return student_vectors


def calculate_novelty(recommendations, student_vectors, exercise_vectors):
    novelty_values = []
    M = 20
    for student_id, exercise_ids in recommendations.items():
        if student_id in student_vectors:
            student_vector = student_vectors[student_id]
            # print("u_vector: ",student_vector)
            novelty_sum = 0
            for e_id in exercise_ids:
                if e_id in exercise_vectors:
                    exercise_vector = exercise_vectors[e_id]
                    # print("e_vector: ",exercise_vector)
                    jaccsim = 1 - jaccard_score(student_vector, exercise_vector)
                    novelty_sum += jaccsim
            novelty_values.append(novelty_sum / M)
            # print("n_value",novelty_values)

    average_novelty = sum(novelty_values) / len(novelty_values) if len(novelty_values) > 0 else 0

    # 计算标准差
    novelty_variance = sum((x - average_novelty) ** 2 for x in novelty_values) / len(novelty_values)
    novelty_std_deviation = math.sqrt(novelty_variance)

    return average_novelty, novelty_std_deviation


only_k_data, e_k_dict, u_answer_e = load_data()

def calculate_accuracy(student_recommendations,e_k_dict,mastery_mapping):
    accuracy_results = {}
    num_students = 0
    total_accuracy = 0
    delta = 0.7
    M = 20
    accuracy_list = []
    for student_id, recommend_exercises in student_recommendations.items():
        total_score = 0
        for e_id in recommend_exercises[:20]:  # 取前20道题
            # 获取习题对应的知识点编号
            k_ids = e_k_dict.get(e_id, [])

            # 计算该习题包含的所有知识点的 mastery 值的乘积
            Dqjk = 1
            for k_id in k_ids:
                # 查找用户对该知识点的 mastery 值
                mastery_value = mastery_mapping.get((student_id, k_id), 0)  # 默认为0
                Dqjk *= mastery_value
            # if student_id in exercise_details and e_id in exercise_details[student_id]:
            #     Dqjk = exercise_details[student_id][e_id]
                score = 1 - abs(delta - Dqjk)
                total_score += score
        accuracy = total_score / M
        accuracy_results[student_id] = accuracy
        num_students += 1
        total_accuracy = total_accuracy + accuracy
        accuracy_list.append(accuracy)

    average_accuracy = total_accuracy / num_students if num_students > 0 else 0

    # 计算标准差
    variance = sum((x - average_accuracy) ** 2 for x in accuracy_list) / num_students
    std_deviation = math.sqrt(variance)
    return accuracy_results,average_accuracy,std_deviation


# 读取 u_k_mastery.csv 文件
mastery_df = pd.read_csv('u_k_mastery.csv', header=None, names=['u_id', 'k_id', 'mastery'])

# 将 mastery 列转换为浮点型
mastery_df['mastery'] = pd.to_numeric(mastery_df['mastery'], errors='coerce')

# 创建用户对知识点的掌握情况的映射字典
user_knowledge_mastery = mastery_df.groupby(['u_id', 'k_id'])['mastery'].mean().reset_index()
mastery_mapping = user_knowledge_mastery.set_index(['u_id', 'k_id'])['mastery'].to_dict()

# 计算accuracy值
accuracy_results, average_accuracy,std_deviation = calculate_accuracy(recommendations,e_k_dict,mastery_mapping)
print("accuracy: ", average_accuracy)
print("Standard Deviation: ", std_deviation)

# 输出结果
for student_id, accuracy in accuracy_results.items():
    print(f"student:  {student_id}, Accuracy: {accuracy:.4f}")

exercise_vectors = create_exercise_vectors(e_k_dict, only_k_data)
student_vectors = create_student_vector(u_answer_e, exercise_vectors, only_k_data)

average_novelty,novelty_std_deviation = calculate_novelty(recommendations, student_vectors, exercise_vectors)
print("Average Novelty: ", average_novelty)
print("Novelty Standard Deviation: ", novelty_std_deviation)


