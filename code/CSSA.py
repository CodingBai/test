import csv
import math

import torch
import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preload_exercise_vectors(filepath):
    exercise_vectors = {}
    with open(filepath, 'r') as f1:
        for line in f1:
            line = line.strip()
            parts = line.split(',')
            e_id = int(parts[0])
            vector_elements = list(map(int, parts[1:]))
            if len(vector_elements) == 130:
                exercise_vectors[e_id] = torch.tensor(vector_elements, dtype=torch.float32).to(device)
    return exercise_vectors

exercise_vectors = preload_exercise_vectors('/mahua/MulOER-SAN/data/exercise_vectors.txt')

def get_exercise_vector(exercise_id, exercise_vectors, mapping_dict):
    e_id = mapping_dict.get(exercise_id)
    if e_id is None or e_id not in exercise_vectors:
        return torch.zeros(130).to(device)
    return exercise_vectors[e_id]

def cal_fitness_q(X, exercise_vectors, mapping_dict):
    rounded_X = [int(x.item()) for x in X]
    vectors = [get_exercise_vector(x, exercise_vectors, mapping_dict) for x in rounded_X]
    vectors = torch.stack(vectors)
    distances = torch.cdist(vectors, vectors, p=2)
    fitness_value = torch.sum(distances) / 2
    return fitness_value

def SinMapping(Max_iter):
    x = torch.zeros((Max_iter, 1), device=device)
    x[0] = torch.rand(1, device=device) * 2 - 1
    for i in range(Max_iter - 1):
        x[i+1] = torch.sin(2 / x[i])
    return x

def initial(pop, dim, ub, lb):
    X = torch.zeros((pop, dim), device=device)
    for i in range(pop):
        SinValue = SinMapping(dim)
        for j in range(dim):
            X[i, j] = SinValue[j] * (ub[j] - lb[j]) + lb[j]
            X[i, j] = torch.clamp(X[i, j], lb[j].item(), ub[j].item())
    return X, lb, ub

def BorderCheck(X, ub, lb):
    for i in range(X.shape[0]):
        X[i, :] = torch.max(torch.min(X[i, :], ub), lb)
    return X

def CaculateFitness(X, fun, exercise_vectors, mapping_dict):
    pop = X.shape[0]
    fitness = torch.zeros((pop, 1), device=device)
    for i in range(pop):
        fitness[i] = fun(X[i, :], exercise_vectors, mapping_dict)
    return fitness

def SortFitness(Fit):
    fitness, index = torch.sort(Fit, descending=True)
    return fitness, index

def SortPosition(X, index):
    Xnew = torch.zeros_like(X)
    for i in range(X.shape[0]):
        Xnew[i, :] = X[index[i], :]
    return Xnew

def binary_encoding(continuous_x, num_bits, lb, ub):
    scaled_x = (continuous_x - lb) / (ub - lb)
    binary = torch.floor(scaled_x * (2 ** num_bits - 1))
    return binary

def binary_decoding(binary_x, num_bits, lb, ub):
    scaled_x = binary_x / (2 ** num_bits - 1)
    return torch.round(scaled_x * (ub - lb) + lb)

def remove_duplicates_and_fix(X, lb, ub):
    rounded_X = torch.round(X).to(torch.int32)
    unique_X = torch.unique(rounded_X)
    if unique_X.size(0) < rounded_X.size(0):
        all_ids = torch.arange(1, ub[0].item() + 1, device=device)
        missing_ids = torch.tensor(np.setdiff1d(all_ids.cpu().numpy(), unique_X.cpu().numpy()), device=device)
        random_ids = missing_ids[torch.randperm(missing_ids.size(0))[:rounded_X.size(0) - unique_X.size(0)]]
        unique_X = torch.cat([unique_X, random_ids])
    return unique_X

def PDUpdate(X, PDNumber, ST, Max_iter, dim, lb, ub, num_bits):
    X_new = X.clone()
    R2 = random.random()
    for j in range(PDNumber):
        if R2 < ST:
            X_new[j, :] = X[j, :] * torch.exp(-j / torch.tensor(random.random() * Max_iter, device=device))
        else:
            X_new[j, :] = X[j, :] + torch.randn(dim, device=device)
        X_new[j, :] = binary_decoding(binary_encoding(X_new[j, :], num_bits, lb, ub), num_bits, lb, ub)
        X_new[j, :] = remove_duplicates_and_fix(X_new[j, :], lb, ub)
    return X_new

def JDUpdate(X, PDNumber, pop, dim, lb, ub, num_bits):
    X_new = X.clone()
    for j in range(PDNumber + 1, pop):
        if j > (pop - PDNumber) / 2 + PDNumber:
            X_new[j, :] = torch.randn(dim, device=device) * torch.exp((X[-1, :] - X[j, :]) / j ** 2)
        else:
            A = torch.ones(dim, device=device)
            A[torch.rand(dim, device=device) > 0.5] = -1
            AA = torch.outer(A, A)
            X_new[j, :] = X[1, :] + torch.abs(X[j, :] - X[1, :]) * AA.diagonal()
        X_new[j, :] = binary_decoding(binary_encoding(X_new[j, :], num_bits, lb, ub), num_bits, lb, ub)
        X_new[j, :] = remove_duplicates_and_fix(X_new[j, :], lb, ub)
    return X_new

def SDUpdate(X, pop, SDNumber, fitness, BestF, dim, lb, ub, num_bits):
    X_new = X.clone()
    Temp = list(range(pop))
    RandIndex = random.sample(Temp, pop)
    SDchooseIndex = RandIndex[:SDNumber]
    for j in range(SDNumber):
        if fitness[SDchooseIndex[j]] < BestF:
            X_new[SDchooseIndex[j], :] = X[1, :] + torch.randn(dim, device=device) * torch.abs(X[SDchooseIndex[j], :] - X[1, :])
        elif fitness[SDchooseIndex[j]] == BestF:
            K = 2 * random.random() - 1
            X_new[SDchooseIndex[j], :] = X[SDchooseIndex[j], :] + K * (
                torch.abs(X[SDchooseIndex[j], :] - X[-1, :]) / (fitness[SDchooseIndex[j]] - fitness[-1] + 10e-8))
        X_new[SDchooseIndex[j], :] = binary_decoding(binary_encoding(X_new[SDchooseIndex[j], :], num_bits, lb, ub), num_bits, lb, ub)
        X_new[SDchooseIndex[j], :] = remove_duplicates_and_fix(X_new[SDchooseIndex[j], :], lb, ub)
    return X_new

def CSSA(pop, dim, lb, ub, Max_iter, fun, exercise_vectors, mapping_dict, num_bits=8):
    ST = 0.8
    PD = 0.3
    SD = 0.4
    PDNumber = int(pop * PD)
    SDNumber = int(pop * SD)
    X, lb, ub = initial(pop, dim, ub, lb)
    fitness = CaculateFitness(X, fun, exercise_vectors, mapping_dict)
    fitness, sortIndex = SortFitness(fitness)
    X = SortPosition(X, sortIndex)
    GbestPositon = torch.zeros((1, dim), device=device)
    GbestPositon[0, :] = X[0, :]
    GbestScore = fitness[0]
    Curve = torch.zeros((Max_iter, 1), device=device)
    for i in range(Max_iter):
        print(f"Iteration {i+1}/{Max_iter}")
        BestF = fitness[0]
        X = PDUpdate(X, PDNumber, ST, Max_iter, dim, lb, ub, num_bits)
        X = JDUpdate(X, PDNumber, pop, dim, lb, ub, num_bits)
        X = SDUpdate(X, pop, SDNumber, fitness, BestF, dim, lb, ub, num_bits)
        X = BorderCheck(X, ub, lb)
        for j in range(pop):
            Temp = X[j, :].clone()
            fitnew = fun(Temp, exercise_vectors, mapping_dict)
            if fitnew > fitness[j]:
                X[j, :] = Temp
                fitness[j] = fitnew
        fitness, sortIndex = SortFitness(fitness)
        X = SortPosition(X, sortIndex)
        if fitness[0] >= GbestScore:
            GbestScore = fitness[0]
            GbestPositon[0, :] = X[0, :]
        Curve[i] = GbestScore
        # print(f"Best Fitness: {GbestScore.item()}, Best Position: {GbestPositon[0, :].tolist()}")
    return GbestScore, GbestPositon, Curve

def load_interactions(filename):
    interactions = {}
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            student_id = int(parts[0])
            interaction_ids = set(map(int, parts[1:]))
            interactions[student_id] = interaction_ids
    return interactions

def load_data(student_recommend_file,exercise_detail_file):
    student_recommend = {}
    with open(student_recommend_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过表头
        for line in reader:
            student_id = int(line[0])
            exercises = list(map(int, line[1].split(',')))
            student_recommend[student_id] = exercises

    exercise_detail = {}
    with open(exercise_detail_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过表头
        for line in reader:
            u_id = int(line[0])
            e_id = int(line[1])
            Dqjk = float(line[2])
            if u_id not in exercise_detail:
                exercise_detail[u_id] = {}
            exercise_detail[u_id][e_id] = Dqjk

    return student_recommend,exercise_detail

# 生成映射字典函数
def generate_id_mapping(exercises):
    old_ids = set(exercises)
    id_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(old_ids), 1)}
    mapping_dict = {new_id: old_id for old_id, new_id in id_mapping.items()}
    return id_mapping, mapping_dict


# 主流程
student_recommend_file = '/mahua/MulOER-SAN/save_data/student_recommend.csv'
exercise_detail_file = '/mahua/MulOER-SAN/data/exercise_detail.csv'
student_recommend,exercise_detail = load_data(student_recommend_file,exercise_detail_file)

# 随机选择100个学生
num_students = len(student_recommend)
selected_students = random.sample(list(student_recommend.keys()), 100)

# 设置参数
pop = 50
Max_iter = 200
dim = 20
lb = torch.tensor([1] * dim, dtype=torch.float32).to(device)
ub = torch.tensor([200] * dim, dtype=torch.float32).to(device)

# def process_student(student_id):
#     exercises = student_recommend[student_id]
#     id_mapping, mapping_dict = generate_id_mapping(exercises)
#     num_runs = 10
#     all_best_scores = []
#     all_best_positions = []
#
#     for _ in range(num_runs):
#         GbestScore, GbestPosition, Curve = CSSA(pop, dim, lb, ub, Max_iter, cal_fitness_q, exercise_vectors, mapping_dict)
#         all_best_scores.append(GbestScore)
#         all_best_positions.append(GbestPosition)
#
#     max_score = max(all_best_scores)
#     max_indices = [i for i, score in enumerate(all_best_scores) if score == max_score]
#     chosen_index = random.choice(max_indices)
#     best_position = all_best_positions[chosen_index][0]
#
#     rounded_X = [int(x.item()) for x in best_position]
#     recommended_old_ids = [mapping_dict[new_id] for new_id in rounded_X]
#
#     return student_id, recommended_old_ids[:20]
#
# # 并行处理每个学生的推荐任务
# recommendations = {}
# with ThreadPoolExecutor(max_workers=8) as executor:
#     futures = [executor.submit(process_student, student_id) for student_id in selected_students]
#     for future in futures:
#         student_id, recommended_exercises = future.result()
#         recommendations[student_id] = recommended_exercises
#
# # 保存推荐结果到 CSV 文件
# with open('/mahua/MulOER-SAN/save_data/final_recommendations.csv', 'w', newline='', encoding='utf-8') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(['student_id', 'recommended_exercises'])
#     for student_id, exercises in recommendations.items():
#         exercises_str = ','.join(map(str, exercises))
#         writer.writerow([student_id, exercises_str])
#
# print('save final_recommendations.csv')

# new
def process_student(student_id, run):
    exercises = student_recommend[student_id]
    id_mapping, mapping_dict = generate_id_mapping(exercises)
    GbestScore, GbestPosition, Curve = CSSA(pop, dim, lb, ub, Max_iter, cal_fitness_q, exercise_vectors, mapping_dict)

    rounded_X = [int(x.item()) for x in GbestPosition[0]]
    recommended_old_ids = [mapping_dict[new_id] for new_id in rounded_X]

    return student_id, recommended_old_ids[:20]


def process_students():
    num_runs = 10

    for run in range(num_runs):
        all_recommendations = {}

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(process_student, student_id, run) for student_id in selected_students]
            for future in futures:
                student_id, recommendations = future.result()
                all_recommendations[student_id] = recommendations

        # 保存本次run的所有学生的推荐结果到独立文件
        with open(f'/mahua/MulOER-SAN/save_data/recommendations_run{run}.csv', 'w', newline='',
                  encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['student_id', 'recommended_exercises'])
            for student_id, recommendations in all_recommendations.items():
                exercises_str = ','.join(map(str, recommendations))
                writer.writerow([student_id, exercises_str])

# 计算accuracy指标
def calculate_accuracy(student_recommendations, exercise_details):
    accuracy_results = {}
    num_students = 0
    total_accuracy = 0
    delta = 0.7
    M = 20
    accuracy_list = []
    for student_id, recommend_exercises in student_recommendations.items():
        u_id = int(student_id)
        total_score = 0
        for e_id in recommend_exercises[:20]:  # 取前20道题
            if u_id in exercise_details and e_id in exercise_details[u_id]:
                Dqjk = exercise_details[u_id][e_id]
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
    return average_accuracy,std_deviation

# before
# # 计算accuracy值
# accuracy_results, average_accuracy,std_deviation = calculate_accuracy(recommendations, exercise_detail)
# print("accuracy: ", average_accuracy)
# print("Standard Deviation: ", std_deviation)
#
# # 输出结果
# for student_id, accuracy in accuracy_results.items():
#     print(f"student:  {student_id}, Accuracy: {accuracy:.4f}")


# 计算novelty指标
import pandas as pd
import numpy as np
from sklearn.metrics import jaccard_score


def load_data():
    # 读取only_k.txt文件
    only_k_data = {}
    with open('/mahua/MulOER-SAN/data/only_k.txt', 'r') as file:
        for line in file:
            k, idx = line.strip().split(',')
            only_k_data[int(k)] = int(idx)

    # 读取e_k_mapping.csv文件
    e_k_mapping = pd.read_csv('/mahua/MulOER-SAN/data/e_k_mapping.csv')
    e_k_dict = {}
    for _, row in e_k_mapping.iterrows():
        e_id = int(row['e_id'])
        k_ids = [int(k) for k in row['k_ids'].split(",")]
        e_k_dict[e_id] = k_ids

    # 读取u_answer_e.txt文件
    u_answer_e = pd.read_csv('/mahua/MulOER-SAN/data/u_answer_e.txt', sep='\t', header=None, names=['student_id', 'e_id', 'attempts', 'score'])

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
        s_id = int(student_id)
        if s_id in student_vectors:
            student_vector = student_vectors[s_id]
            # print("u_vector: ",student_vector)
            novelty_sum = 0
            for e_id in exercise_ids:
                if e_id in exercise_vectors:
                    exercise_vector = exercise_vectors[e_id]
                    # print("e_vector: ",exercise_vector)
                    # 如果是 PyTorch 张量，将其转换为 NumPy 数组
                    if isinstance(student_vector, torch.Tensor):
                        student_vector = student_vector.cpu().numpy()
                    else:
                        student_vector = np.array(student_vector)

                    if isinstance(exercise_vector, torch.Tensor):
                        exercise_vector = exercise_vector.cpu().numpy()
                    else:
                        exercise_vector = np.array(exercise_vector)
                        
                    jaccsim = 1 - jaccard_score(student_vector, exercise_vector)
                    novelty_sum += jaccsim
            novelty_values.append(novelty_sum / M)

    average_novelty = sum(novelty_values) / len(novelty_values) if len(novelty_values) > 0 else 0

    # 计算标准差
    novelty_variance = sum((x - average_novelty) ** 2 for x in novelty_values) / len(novelty_values)
    novelty_std_deviation = math.sqrt(novelty_variance)

    return average_novelty, novelty_std_deviation


only_k_data, e_k_dict, u_answer_e = load_data()

exercise_vector_2 = create_exercise_vectors(e_k_dict, only_k_data)
student_vectors = create_student_vector(u_answer_e, exercise_vector_2, only_k_data)

# average_novelty,novelty_std_deviation = calculate_novelty(recommendations, student_vectors, exercise_vectors)
# print("Average Novelty: ", average_novelty)
# print("Novelty Standard Deviation: ", novelty_std_deviation)



def calculate_accuracy_and_novelty(file_prefix):
    num_runs = 10
    accuracy_results = []
    novelty_results = []

    for run in range(num_runs):
        file_path = f'{file_prefix}_run{run}.csv'
        recommendations = {}

        with open(file_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                student_id = row['student_id']
                recommended_exercises = list(map(int, row['recommended_exercises'].split(',')))
                recommendations[student_id] = recommended_exercises

        # 计算 accuracy
        avg_acc,std_acc = calculate_accuracy(recommendations, exercise_detail)
        accuracy_results.append((avg_acc,std_acc))

        # 计算 novelty
        avg_nov,std_nov = calculate_novelty(recommendations, student_vectors, exercise_vectors)
        novelty_results.append((avg_nov,std_nov))

    return accuracy_results, novelty_results


# 计算并保存最终的平均值和方差
def summarize_results(accuracy_results, novelty_results):
    num_runs = len(accuracy_results)

    # 提取 accuracy 的平均值和标准差
    avg_accuracies = [res[0] for res in accuracy_results]
    accuracy_std_deviation = [res[1] for res in accuracy_results]

    # 提取 novelty 的平均值和标准差
    avg_novelties = [res[0] for res in novelty_results]
    novelty_std_deviation = [res[1] for res in novelty_results]

    average_accuracy = sum(avg_accuracies) / num_runs
    accuracy_std_deviation = sum(accuracy_std_deviation) / num_runs

    average_novelty = sum(avg_novelties) / num_runs
    novelty_std_deviation = sum(novelty_std_deviation) / num_runs

    return average_accuracy, accuracy_std_deviation, average_novelty, novelty_std_deviation


# 执行学生推荐处理
process_students()

# 计算 accuracy 和 novelty
file_prefix = '/mahua/MulOER-SAN/save_data/recommendations'
accuracy_results, novelty_results = calculate_accuracy_and_novelty(file_prefix)

# 汇总结果
avg_accuracy, acc_std_dev, avg_novelty, nov_std_dev = summarize_results(accuracy_results, novelty_results)

print("Average Accuracy: ", avg_accuracy)
print("Accuracy Standard Deviation: ", acc_std_dev)
print("Average Novelty: ", avg_novelty)
print("Novelty Standard Deviation: ", nov_std_dev)

