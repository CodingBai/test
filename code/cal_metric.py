# 计算平均的指标
import os
import csv

def load_interactions(filename):
    interactions = {}
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            student_id = int(parts[0])
            interaction_ids = set(map(int, parts[1:]))
            interactions[student_id] = interaction_ids
    return interactions

def load_student_recommendations(file_path):
    student_recommend = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过表头
        for line in reader:
            student_id = int(line[0])
            exercises = line[1].split(',')
            exercises = [int(exercise) for exercise in exercises]
            student_recommend[student_id] = exercises
    return student_recommend

def calculate_metrics(recommendations, interactions, k=20):
    metrics_per_student = []

    for student_id, recommended_exercises in recommendations.items():
        history = interactions.get(student_id, set())
        intersection = set(recommended_exercises) & history
        precision = len(intersection) / k if len(recommended_exercises) >= k else len(intersection) / len(
            recommended_exercises)
        recall = len(intersection) / len(history) if history else 0
        f1 = (2.0 * precision * recall) / (precision + recall) if precision + recall > 0 else 0
        hit_ratio = 1 if intersection else 0

        metrics_per_student.append((precision, recall, f1, hit_ratio))

    avg_precision = sum(precision for precision, _, _, _ in metrics_per_student) / len(metrics_per_student)
    avg_recall = sum(recall for _, recall, _, _ in metrics_per_student) / len(metrics_per_student)
    avg_f1 = sum(f1 for _, _, f1, _ in metrics_per_student) / len(metrics_per_student)
    avg_hit_ratio = sum(hit_ratio for _, _, _, hit_ratio in metrics_per_student) / len(metrics_per_student)

    return avg_precision, avg_recall, avg_f1, avg_hit_ratio

def calculate_average_metrics(folder_path, num_runs=10):
    total_precision, total_recall, total_f1, total_hit_ratio = 0, 0, 0, 0

    for run in range(num_runs):
        file_path = os.path.join(folder_path, f'recommendations_run{run}.csv')
        recommendations = load_student_recommendations(file_path)
        precision, recall, f1, hit_ratio = calculate_metrics(recommendations, interactions, k=20)

        total_precision += precision
        total_recall += recall
        total_f1 += f1
        total_hit_ratio += hit_ratio

    avg_precision = total_precision / num_runs
    avg_recall = total_recall / num_runs
    avg_f1 = total_f1 / num_runs
    avg_hit_ratio = total_hit_ratio / num_runs

    return avg_precision, avg_recall, avg_f1, avg_hit_ratio

# 文件路径
folder_path = './new_data/data_platform/save_data/'
interactions = load_interactions("./new_data/data_platform/data/test_process.txt")

# 计算平均指标
precision, recall, f1, hr = calculate_average_metrics(folder_path, num_runs=10)

# 保存结果
with open('./new_data/data_platform/save_data/metrics.txt', 'w') as f:
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall: {recall}\n")
    f.write(f"F1 Score: {f1}\n")
    f.write(f"Hit Ratio @10: {hr}\n")

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Hit Ratio @10: {hr}")