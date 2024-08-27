import csv
import random
import numpy as np

# 定义四个列表：EB,e_k_mapping,pkc,pkm
EB_list = []
e_k_list = {}
pkc_list = {}
pkm_list = {}
exercise_detail = []
student_recommend = []

def load_data_from_csv(EB, pkc, pkm, e_k_mapping):
    # 读取习题集
    with open(EB, 'r') as f1:
        lines1 = f1.readlines()
        for l1 in lines1[1:]:
            EB_list.append(int(l1.strip()))

    # 读取学生知识点覆盖范围
    with open(pkc, 'r') as f2:
        reader = csv.reader(f2)
        next(reader)
        for row in reader:
            student_id, knowledge_id, pkc_value = int(row[0]), int(row[1]), float(row[2])
            if student_id not in pkc_list:
                pkc_list[student_id] = {}
            pkc_list[student_id][knowledge_id] = pkc_value

    # 读取学生知识点掌握情况
    with open(pkm, 'r') as f3:
        reader = csv.reader(f3)
        next(reader)
        for row in reader:
            student_id, knowledge_id, proficiency = int(row[0]), int(row[1]), float(row[2])
            if student_id not in pkm_list:
                pkm_list[student_id] = {}
            pkm_list[student_id][knowledge_id] = proficiency

    # 读取习题和知识点映射
    with open(e_k_mapping, 'r') as f4:
        reader = csv.reader(f4)
        next(reader)
        for row in reader:
            exercise_id = int(row[0])
            knowledge_sequence = list(map(int, row[1].split(',')))
            e_k_list[exercise_id] = knowledge_sequence

    return EB_list, pkc_list, pkm_list, e_k_list

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2) if norm_vec1 != 0 and norm_vec2 != 0 else 0

def compute_qjk_vector(student_id, e_id, pkc_list, e_k_list):
    e_id = int(e_id)
    try:
        k_ids = e_k_list[e_id]
    except KeyError:
        print("no found e_id")
        return [0] * len(next(iter(pkc_list.values())).keys())

    new_pkc_list = {k: v.copy() for k, v in pkc_list.items()}

    if student_id in new_pkc_list:
        for k_id in new_pkc_list[student_id]:
            if k_id in k_ids:
                new_pkc_list[student_id][k_id] = 1
            else:
                new_pkc_list[student_id][k_id] = 0

    if student_id in new_pkc_list:
        return list(new_pkc_list[student_id].values())
    else:
        return [0] * len(next(iter(new_pkc_list.values())).keys())

def calculate_Dqj_k(e_id, pkm_list, e_k_list, student_id):
    e_id = int(e_id)
    k_ids = e_k_list[e_id]
    Rqjk = 1.0
    for k_id in k_ids:
        if student_id in pkm_list and k_id in pkm_list[student_id]:
            mastery = pkm_list[student_id][k_id]
            Rqjk *= mastery

    return 1 - Rqjk

def calculate_dis(Dqj_k):
    delta = 0.7
    return delta - Dqj_k

def calculate_o(sim, dis):
    Omega_qj_k = np.sqrt(np.square(sim) + np.square(dis))
    return Omega_qj_k

def generate_CS_batch(student_ids, EB_list, e_k_list, pkc_list, pkm_list, batch_size=100):
    student_recommend = []
    num_batches = (len(student_ids) + batch_size - 1) / batch_size

    for batch in range(int(num_batches)):
        print("batch: ",batch)
        batch_start = batch * batch_size
        batch_end = min((batch + 1) * batch_size, len(student_ids))
        batch_student_ids = student_ids[batch_start:batch_end]

        batch_recommendations = []
        for student_id in batch_student_ids:
            exercise_subset = {}
            for i in range(len(EB_list)):
                e_id = random.choice(EB_list)
                if check_interactions(student_id, e_id):
                    if student_id in pkc_list and pkc_list[student_id]:
                        coverage_vector = list(pkc_list[student_id].values())
                    else:
                        coverage_vector = []

                    qjk_vector = compute_qjk_vector(student_id, e_id, pkc_list, e_k_list)
                    sim = cosine_similarity(coverage_vector, qjk_vector)
                    Dqjk = calculate_Dqj_k(e_id, pkm_list, e_k_list, student_id)
                    dis = calculate_dis(Dqjk)
                    score_e_id = calculate_o(sim, dis)
                    exercise_subset[e_id] = score_e_id

            sorted_scores = sorted(exercise_subset.items(), key=lambda x: x[1])
            CS = [item[0] for item in sorted_scores[:200]]
            batch_recommendations.append([student_id, CS])

        student_recommend.extend(batch_recommendations)

    return student_recommend

def check_interactions(student_id, exercise_id):
    exercise_not_interacted = exercise_id is not None and student_id in student_exercise_interactions and exercise_id not in student_exercise_interactions[student_id]
    return exercise_not_interacted

def load_interactions(filename):
    interactions = {}
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            student_id = int(parts[0])
            interaction_ids = set(map(int, parts[1:]))
            interactions[student_id] = interaction_ids
    return interactions

if __name__ == "__main__":
    EB = '/mahua/MulOER-SAN/data/EB.txt'
    e_k_mapping = '/mahua/MulOER-SAN/data/e_k_mapping.csv'
    pkc = '/mahua/MulOER-SAN/data/student_knowledge_PKC.csv'
    pkm = '/mahua/MulOER-SAN/data/s_k_PKm.csv'

    EB_list, pkc_list, pkm_list, e_k_list = load_data_from_csv(EB, pkc, pkm, e_k_mapping)

    student_exercise_interactions = load_interactions('/mahua/MulOER-SAN/data/train_process.txt')

    student_ids = list(student_exercise_interactions.keys())
    student_recommend = generate_CS_batch(student_ids, EB_list, e_k_list, pkc_list, pkm_list, batch_size=100)

    with open('/mahua/MulOER-SAN/save_data/student_recommend.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['student_id', 'exercise'])
        for row in student_recommend:
            student_id, exercises = row
            exercises_str = ','.join(map(str, exercises))
            writer.writerow([student_id, exercises_str])
