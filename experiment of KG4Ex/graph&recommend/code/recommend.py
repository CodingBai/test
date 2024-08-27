import codecs
import torch
import numpy as np
import random
import pandas as pd
from collections import defaultdict

# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

entities2id = {}
relations2id = {}
triple_list = []

def find_by_head_tail(head, tail, triple_list):
    for triple in triple_list:
        if triple[0] == head and triple[2] == tail:
            return triple[1]
    return None

def load_data(file1, file2, file3):
    entity_embeddings = []
    relation_embeddings = []
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        lines1 = f1.readlines()
        for line in lines1:
            line = line.strip().split('\t')
            if len(line) != 2:
                continue
            vector_floats = [float(num) for num in line[1].strip('[]').split(', ')]
            entity_embeddings.append(torch.tensor(vector_floats).to(device))

        lines2 = f2.readlines()
        for line in lines2:
            line = line.strip().split('\t')
            if len(line) != 2:
                continue
            vector_floats = [float(num) for num in line[1].strip('[]').split(', ')]
            relation_embeddings.append(torch.tensor(vector_floats).to(device))

    with codecs.open(file3, 'r') as f:
        content = f.readlines()
        for line in content:
            triple = line.strip().split(",")
            if len(triple) != 3:
                continue
            triple_list.append([triple[0], triple[1], triple[2]])

    return entity_embeddings, relation_embeddings, triple_list

batch_size = 10  # 设置批处理大小

def calculate_smlkc_spkc_sefr(entity_embedding, triple_list):
    smlkc = defaultdict(list)
    spkc = defaultdict(list)
    sefr = defaultdict(list)

    sample_size = int(len(student_ids) * 0.2)

    # 随机选取20%的学生ID
    sampled_student_ids = random.sample(student_ids, sample_size)
    tmp = 1

    for student_id in sampled_student_ids:

        print("print_cal: ", tmp)
        tmp = tmp + 1
        # 批处理知识点ID
        for i in range(0, len(knowledge_point_ids), batch_size):
            batch_k_ids = knowledge_point_ids[i:i+batch_size]
            print("batch_k_ids: ", batch_k_ids)
            k_embeddings = torch.stack([entity_embedding[k_id] for k_id in batch_k_ids]).to(device)
            m1_list = [find_by_head_tail(str(k_id), str(student_id), triple_list) for k_id in batch_k_ids]
            mastery_embeddings = torch.stack([entity_embedding[int(m1)].to(device) if m1 else torch.zeros_like(k_embeddings[j]) for j, m1 in enumerate(m1_list)]).to(device)
            for j, m1 in enumerate(m1_list):
                if not m1:
                    print(f"Using zero vector for mastery_embedding for student_id: {student_id}, knowledge_point_id: {batch_k_ids[j]}")
            smlkc_embeddings = k_embeddings + mastery_embeddings
            for j, k_id in enumerate(batch_k_ids):
                smlkc[student_id].append((k_id, smlkc_embeddings[j].cpu().numpy()))

        # 批处理知识点ID
        for i in range(0, len(knowledge_point_ids), batch_size):
            batch_k_ids = knowledge_point_ids[i:i+batch_size]
            k_embeddings = torch.stack([entity_embedding[k_id] for k_id in batch_k_ids]).to(device)
            p1_list = [find_by_head_tail(str(k_id), str(student_id), triple_list) for k_id in batch_k_ids]
            occur_embeddings = torch.stack([entity_embedding[int(p1)].to(device) if p1 else torch.zeros_like(k_embeddings[j]) for j, p1 in enumerate(p1_list)]).to(device)
            for j, p1 in enumerate(p1_list):
                if not p1:
                    print(f"Using zero vector for occur_embedding for student_id: {student_id}, knowledge_point_id: {batch_k_ids[j]}")
            spkc_embeddings = k_embeddings + occur_embeddings
            for j, k_id in enumerate(batch_k_ids):
                spkc[student_id].append((k_id, spkc_embeddings[j].cpu().numpy()))

        # 批处理习题ID
        for i in range(0, len(exercise_ids), batch_size):
            batch_e_ids = exercise_ids[i:i+batch_size]
            e_embeddings = torch.stack([entity_embedding[e_id] for e_id in batch_e_ids]).to(device)
            e1_list = [find_by_head_tail(str(e_id), str(student_id), triple_list) for e_id in batch_e_ids]
            forget_embeddings = torch.stack([entity_embedding[int(e1)].to(device) if e1 else torch.zeros_like(e_embeddings[j]) for j, e1 in enumerate(e1_list)]).to(device)
            for j, e1 in enumerate(e1_list):
                if not e1:
                    print(f"Using zero vector for forget_embedding for student_id: {student_id}, exercise_id: {batch_e_ids[j]}")
            sefr_embeddings = e_embeddings + forget_embeddings
            for j, e_id in enumerate(batch_e_ids):
                sefr[student_id].append((e_id, sefr_embeddings[j].cpu().numpy()))

    return smlkc, spkc, sefr, sampled_student_ids

def norm_l2(h, r, t):
    return torch.sum((h + r - t) ** 2).item()

def recommend_exercises(smlkc, spkc, sefr, entity_embedding, relation_embedding, triple_list,sampled_student_ids):
    recommendations = {}
    Nk = len(knowledge_point_ids)
    Ne = len(exercise_ids)

    tmp = 1
    for student_id in sampled_student_ids:
        print("print_recommend: ", tmp)
        tmp = tmp + 1
        scores = []
        for i in range(0, len(exercise_ids), batch_size):
            batch_e_ids = exercise_ids[i:i + batch_size]
            batch_e_ids = [e_id for e_id in batch_e_ids if check_interactions(student_id, e_id)]
            if not batch_e_ids:
                print("already interact")
                continue

            recommend_embeddings = []
            for exercise_id in batch_e_ids:
                r1 = find_by_head_tail(str(student_id), str(exercise_id), triple_list)
                recommend_embedding = entity_embedding[int(r1)].to(device) if r1 else torch.zeros_like(entity_embedding[0]).to(device)
                recommend_embeddings.append(recommend_embedding)
                if not r1:
                    print(f"Using zero vector for recommend_embedding for student_id: {student_id}, exercise_id: {exercise_id}")
            recommend_embeddings = torch.stack(recommend_embeddings).to(device)

            for exercise_id, recommend_embedding in zip(batch_e_ids, recommend_embeddings):
                smlkc_embeddings = torch.stack([torch.tensor(smlkc[student_id][j][1]).to(device) for j in range(len(smlkc[student_id]))])
                spkc_embeddings = torch.stack([torch.tensor(spkc[student_id][j][1]).to(device) for j in range(len(spkc[student_id]))])
                sefr_embeddings = torch.stack([torch.tensor(sefr[student_id][j][1]).to(device) for j in range(len(sefr[student_id]))])

                score_nk = (
                    torch.tensor(norm_l2(smlkc_embeddings, relation_embedding[7], recommend_embedding), device=device) +
                    torch.tensor(norm_l2(recommend_embedding, relation_embedding[6], entity_embedding[exercise_id]), device=device) +
                    torch.tensor(norm_l2(spkc_embeddings, relation_embedding[7], recommend_embedding), device=device) +
                    torch.tensor(norm_l2(recommend_embedding, relation_embedding[6], entity_embedding[exercise_id]), device=device)
                ) / Nk

                score_ne = (
                    torch.tensor(norm_l2(sefr_embeddings, relation_embedding[7], recommend_embedding), device=device) +
                    torch.tensor(norm_l2(recommend_embedding, relation_embedding[6], entity_embedding[exercise_id]), device=device)
                ) / Ne

                score = score_ne + score_nk
                scores.append((exercise_id, score.item()))

        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        top_20_exercises = [(exercise, score) for exercise, score in sorted_scores[:20]]
        recommendations[student_id] = top_20_exercises

    return recommendations

def calculate_metrics(recommendations, interactions, k=20):
    metrics_per_student = []

    for student_id, recommended_exercises in recommendations.items():
        history = interactions.get(student_id, set())
        intersection = set([exercise[0] for exercise in recommended_exercises]) & history
        precision = len(intersection) / k if len(recommended_exercises) >= k else len(intersection) / len(recommended_exercises)
        recall = len(intersection) / len(history) if history else 0
        f1 = (2.0 * precision * recall) / (precision + recall) if precision + recall > 0 else 0
        hit_ratio = 1 if intersection else 0

        metrics_per_student.append((precision, recall, f1, hit_ratio))

    avg_precision = sum(precision for precision, _, _, _ in metrics_per_student) / len(metrics_per_student)
    avg_recall = sum(recall for _, recall, _, _ in metrics_per_student) / len(metrics_per_student)
    avg_f1 = sum(f1 for _, _, f1, _ in metrics_per_student) / len(metrics_per_student)
    avg_hit_ratio = sum(hit_ratio for _, _, _, hit_ratio in metrics_per_student) / len(metrics_per_student)

    return avg_precision, avg_recall, avg_f1, avg_hit_ratio

# def norm_l2(h, r, t):
#     return torch.sum((h + r - t) ** 2).item()

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

def save_to_csv(data, filename):
    rows = []
    for key, values in data.items():
        for value in values:
            rows.append([key] + list(value))
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False, header=False)

def save_recommendations_to_csv(recommendations, filename):
    rows = []
    for student_id, recs in recommendations.items():
        for exercise_id, score in recs:
            rows.append([student_id, exercise_id, score])
    df = pd.DataFrame(rows, columns=["student_id", "exercise_id", "score"])
    df.to_csv(filename, index=False)

if __name__ == '__main__':
    file1 = "/mahua/transE-master/MOOPer_triple_student_6_4_130nk_TransE_entity_1000dim_batch1024"
    file2 = "/mahua/transE-master/MOOPer_triple_student_6_4_130nk_TransE_relation_1000dim_batch1024"
    file3 = "/mahua/transE-master/recent_6_4/student_6_4_130nk_add_all.csv"
    Nk = 130
    Ne = 478
    entity_embedding, relation_embedding, triple_list = load_data(file1, file2, file3)
    print("load data!")
    student_ids = list(range(608, 3139))
    knowledge_point_ids = list(range(0, 130))
    exercise_ids = list(range(130, 607))

    student_exercise_interactions = load_interactions('/mahua/transE-master/recent_6_4/train_converted.txt')

    print("cal_mlkc_pkc_efr")
    smlkc, spkc, sefr, sampled_student_ids = calculate_smlkc_spkc_sefr(entity_embedding, triple_list)
    print("Done!")
    save_to_csv(smlkc, '/mahua/transE-master/recent_6_4/smlkc.csv')
    save_to_csv(spkc, '/mahua/transE-master/recent_6_4/spkc.csv')
    save_to_csv(sefr, '/mahua/transE-master/recent_6_4/sefr.csv')

    print("cal_recommend!")
    recommendations = recommend_exercises(smlkc, spkc, sefr, entity_embedding, relation_embedding,triple_list,sampled_student_ids)
    save_recommendations_to_csv(recommendations, '/mahua/transE-master/recent_6_4/recommendations.csv')

    interactions = load_interactions("/mahua/transE-master/recent_6_4/test_converted.txt")

    print("cal_metrics!")
    precision, recall, f1, hr = calculate_metrics(recommendations, interactions, k=20)

    with open('/mahua/transE-master/recent_6_4/metrics.txt', 'w') as f:
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"F1 Score: {f1}\n")
        f.write(f"Hit Ratio @10: {hr}\n")

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Hit Ratio @10: {hr}")

