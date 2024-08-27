# krt 是答对考察K的习题的数量，也就是习题作对 & 习题包含K
# kat 是作答考察K的习题的次数，也就是 习题包含K，这里没考虑到重复作答
import csv

import pandas as pd

u_e_list = []
e_k_list = []

def load_data(u_e_path, e_k_path):
    with open(u_e_path,'r') as fu,open(e_k_path,'r') as fk:
        lines1 = fu.readlines()
        lines2 = fk.readlines()
        for line in lines1:
            line = line.strip().split('\t')
            if len(line) != 4:
                continue
            u_id = int(line[0])
            e_id = int(line[1])
            try_id = int(line[2])
            score_id = float(line[3])
            u_e_list.append([u_id,e_id,try_id,score_id])

        for li in lines2:
            li = li.strip().split('\t')
            if len(li) != 2:
                continue
            e_k_id = int(li[0])
            k_id = int(li[1])
            e_k_list.append([e_k_id,k_id])

    return u_e_list,e_k_list

def get_new_e_k_list(e_k_list):

    # 使用字典来收集每个习题对应的k_id，这里k_id可以视为知识点
    exercise_to_knowledge = {}

    # 遍历列表，填充字典
    for e_k_pair in e_k_list:
        exercise_id, knowledge_id = e_k_pair
        if exercise_id in exercise_to_knowledge:
            # 如果习题ID已存在，则添加新的知识点到该习题的知识点列表
            if knowledge_id not in exercise_to_knowledge[exercise_id]:
                exercise_to_knowledge[exercise_id].append(knowledge_id)
        else:
            # 如果习题ID不存在，则创建一个新的知识点列表
            exercise_to_knowledge[exercise_id] = [knowledge_id]

    # 计算唯一k_id的数量，实际上就是所有知识点的集合长度，但这里直接通过字典的keys获取并计算数量
    # unique_k_id_count = len(set(k for sublist in exercise_to_knowledge.values() for k in sublist))

    # 创建新列表，每个元素是一个元组，包含习题ID和它关联的所有知识点列表
    new_list = [(exercise_id, sorted(knowledge_list)) for exercise_id, knowledge_list in exercise_to_knowledge.items()]

    # print("唯一k_id的数量:", unique_k_id_count)
    # print("新的列表形式（习题ID, 习题包含的知识点列表）:", new_list)
    return new_list


def get_IWKC(u_e_list, new_e_k_list):

    # 首先从u_e_list和new_e_k_list中提取所有涉及的学生ID和知识点ID
    students = set()
    knowledge_points = set()

    for u_id, _, _, _ in u_e_list:
        students.add(u_id)

    for _, knowledge_list in new_e_k_list:
        knowledge_points.update(knowledge_list)

    # 初始化学生知识点统计字典，只针对实际出现的学生和知识点
    student_knowledge_stats = {}
    for u_id in students:
        for k_id in knowledge_points:
            student_knowledge_stats[(u_id, k_id)] = {"kat": 0, "krt": 0}

    # 遍历u_e_list，累积计算kat和krt
    for u_id, e_id, _, score_id in u_e_list:
        # 在new_e_k_list中找到与e_id匹配的习题及其知识点列表
        matching_entry = next((entry for entry in new_e_k_list if entry[0] == e_id), None)
        if matching_entry is not None:
            for k_id in matching_entry[1]:
                student_knowledge_stats[(u_id, k_id)]["kat"] += 1
                if score_id == 1.0:
                    student_knowledge_stats[(u_id, k_id)]["krt"] += 1
        else:
            print(f"Warning: No matching entry found for e_id {e_id} in new_e_k_list.")


    # 计算IWKC并输出到文件
    with open("student_knowledge_IWKC.csv", "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Student", "Knowledge", "IWKC"])

        for (u_id, k_id), stats in student_knowledge_stats.items():
            kat = stats["kat"]
            krt = stats["krt"]
            IWKC = 1 if kat == 0 else 1 - (krt / kat)
            writer.writerow([u_id, k_id, IWKC])

    print("IWKC计算完成，结果已输出到student_knowledge_IWKC.csv")


def calculate_IWKC_average(input_csv_path):
    # 读取CSV文件
    df = pd.read_csv(input_csv_path)

    # 假设CSV文件的列名为'Student', 'Knowledge', 'IWKC'
    # 对'IWKC'按'Knowledge'分组求和，然后计算平均值
    iwkc_avg = df.groupby('Knowledge')['IWKC'].sum() / df['Student'].nunique()

    # iwkc_avg保留小数点后三位
    iwkc_avg = iwkc_avg.round(3)
    # 将结果转换为DataFrame，准备输出到新CSV
    result_df = pd.DataFrame({'Knowledge': iwkc_avg.index, 'Average_IWKC': iwkc_avg.values})

    # 输出到新的CSV文件
    output_csv_path = 'knowledge_IWKC_average.csv'
    result_df.to_csv(output_csv_path, index=False)
    print(f"Average IWKC per Knowledge has been saved to {output_csv_path}")

def save_new_e_k_list_to_csv(new_e_k_list, output_filename='e_k_mapping.csv'):
    """
    将new_e_k_list的内容保存为CSV文件。
    :param new_e_k_list: 格式为[习题编号，[知识点1，知识点2...]]的列表
    :param output_filename: 输出CSV文件的名称，默认为'e_k_mapping.csv'
    """
    with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['e_id', 'k_ids']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for exercise_id, knowledge_ids in new_e_k_list:
            # 将知识点列表转换为字符串，以便于在CSV中表示
            knowledge_ids_str = ','.join(str(kid) for kid in knowledge_ids)
            writer.writerow({'e_id': exercise_id, 'k_ids': knowledge_ids_str})




if __name__ == '__main__':
    u_e_path = './data/u_answer_e.txt'
    e_k_path = './data/e_k_old_id.txt'
    u_e_list,e_k_list = load_data(u_e_path,e_k_path)
    new_e_k_list = get_new_e_k_list(e_k_list)

    # 调用函数保存数据到CSV
    save_new_e_k_list_to_csv(new_e_k_list)
    get_IWKC(u_e_list,new_e_k_list)
    # 调用函数
    calculate_IWKC_average('student_knowledge_IWKC.csv')