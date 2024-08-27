#用于处理结果文件
import pandas as pd
import os
import numpy as np

# # 选取20%的学生
# df = pd.read_csv('MOOPer/inter_over_50_with_new_score.csv')
# students_id = df.iloc[:, 1].unique()
# num_to_select = int(len(students_id) * 0.2)
# students_selected = np.random.choice(students_id, size=num_to_select, replace=False)
# with open('data/students_selected.txt', 'w') as file:
#     for student_id in students_selected:
#         file.write(f"{student_id}\n")


# # 拼接结果文件
# # 假设所有文件存储在名为 'output' 的文件夹中
# output_folder = 'ck_selected'
# # 读取学生编号
# students_selected = []
# with open('data/students_selected.txt', 'r') as file:
#     for line in file:
#         student_id = int(line.strip())  # 去除行末的换行符和空白符
#         students_selected.append(student_id)  # 将学生编号添加到列表中
# # 存储所有数据的列表
# all_data = []
# # 遍历所有学生编号，读取对应的文件并添加学生编号列
# for student_id in students_selected:
#     file_path = os.path.join(output_folder, f'knowledge_prediction_{student_id}.csv')
#     if os.path.exists(file_path):
#         student_data = pd.read_csv(file_path)
#         student_data.insert(0, 'Student_ID', student_id)  # 添加学生编号列为第一列
#         all_data.append(student_data)
# # 拼接所有数据
# combined_df = pd.concat(all_data, ignore_index=True)
# # 将第二列（知识点编号）调整为整型
# combined_df.iloc[:, 1] = combined_df.iloc[:, 1].astype(int)
# # 删除第二列（知识点编号）值为0的行
# combined_df = combined_df[combined_df.iloc[:, 1] != 0]
# # 修改列名
# combined_df.columns = ['Student', 'Knowledge', 'CK']
# # 保存处理后的CSV文件
# processed_file_path = 'data/original_s_k_CK.csv'
# combined_df.to_csv(processed_file_path, index=False)
# print(f'处理后的CSV文件已保存到 {processed_file_path}')


# 处理小数点位数
final_file_path = 'data/original_s_k_CK.csv'
final_df = pd.read_csv(final_file_path)
# 对第二列（CK）的数值保留到小数点后三位，不足的补0
final_df['CK'] = final_df['CK'].apply(lambda x: f"{x:.4f}")
# 保存处理后的CSV文件
processed_final_file_path = 'data/s_k_CK.csv'
final_df.to_csv(processed_final_file_path, index=False)
print(f'处理后的CSV文件已保存到 {processed_final_file_path}')

