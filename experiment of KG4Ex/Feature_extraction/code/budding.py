import pandas as pd
import numpy as np
import os


# # 读取train_updated.txt文件,
# train_file_path = 'data/train_updated.txt'
# with open(train_file_path, 'r') as file:
#     train_data = [line.strip().split(',') for line in file]
# # 动态生成列名
# max_columns = max(len(row) for row in train_data)
# columns = ['student_id'] + [f'problem_id_{i}' for i in range(1, max_columns)]
# # 将train_data转换为DataFrame
# train_df = pd.DataFrame(train_data, columns=columns)
# # 读取inter_over_50_with_new_score.csv文件
# interactions_file_path = 'MOOPer/inter_over_50_with_new_score.csv'
# interactions_df = pd.read_csv(interactions_file_path)
# # 筛选出在train_updated.txt文件中出现的交互记录
# result_df = pd.DataFrame()
# for row in train_df.itertuples(index=False):
#     student_id = row[0]
#     problem_ids = row[1:]
#     for problem_id in problem_ids:
#         if pd.notna(problem_id) and problem_id:  # 处理空值和空字符串
#             problem_id = int(problem_id)  # 将问题ID转换为整数
#             matched_records = interactions_df[(interactions_df.iloc[:, 1] == int(student_id)) & (interactions_df.iloc[:, 2] == problem_id)]
#             result_df = pd.concat([result_df, matched_records])
# # 将结果保存到新的CSV文件中
# output_file_path = 'data/train_updated.csv'
# result_df.to_csv(output_file_path, index=False)
# print(f"匹配的记录已保存到 {output_file_path}")


# # 按时间顺序处理训练集
# # 读取CSV文件
# file_path = 'data/train_updated.csv'
# df = pd.read_csv(file_path)
# # 获取第11列的列名
# time_column_name = df.columns[10]
# # 将第11列转换为datetime类型
# df[time_column_name] = pd.to_datetime(df[time_column_name])
# # 按第11列时间先后进行排序
# df_sorted = df.sort_values(by=time_column_name)
# # 保存排序后的数据到新文件
# sorted_file_path = 'data/sorted_train_updated.csv'
# df_sorted.to_csv(sorted_file_path, index=False)
# print(f"数据已按时间排序并保存到 {sorted_file_path}")



# # 拼接frkc结果文件
# output_folder = 'all_s_frkc'
# # 读取学生编号
# inter_file_path = 'MOOPer/inter_over_50_with_new_score.csv'
# df = pd.read_csv(inter_file_path)
# students = df.iloc[:, 1].unique()
# # 存储所有数据的列表
# all_data = []
# # 遍历所有学生编号，读取对应的文件并添加学生编号列
# for student_id in students:
#     file_path = os.path.join(output_folder, f'k_s_frkc_{student_id}.csv')
#     if os.path.exists(file_path):
#         student_data = pd.read_csv(file_path)
#         all_data.append(student_data)
# # 拼接所有数据
# combined_df = pd.concat(all_data, ignore_index=True)
# # 保存到新的CSV文件
# combined_file_path = 'relations/k_s_frkc.csv'
# combined_df.to_csv(combined_file_path, index=False)
# print(f'所有CSV文件已拼接并保存到 {combined_file_path}')



# # 拼接EFR结果文件
# output_folder = 'all_e_s_EFR_NK_130'
# # 读取学生编号
# inter_file_path = 'MOOPer/inter_over_50_with_new_score.csv'
# df = pd.read_csv(inter_file_path)
# students = df.iloc[:, 1].unique()
# # 存储所有数据的列表
# all_data = []
# # 遍历所有学生编号,拼接所有数据
# for student_id in students:
#     file_path = os.path.join(output_folder, f'e_s_EFR_{student_id}.csv')
#     if os.path.exists(file_path):
#         student_data = pd.read_csv(file_path)
#         all_data.append(student_data)
# combined_df = pd.concat(all_data, ignore_index=True)
# # 保存到新的CSV文件
# combined_file_path = 'relations/e_s_EFR_NK_130.csv'
# combined_df.to_csv(combined_file_path, index=False)
# print(f'所有CSV文件已拼接并保存到 {combined_file_path}')



# # 拼接MLKC结果文件
# output_folder = 'all_k_s_MLKC'
# # 读取学生编号
# inter_file_path = 'MOOPer/inter_over_50_with_new_score.csv'
# df = pd.read_csv(inter_file_path)
# students = df.iloc[:, 1].unique()
# # 存储所有数据的列表
# all_data = []
# # 遍历所有学生编号,拼接所有数据
# for student_id in students:
#     file_path = os.path.join(output_folder, f'k_s_MLKC_{student_id}.csv')
#     if os.path.exists(file_path):
#         student_data = pd.read_csv(file_path)
#         all_data.append(student_data)
# combined_df = pd.concat(all_data, ignore_index=True)
# # 保存到新的CSV文件
# combined_file_path = 'relations/k_s_MLKC_no_knowledge_id.csv'
# combined_df.to_csv(combined_file_path, index=False)
# print(f'所有CSV文件已拼接并保存到 {combined_file_path}')



# # 拼接PKC结果文件
# output_folder = 'all_k_s_PKC'
# # 读取学生编号
# inter_file_path = 'MOOPer/inter_over_50_with_new_score.csv'
# df = pd.read_csv(inter_file_path)
# students = df.iloc[:, 1].unique()
# # 存储所有数据的列表
# all_data = []
# # 遍历所有学生编号,拼接所有数据
# for student_id in students:
#     file_path = os.path.join(output_folder, f'k_s_PKC_{student_id}.csv')
#     if os.path.exists(file_path):
#         student_data = pd.read_csv(file_path)
#         all_data.append(student_data)
# combined_df = pd.concat(all_data, ignore_index=True)
# # 保存到新的CSV文件
# combined_file_path = 'relations/k_s_PKC_no_knowledge_id.csv'
# combined_df.to_csv(combined_file_path, index=False)
# print(f'所有CSV文件已拼接并保存到 {combined_file_path}')


# # 对拼接好的MLKC和PKC结果文件继续进行处理，将拼接后的文件的知识点编号对应成新编号
# ks_mlkc_df = pd.read_csv('relations/k_s_MLKC_no_knowledge_id.csv')
# # 对知识点编号列（第二列）的数值加478
# ks_mlkc_df.iloc[:, 1] = ks_mlkc_df.iloc[:, 1] + 478
# # 将文件中的知识点编号进行对应
# new_to_old_map = {
#     478: 3, 479: 32, 480: 34, 481: 60, 482: 66, 483: 63, 484: 84, 485: 125,
#     486: 127, 487: 128, 488: 129, 489: 135, 490: 139, 491: 176, 492: 191,
#     493: 198, 494: 206, 495: 207, 496: 208, 497: 209, 498: 220, 499: 186,
#     500: 211, 501: 213, 502: 290, 503: 330, 504: 333, 505: 339, 506: 341,
#     507: 405, 508: 414, 509: 250, 510: 470, 511: 297, 512: 499, 513: 524,
#     514: 562, 515: 573, 516: 412, 517: 368, 518: 751, 519: 920, 520: 1049,
#     521: 549, 522: 1181, 523: 1182, 524: 1211, 525: 1032, 526: 1238, 527: 1259,
#     528: 1268, 529: 1276, 530: 1277, 531: 1217, 532: 1278, 533: 631, 534: 1328,
#     535: 1329, 536: 252, 537: 1381, 538: 1385, 539: 1386, 540: 1387, 541: 961,
#     542: 1402, 543: 361, 544: 1425, 545: 1553, 546: 1558, 547: 1559, 548: 1560,
#     549: 1563, 550: 980, 551: 375, 552: 189, 553: 1615, 554: 1033, 555: 212,
#     556: 210, 557: 1708, 558: 1720, 559: 347, 560: 1734, 561: 96, 562: 1043,
#     563: 1843, 564: 1484, 565: 1844, 566: 1845, 567: 378, 568: 1186, 569: 126,
#     570: 2305, 571: 2361, 572: 2362, 573: 2171, 574: 2172, 575: 2173, 576: 2175,
#     577: 2176, 578: 944, 579: 2157, 580: 2174, 581: 2380, 582: 2196, 583: 2487,
#     584: 2133, 585: 2135, 586: 1860, 587: 943, 588: 942, 589: 2141, 590: 699,
#     591: 2587, 592: 2588, 593: 304, 594: 2169, 595: 2170, 596: 616, 597: 2274,
#     598: 1064, 599: 1068, 600: 1069, 601: 2184, 602: 566, 603: 785, 604: 2956,
#     605: 416, 606: 358, 607: 1571
# }
# # 使用映射字典将新编号替换为旧编号
# ks_mlkc_df.iloc[:, 1] = ks_mlkc_df.iloc[:, 1].map(new_to_old_map)
# # 保存结果到新的CSV文件
# ks_mlkc_df.to_csv('relations/k_s_MLKC_knowledge_old_id.csv', index=False, header=['student', 'knowledge', 'MLKC'])
# print("新编号已替换为旧编号。")


# # 处理小数点位数
# final_file_path = 'relations_original_data/e_s_recommend_NK_130_original_data.csv'
# final_df = pd.read_csv(final_file_path)
# final_df['recommend'] = final_df['recommend'].apply(lambda x: f"{x:.5f}")
# # 保存处理后的CSV文件
# processed_final_file_path = 'new_relations_original_data/e_s_recommend_NK_130.csv'
# final_df.to_csv(processed_final_file_path, index=False)
# print(f'处理后的CSV文件已保存到 {processed_final_file_path}')



# # 拼接recommend_NK_130结果文件
# output_folder = 'all_e_s_recommend_NK_130'
# # 读取学生编号
# inter_file_path = 'MOOPer/inter_over_50_with_new_score.csv'
# df = pd.read_csv(inter_file_path)
# students = df.iloc[:, 1].unique()
# # 存储所有数据的列表
# all_data = []
# # 遍历所有学生编号,拼接所有数据
# for student_id in students:
#     file_path = os.path.join(output_folder, f'e_s_recommend_{student_id}.csv')
#     if os.path.exists(file_path):
#         student_data = pd.read_csv(file_path)
#         all_data.append(student_data)
# combined_df = pd.concat(all_data, ignore_index=True)
# # 保存到新的CSV文件
# combined_file_path = 'relations_original_data/e_s_recommend_NK_130_original_data.csv'
# combined_df.to_csv(combined_file_path, index=False)
# print(f'所有CSV文件已拼接并保存到 {combined_file_path}')


# # 拼接recommend结果文件
# output_folder = 'all_e_s_recommend'
# # 读取学生编号
# inter_file_path = 'MOOPer/inter_over_50_with_new_score.csv'
# df = pd.read_csv(inter_file_path)
# students = df.iloc[:, 1].unique()
# # 存储所有数据的列表
# all_data = []
# # 遍历所有学生编号,拼接所有数据
# for student_id in students:
#     file_path = os.path.join(output_folder, f'e_s_recommend_{student_id}.csv')
#     if os.path.exists(file_path):
#         student_data = pd.read_csv(file_path)
#         all_data.append(student_data)
# combined_df = pd.concat(all_data, ignore_index=True)
# # 保存到新的CSV文件
# combined_file_path = 'relations_original_data/e_s_recommend_original_data.csv'
# combined_df.to_csv(combined_file_path, index=False)
# print(f'所有CSV文件已拼接并保存到 {combined_file_path}')


