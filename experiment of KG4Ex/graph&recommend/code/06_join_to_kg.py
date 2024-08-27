import pandas as pd
import os

'''
origin_dir: 指定多个csv文件所在的目录
'''
def join_to_one_file(origin_dir, to_file_path):

    # 获取目录下所有csv文件的文件名
    file_names = os.listdir(origin_dir)
    df_list = []

    # 循环读取每个csv文件的数据并添加到data_list列表中
    for file_name in file_names:
        if file_name.endswith('.csv') and file_names:
            file_path = os.path.join(origin_dir, file_name)
            cur_data = pd.read_csv(file_path, header=None)
            df_list.append(cur_data)

    # 使用pandas的concat函数将所有数据按行合并为一个DataFrame对象
    all_data = pd.concat(df_list, axis=0, ignore_index=True)

    # 将合并后的所有数据保存为一个csv文件
    all_data.to_csv(to_file_path, index=False, header=None)
    print('Done!')


if __name__ == '__main__':

    new_pre_saved_path = '../data/KG4Ex/data/'

    # # 按照学生个人交互记录划分的relation
    # join_to_one_file(new_pre_saved_path + './relation/', new_pre_saved_path + 'kg/student_6_4_130nk_all.csv')
    # 按照学生个人交互记录划分的relation
    join_to_one_file(new_pre_saved_path + './add_relation/', new_pre_saved_path + 'add_relation/student_6_4_130nk_add_all.csv')
    # join_to_one_file(new_pre_saved_path+'relation_for_node2vec/', new_pre_saved_path+'kg/kg_new_score_all_for_node2vec.csv')
