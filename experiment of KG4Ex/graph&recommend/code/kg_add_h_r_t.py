import pandas as pd
from utils import open_file, get_dict, get_dict_kg


# 单类生成关系方法
def print_r(read_path,pre_name1, pre_name2,pre_name3, col1, col2,col3, save_path, pre_path):
    data = pd.read_csv(read_path)
    len_1 = len(data)

    data['pre1'] = pre_name1
    data['pre2'] = pre_name2
    data['pre3'] = pre_name3
    data['entity1'] = data['pre1'] + data[col1].apply(str)
    data['r'] = data['pre3'] + data[col3].apply((str))
    data['entity2'] = data['pre2'] + data[col2].apply(str)
    data['entity1'] = data['entity1'].map(new_id_dict)
    data['entity2'] = data['entity2'].map(new_id_dict)
    data['r'] = data['r'].map(new_id_dict)

    data = data.drop_duplicates(subset=['entity1','r','entity2'])
    len_2 = len(data)

    print('\tdelete ', len_1-len_2)

    data.to_csv(pre_path+'add_relation/'+save_path+'.csv', index=False, columns=['entity1', 'r', 'entity2'], header=None)
    # data.to_csv(pre_path+'relation_for_node2vec/' + save_path + '.csv', index=False, columns=['entity1', 'entity2'],
    #             header=None)
    print('\tDone!')



# 所有关系处理函数
def get_relation(pre_path):
    # 读一个文件，根据两个列名获得关系
    # 现在修改为只有四种关系
    # 1. 知识点 掌握情况 学生
    # 2. 知识点 出现概率 学生
    # 3. 习题 遗忘率 学生
    # 4. 学生 推荐 习题
    # pre_name1:head pre_name2:tail pre_name3:relation
    print_r(pre_path + 'old_relation/e_id_forget.csv', 'e_', 'u_', 'forget_', 'e_id', 'u_id','forget',
            'r0_e_forget_u', pre_path)
    print_r(pre_path + 'old_relation/k_id_mastery.csv', 'k_', 'u_', 'mastery_','k_id', 'u_id','mastery',
            'r1_k_mastery_u', pre_path)
    print_r(pre_path + 'old_relation/k_id_occur.csv',  'k_', 'u_', 'occur_','k_id', 'u_id','occur',
            'r2_k_id_occur_u', pre_path)
    print_r(pre_path + 'old_relation/recommend_e_id.csv', 'u_','e_', 'recommend_', 'u_id','e_id', 'recommend',
            'r3_u_recommend_e_id', pre_path)


if __name__ == '__main__':

    new_pre_saved_path = '../data/KG4Ex/data/'
    # 新老id映射字典
    new_id_dict = get_dict_kg(new_pre_saved_path+'entity/entity.txt', new_pre_saved_path+'entity/entity_new_start_id.txt')
    # get_e_k()
    get_relation(pre_path=new_pre_saved_path)
    # divide_all_data('../final_data/train_test/Ks_15_80.txt', '../final_data/train_test/')
