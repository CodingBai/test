
import pandas as pd
from utils import open_file, get_dict, get_dict_kg


# 单类生成关系方法
def print_r(read_path, r_id, pre_name1, pre_name2, col1, col2, save_path, pre_path):
    data = pd.read_csv(read_path)
    len_1 = len(data)

    data['pre1'] = pre_name1
    data['pre2'] = pre_name2
    data['entity1'] = data['pre1'] + data[col1].apply(str)
    data['r'] = r_id
    data['entity2'] = data['pre2'] + data[col2].apply(str)
    data['entity1'] = data['entity1'].map(new_id_dict)
    data['entity2'] = data['entity2'].map(new_id_dict)

    data = data.drop_duplicates(subset=['entity1','r','entity2'])
    len_2 = len(data)

    print('\tdelete ', len_1-len_2)

    data.to_csv(pre_path+'relation/'+save_path+'.csv', index=False, columns=['entity1', 'r', 'entity2'], header=None)
    print(r_id,'\tDone!')



# 所有关系处理函数
def get_relation(pre_path):
    # 读一个文件，根据两个列名获得关系
    print_r(pre_path + 'old_relation/e_id_forget.csv', 0, 'e_', 'forget_', 'e_id', 'forget',
            'r0_e_forget', pre_path)
    print_r(pre_path + 'old_relation/forget_u_id.csv', 1, 'forget_', 'u_', 'forget', 'u_id',
            'r1_forget_u_id', pre_path)
    print_r(pre_path + 'old_relation/k_id_mastery.csv', 2, 'k_', 'mastery_', 'k_id', 'mastery',
            'r2_k_mastery', pre_path)
    print_r(pre_path + 'old_relation/k_id_occur.csv', 3, 'k_', 'occur_', 'k_id', 'occur',
            'r3_k_id_occur', pre_path)
    print_r(pre_path + 'old_relation/mastery_u_id.csv', 4, 'mastery_', 'u_', 'mastery', 'u_id',
            'r4_mastery_u_id', pre_path)
    print_r(pre_path + 'old_relation/occur_u_id.csv', 5, 'occur_', 'u_', 'occur', 'u_id',
            'r5_occur_u_id', pre_path)
    print_r(pre_path + 'old_relation/recommend_e_id.csv', 6, 'recommend_', 'e_', 'recommend', 'e_id',
            'r6_recommend_e_id', pre_path)
    print_r(pre_path + 'old_relation/u_id_recommend.csv', 7, 'u_', 'recommend_', 'u_id', 'recommend',
            'r7_u_id_recommend', pre_path)



if __name__ == '__main__':

    new_pre_saved_path = '../data/KG4Ex/data/'
    # 新老id映射字典
    new_id_dict = get_dict_kg(new_pre_saved_path+'entity/entity.txt', new_pre_saved_path+'entity/entity_new_start_id.txt')
    # get_e_k()
    get_relation(pre_path=new_pre_saved_path)
    # divide_all_data('../final_data/train_test/Ks_15_80.txt', '../final_data/train_test/')
