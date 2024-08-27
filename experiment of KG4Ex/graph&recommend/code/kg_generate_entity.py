import pandas as pd
from tqdm import tqdm

from process.utils import get_entity_by_path_col, get_entity_by_path_col_preList, \
    get_initial_relation_by_path_col_preList


def generate_entity(inter_file_path, pre_path):
    entity_list = []

    u_list = get_entity_by_path_col(inter_file_path,'user_id')
    # recommend_list = get_entity_by_path_col_preList('../data/KG4Ex/u_e_recommend.csv','u_id','recommend',u_list,pre_path)
    recommend_list = get_entity_by_path_col_preList('../data/KG4Ex/u_e_recommend_no_130.csv','u_id','recommend',u_list,pre_path)

    e_list1 = get_entity_by_path_col(inter_file_path,'challenge_id')
    # forget_list = get_entity_by_path_col_preList('../data/KG4Ex/u_e_forget.csv','e_id','forget',e_list1,pre_path)
    forget_list = get_entity_by_path_col_preList('../data/KG4Ex/u_e_forget_no_130.csv','e_id','forget',e_list1,pre_path)

    k_path = '../data/KG4Ex/k.csv'
    k_list = get_entity_by_path_col(k_path,'k_id')

    mastery_list = get_entity_by_path_col_preList('../data/KG4Ex/u_k_mastery.csv','k_id','mastery',k_list,pre_path)
    occur_list = get_entity_by_path_col_preList('../data/KG4Ex/u_k_occur.csv','k_id','occur',k_list,pre_path)

    for i in [k_list,e_list1,u_list,mastery_list,occur_list,forget_list,recommend_list]:
        entity_list.append(i)

    # 1,4,6,8的关系
    get_initial_relation_by_path_col_preList('../data/KG4Ex/u_k_mastery.csv','mastery','u_id',mastery_list,pre_path)
    get_initial_relation_by_path_col_preList('../data/KG4Ex/u_k_occur.csv','occur','u_id',occur_list,pre_path)
    # get_initial_relation_by_path_col_preList('../data/KG4Ex/u_e_recommend.csv','recommend','e_id',recommend_list,pre_path)
    # get_initial_relation_by_path_col_preList('../data/KG4Ex/u_e_forget.csv','forget','u_id',forget_list,pre_path)
    get_initial_relation_by_path_col_preList('../data/KG4Ex/u_e_recommend_no_130.csv','recommend','e_id',recommend_list,pre_path)
    get_initial_relation_by_path_col_preList('../data/KG4Ex/u_e_forget_no_130.csv','forget','u_id',forget_list,pre_path)



    entity_name = ['k','e','student','mastery','occur','forget','recommend']

    return entity_name,entity_list


def print_entity(all_entity_name, all_entity_list, entity_output_path, entity_startId_path):
    old_id = []
    begin = 0
    begin_list = []

    # 记录每一类实体的old_id和new_id，并记录其起始 new_id
    for i in tqdm(range(len(all_entity_list))):
        cur_list = all_entity_list[i]
        begin_list.append(begin)
        begin += len(cur_list)
        old_id[len(old_id):len(cur_list)] = cur_list

    data = pd.DataFrame(columns=['old_id', 'new_id'])
    new_entity = pd.DataFrame(columns=['entity_name', 'start_id'])

    data['old_id'] = old_id
    data['new_id'] = [i for i in range(0, len(old_id))]
    new_entity['entity_name'] = all_entity_name
    new_entity['start_id'] = begin_list

    data.to_csv(entity_output_path, index=False)
    new_entity.to_csv(entity_startId_path, index=False)




if __name__ == '__main__':
    # new_pre_saved_path = '../data/KG4Ex/data/'
    new_pre_saved_path = '../data/KG4Ex/data_no_130/'
    inter_file_path = '../data/origin/inter_over_50_with_new_score.csv'
    entity_output_path = new_pre_saved_path+'entity/entity.txt'
    entity_startId_path = new_pre_saved_path+'entity/entity_new_start_id.txt'

    # 生成实体
    entity_name, entity_list = generate_entity(inter_file_path, new_pre_saved_path)
    # print_entity(entity_name, entity_list, entity_output_path, entity_startId_path)
    print_entity(entity_name, entity_list, entity_output_path, entity_startId_path)
