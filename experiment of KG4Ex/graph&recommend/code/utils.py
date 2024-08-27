import numpy as np
import pandas as pd


def open_file(path, type_file):
    if type_file == 'txt':
        return np.loadtxt(path, delimiter=',', dtype=str, encoding='utf-8')
    elif type_file == 'lines':
        return open(path, 'r', encoding='utf-8').readlines()


def save_matrix(path, data, flag):
    print(len(data))
    with open(path, 'a', encoding='utf-8') as f:
        for i in range(len(data)):
            if i != 0:
                f.write('\n')
            for j in range(len(data[0])):
                if j == 0:
                    if flag == 'float':
                        f.write('{:.5f}'.format(data[i][j]))
                    else:
                        f.write('{:d}'.format(data[i][j]))
                else:
                    if flag == 'float':
                        f.write(',{:.5f}'.format(data[i][j]))
                    else:
                        f.write(',{:d}'.format(data[i][j]))


# 根据path读取文件，再根据列名找到这一列中得所有唯一值，将其再组合成列表
def get_entity_by_path_col(path, col_name):
    df = pd.read_csv(path)
    entity_list = df[col_name].unique().tolist()
    return entity_list

# 根据path读取文件，找出跟pre_list指定的pre_name一样的行，根据pre_name,col_name来去重，最后取col_name这两列
def get_entity_by_path_col_preList(path, pre_name, col_name, pre_list, pre_path):
    df = pd.read_csv(path)
    df = df[df[pre_name].isin(pre_list)]
    df.drop_duplicates(subset=[pre_name, col_name])
    df.to_csv(pre_path+'old_relation/'+pre_name+'_'+col_name+'.csv', index=False)
    entity_list = df[col_name].unique().tolist()
    return entity_list

def get_initial_relation_by_path_col_preList(path, pre_name, col_name, pre_list, pre_path):
    df = pd.read_csv(path)
    df = df[df[pre_name].isin(pre_list)]
    df.drop_duplicates(subset=[pre_name, col_name])
    df.to_csv(pre_path+'old_relation/'+pre_name+'_'+col_name+'.csv', index=False)

def get_dict_kg(entity_map_path, entity_start_id_path):
    all_arr = open_file(path=entity_map_path, type_file='lines')[1:]
    entity_id = pd.read_csv(entity_start_id_path)['start_id'].tolist()

    new_id_dict = {}
    new_id_dict['n_e'] = entity_id[1]

    n_entity = len(all_arr)
    new_id_dict['n_entity'] = n_entity

    # for i in range(len(all_arr)):
    for line in all_arr:
        temp = line.strip().split(',')
        i = int(temp[1])
        if i < entity_id[1]:
            key_s = 'k_'
        elif i >= entity_id[1] and i < entity_id[2]:
            key_s = 'e_'
        elif i >= entity_id[2] and i < entity_id[3]:
            key_s = 'u_'
        elif i >= entity_id[3] and i < entity_id[4]:
            key_s = 'mastery_'
        elif i >= entity_id[4] and i < entity_id[5]:
            key_s = 'occur_'
        elif i >= entity_id[5] and i < entity_id[6]:
            key_s = 'forget_'
        elif i >= entity_id[6]:
            key_s = 'recommend_'

    #
        # =========================================================
        new_id_dict[key_s + str(temp[0])] = i


    return new_id_dict

