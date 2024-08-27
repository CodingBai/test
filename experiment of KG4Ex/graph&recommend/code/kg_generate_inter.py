import pandas as pd

from process.utils import get_entity_by_path_col


def get_inter(inter_file_path,entity_output_path):
#     unique_rows.to_csv('../data/KG4Ex/data/'+'e_id_u_id'+'.txt', sep='\t', index=False)

    # 加载映射关系
    entity_map = pd.read_csv(entity_output_path, sep=',', names=['old_id', 'new_id'], dtype=str)
    # 根据old_id换成new_id
    entity_dict = entity_map.set_index('old_id')['new_id'].to_dict()
    # 根据new_id换成old_id
    # entity_dict = entity_map.set_index('new_id')['old_id'].to_dict()

    # 读取并转换 output.txt 文件中的 challenge_id 和 user_id
    data = pd.read_csv(inter_file_path, sep='\t', header=None, names=['challenge_id', 'user_id'], dtype=str)
    # data = pd.read_csv(inter_file_path, sep='\t', header=None, names=['e_id', 'k_id'], dtype=str)

    # 使用映射关系替换 old_id 为 new_id
    data['challenge_id'] = data['challenge_id'].map(entity_dict).fillna(data['challenge_id'])
    data['user_id'] = data['user_id'].map(entity_dict).fillna(data['user_id'])

    # data['e_id'] = data['e_id'].map(entity_dict).fillna(data['e_id'])
    # data['k_id'] = data['k_id'].map(entity_dict).fillna(data['k_id'])


    # 保存转换后的数据回 output.txt 文件
    # data.to_csv('../data/KG4Ex/data/e_k_old_id'+'.txt', sep='\t', index=False)
    data.to_csv('../data/KG4Ex/data/inter_e_u'+'.txt', sep='\t', index=False)
    # data.to_csv('../data/KG4Ex/data/inter_e_u_find_error'+'.txt', sep='\t', index=False)


if __name__ == '__main__':
    new_pre_saved_path = '../data/KG4Ex/data/'
    # inter_file_path = '../data/origin/inter_over_50_with_new_score.csv'
    inter_file_path = '../data/KG4Ex/data/'+'e_id'+'_'+'u_id'+'.txt'
    # for_find_error
    # inter_file_path = '../data/KG4Ex/data/inter_find_error.txt'
    # inter_file_path = '../data/relation/r3_e_k.csv'
    # entity_output_path = new_pre_saved_path+'entity/entity.txt'
    # test:只有k,e,u的编号
    entity_output_path = new_pre_saved_path+'only_k_e_u_id.txt'

    get_inter(inter_file_path,entity_output_path)