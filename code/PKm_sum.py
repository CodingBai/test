import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 读取训练集文件，训练集数据已按照时间进行排序
inter_file_path = 'data/sorted_train_updated.csv'
df = pd.read_csv(inter_file_path)

# 读取data.txt文件
data_file_path = 'MOOPer/data.txt'
data = np.loadtxt(data_file_path)

# 获取所有学生和习题编号,并创建索引字典
s_e_id_df = pd.read_csv('MOOPer/inter_over_50_with_new_score.csv')
students = s_e_id_df.iloc[:, 1].unique()
questions = s_e_id_df.iloc[:, 2].unique()
student_index = {student_id: idx for idx, student_id in enumerate(students)}
question_index = {question_id: idx for idx, question_id in enumerate(questions)}

# 习题包含的知识点文件
with open('data/e_k_old_id.txt', 'r') as file:
    lines = file.readlines()

# 读取知识点独热编码文件
one_hot_file_path = 'data/knowledge_point_one_hot_encoded.csv'
one_hot_df = pd.read_csv(one_hot_file_path, index_col=0)
knowledge_point_data = pd.read_csv(one_hot_file_path)
knowledge_point_ids = knowledge_point_data.iloc[:, 0].values
# 确保knowledge_point_ids长度为135，不足时用0补充
if len(knowledge_point_ids) < 135:
    knowledge_point_ids = np.pad(knowledge_point_ids, (0, 135 - len(knowledge_point_ids)), 'constant')

students_selected = []
with open('data/students_selected.txt', 'r') as file:
    for line in file:
        student_id = int(line.strip())
        students_selected.append(student_id)

for student_id in students_selected:
    # 提取学生的历史回答序列
    student_records = df[df.iloc[:, 1] == student_id]
    answer_sequence = student_records.iloc[:, 2].values
    max_length = 200  # 最长序列

    # 提取学生回答习题的得分
    scores = []
    for _, row in student_records.iterrows():
        question_id = row[2]
        student_idx = student_index[student_id]
        question_idx = question_index[question_id]
        score = data[student_idx, question_idx]
        scores.append([score])

    # 转换为 n*1 矩阵
    score_matrix = np.array(scores)
    # 使用 tanh 激活函数处理 score_matrix
    tanh_scores = np.tanh(score_matrix)
    if len(tanh_scores) < max_length:
        padding = np.zeros((max_length - len(tanh_scores), 1))
        tanh_scores = np.vstack((tanh_scores, padding))
    elif len(tanh_scores) > max_length:
        tanh_scores = tanh_scores[:max_length]

    # 构建习题编号到知识点独热编码向量的映射
    exercise_to_knowledge_one_hot = {}
    for exercise_id in answer_sequence:
        knowledge_points = []
        for line in lines:
            values = line.split()
            if values[0] == str(exercise_id):
                knowledge_points.append(int(values[1]))
        knowledge_one_hot_sum = np.zeros(one_hot_df.shape[1])
        for kp in knowledge_points:
            if kp in one_hot_df.index:
                knowledge_one_hot_sum += one_hot_df.loc[kp].values
        exercise_to_knowledge_one_hot[exercise_id] = knowledge_one_hot_sum

    # 构建E_K_t矩阵
    E_K_t = []
    for exercise_id in answer_sequence:
        if exercise_id in exercise_to_knowledge_one_hot:
            knowledge_one_hot_sum = exercise_to_knowledge_one_hot[exercise_id]
            E_K_t.append([exercise_id] + knowledge_one_hot_sum.tolist())
    E_K_t_matrix = np.array(E_K_t)
    # 对E_K_t矩阵进行处理，得到200行的嵌入矩阵，如果行数不足200，使用全零行进行填充；如果行数超过200，截取前200行
    if E_K_t_matrix.shape[0] < max_length:
        padding = np.zeros((max_length - E_K_t_matrix.shape[0], E_K_t_matrix.shape[1]))
        E_K_t_matrix = np.vstack((E_K_t_matrix, padding))
    else:
        E_K_t_matrix = E_K_t_matrix[:max_length, :]

    # 位置编码函数
    def position_encoding(seq_length, d_model):
        pos_enc = np.zeros((seq_length, d_model))
        for pos in range(seq_length):
            for i in range(0, d_model, 2):
                pos_enc[pos, i] = np.sin(pos / (10000 ** ((2 * i) / d_model)))
                if i + 1 < d_model:
                    pos_enc[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        return pos_enc

    # 生成位置编码矩阵P_t
    seq_length = len(answer_sequence)
    d_model = 130
    P_t = position_encoding(seq_length, d_model)

    # 处理P_t矩阵，使其行数变为200
    if len(P_t) < max_length:
        # 用零填充
        padding = np.zeros((max_length - len(P_t), d_model))
        P_t = np.vstack([P_t, padding])
    elif len(P_t) > max_length:
        # 截取前200行
        P_t = P_t[:max_length]

    # 将P_t矩阵与E_K_t_matrix相加，得到final_input
    final_input = E_K_t_matrix.copy()
    final_input[:, 1:1 + d_model] += P_t

    # 将 tanh_scores 与 final_input 进行逐元素乘法运算，得到最终的输入
    for i in range(final_input.shape[0]):
        final_input[i, :] = final_input[i, :] * tanh_scores[i, 0]


    class TransformerModel(nn.Module):
        def __init__(self, input_dim, nhead, num_encoder_layers, num_decoder_layers, hidden_dim, dropput):
            super(TransformerModel, self).__init__()
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=hidden_dim,
                                                            dropout=dropput)
            self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
            self.decoder_layer = nn.TransformerDecoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=hidden_dim,
                                                            dropout=dropput)
            self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, input_dim - 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, src, tgt):
            memory = self.encoder(src)
            output = self.decoder(tgt, memory)
            output = self.fc1(output)
            output = self.relu(output)
            output = self.dropout(output)
            output = self.fc2(output)
            output = self.fc3(output)
            output = self.sigmoid(output)
            return output

    # 超参数
    input_dim = 136  # 更新为新的维度
    nhead = 8
    num_encoder_layers = 2
    num_decoder_layers = 2
    hidden_dim = 100
    batch_size = 32
    learning_rate = 0.001
    dropout = 0.1
    num_epochs = 100

    # 补充新的维度，用0填充
    padding = np.zeros((final_input.shape[0], input_dim - final_input.shape[1]))
    final_input = np.hstack((final_input, padding))
    padding_E_k_t = np.zeros((E_K_t_matrix.shape[0], input_dim - E_K_t_matrix.shape[1]))
    excercise_one_hot = np.hstack((E_K_t_matrix, padding_E_k_t))

    # 将数据转换为张量
    input_data = torch.tensor(final_input, dtype=torch.float32)
    target_data = torch.roll(input_data, shifts=-1, dims=0)
    excercise_one_hot_data = torch.tensor(excercise_one_hot, dtype=torch.float32)

    # 数据集和数据加载器
    dataset = torch.utils.data.TensorDataset(input_data, target_data, excercise_one_hot_data)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型、损失函数和优化器
    model = TransformerModel(input_dim=input_dim, nhead=nhead, num_encoder_layers=num_encoder_layers,
                             num_decoder_layers=num_decoder_layers, hidden_dim=hidden_dim, dropput=dropout)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (src, tgt, e_one_hot) in enumerate(dataloader):
            optimizer.zero_grad()
            src = src.unsqueeze(1)
            tgt = tgt.unsqueeze(1)
            output = model(src, tgt)
            pkm_batch = output[:, -1, :]
            Rqtk = []
            true_labels = []
            for i in range(e_one_hot.size(0)):
                pkm = pkm_batch[i]
                e_one_hot_matrix = e_one_hot[i]
                excercise_id = int(e_one_hot_matrix[0].item())
                if excercise_id in questions:
                    one_hot = e_one_hot_matrix[1:]
                    # 查找对应得分存储在 true_labels 中
                    student_idx = student_index[student_id]
                    question_idx = question_index[exercise_id]
                    true_label = data[student_idx, question_idx]
                    true_labels.append(true_label)
                    # 计算 Rqtk
                    Rqtk_value = torch.prod(one_hot * pkm)
                    Rqtk.append(Rqtk_value)
            # 将 Rqtk 和 true_labels 转换为张量
            if len(Rqtk) > 0:
                Rqtk = torch.tensor(Rqtk, dtype=torch.float32)
                true_labels = torch.tensor(true_labels, dtype=torch.float32, requires_grad=True)
                # 计算损失：二元交叉熵损失
                loss = criterion(Rqtk, true_labels)
            else:
                # 如果 Rqtk 为空，则设定 loss = 0
                loss = torch.tensor(0.0, requires_grad=True)
            # 反向传播和优化
            loss.backward()
            optimizer.step()


    model.eval()
    with torch.no_grad():
        prediction = model(input_data.unsqueeze(1), input_data.unsqueeze(1))
        vt = prediction.squeeze(0).numpy()[-1]

    # 获取知识点编号并与预测结果关联
    output_data = np.vstack([knowledge_point_ids, vt])
    output_df = pd.DataFrame(output_data.T, columns=['Knowledge', 'PKm'])
    output_file_path = f'pkm_selected/knowledge_prediction_{student_id}.csv'
    output_df.to_csv(output_file_path, index=False)