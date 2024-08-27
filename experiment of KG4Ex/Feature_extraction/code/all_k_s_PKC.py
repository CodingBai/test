import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 读取数据文件
sorted_train_set_file = 'data/sorted_train_updated.csv'
data_file = 'MOOPer/data.txt'
qqq_file = 'MOOPer/qqq.txt'
df_file = 'MOOPer/inter_over_50_with_new_score.csv'

train_df = pd.read_csv(sorted_train_set_file)
data_matrix = np.loadtxt(data_file)
df = pd.read_csv(df_file)
with open(qqq_file, 'r') as file:
    qqq_matrix = [list(map(int, line.split())) for line in file.readlines()]

# 获取所有学生和习题编号
students_id = df.iloc[:, 1].unique()
questions_id = df.iloc[:, 2].unique()

# 创建学生编号和习题编号的索引字典
students = {student_id: idx for idx, student_id in enumerate(students_id)}
questions = {question_id: idx for idx, question_id in enumerate(questions_id)}

for student_id in students_id:
    student_records = train_df[train_df.iloc[:, 1] == student_id]
    # 提取该学生回答过的习题编号
    answer_seq = student_records.iloc[:, 2].tolist()

    # 提取 q_matrix 和 scores_matrix
    q_matrix = []
    scores_matrix = []
    for item_id in answer_seq:
        student_idx = students[student_id]
        item_idx = questions[item_id]
        q_matrix.append(qqq_matrix[item_idx])
        scores_matrix.append(data_matrix[student_idx, item_idx])

    q_matrix = np.array(q_matrix)
    scores_matrix = np.array(scores_matrix).reshape(-1, 1)

    # 将 scores_matrix 与 q_matrix 进行逐元素乘法
    final_input = scores_matrix * q_matrix

    # 定义 LSTM 模型
    class KnowledgeTracingLSTM(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(KnowledgeTracingLSTM, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])
            out = self.sigmoid(out)
            return out

    # 超参数
    input_size = final_input.shape[1]
    hidden_size = 200
    output_size = q_matrix.shape[1]
    num_epochs = 100
    batch_size = 64
    learning_rate = 0.001

    # 数据准备
    class StudentDataset(Dataset):
        def __init__(self, data, targets):
            self.data = data
            self.targets = targets

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx], self.targets[idx]

    dataset = StudentDataset(final_input, q_matrix)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 模型初始化
    model = KnowledgeTracingLSTM(input_size, hidden_size, output_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    model.train()
    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            inputs = inputs.float().unsqueeze(1)
            targets = targets.float()
            outputs = model(inputs)
            one_hot = outputs * targets
            one_matrix = torch.ones_like(inputs.squeeze(1))
            loss = criterion(one_hot.squeeze(1), one_matrix)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(final_input).float().unsqueeze(0)
        predictions = model(inputs).squeeze(0).numpy()

    # 保存结果为 CSV 文件
    result = []
    for knowledge_idx in range(predictions.shape[0]):
        result.append([student_id, knowledge_idx, predictions[knowledge_idx]])

    result_df = pd.DataFrame(result, columns=['student', 'knowledge', 'PKC'])
    output_file_path = f'all_k_s_PKC/k_s_PKC_{student_id}.csv'
    result_df.to_csv(output_file_path, index=False)