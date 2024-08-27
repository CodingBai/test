import csv

def multiply_third_column_values(file1, file2, output_file):
    # 读取第一个文件的内容
    data1 = {}
    with open(file1, mode='r', newline='') as csvfile1:
        reader1 = csv.reader(csvfile1)
        next(reader1)  # 跳过表头
        for row in reader1:
            key = (row[0], row[1])  # 将前两列作为键
            data1[key] = float(row[2])  # 第三列为值

    # 读取第二个文件的内容
    data2 = {}
    with open(file2, mode='r', newline='') as csvfile2:
        reader2 = csv.reader(csvfile2)
        next(reader2)  # 跳过表头
        for row in reader2:
            key = (row[0], row[1])  # 将前两列作为键
            data2[key] = float(row[2])  # 第三列为值

    # 创建新的CSV文件
    with open(output_file, mode='w', newline='') as csvfile_out:
        writer = csv.writer(csvfile_out)
        writer.writerow(['Student', 'Knowledge', 'PKC'])  # 写入表头

        # 遍历第一个文件的数据
        for key in data1.keys() & data2.keys():  # 只处理两个文件都存在的键
            student, knowledge = key
            pkc = round(data1[key] * data2[key], 4)  # 计算第三列的乘积并保留四位小数
            writer.writerow([student, knowledge, pkc])

# 使用函数
file1 = './data/student_knowledge_IWKC.csv'
file2 = './new_data/s_k_CK.csv'
output_file = './new_data/student_knowledge_PKC.csv'

multiply_third_column_values(file1, file2, output_file)