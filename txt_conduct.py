import os
import numpy as np

source_file_path = '../tracker0/dancetrack0026.txt'  # 请替换为你的源文件路径

# 设定目标文件夹路径，用于存放新创建的文本文件
target_folder_path = '../output'  # 请替换为你的目标文件夹路径

# 确保目标文件夹存在
if not os.path.exists(target_folder_path):
    os.makedirs(target_folder_path)

out_line = []


with open(source_file_path, 'r', encoding='utf-8') as source_file:
    # 读取所有行
    lines = source_file.readlines()

    # 遍历数字1到9
    for i in range(1,303):
        out_line = []
        num = str(i)
        for line in lines:
            if line.startswith(num + ','):  # 检查行是否以特定数字和逗号开头
                out_line.append(line)
        if out_line:  # 如果找到匹配的行
            new_file_name = f"{i}.txt"
            # 构造完整的新文件路径
            new_file_path = os.path.join(target_folder_path, new_file_name)
            # 将匹配的行写入到新的文本文件中
            with open(new_file_path, 'w', encoding='utf-8') as new_file:
                for item in out_line:
                    new_file.write(item)