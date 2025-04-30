import matplotlib.pyplot as plt

# 读取数据
file_path = 'CNN_model_animal_croped_loss.txt'  # 替换为你的文件路径
with open(file_path, 'r') as file:
    dense_data = [float(line.strip()) for line in file]

file_path = 'CNN_model_animal_not_croped_loss.txt'  # 替换为你的文件路径
with open(file_path, 'r') as file:
    res_data = [float(line.strip()) for line in file]

cut = 300
dense_data = dense_data[:cut]
res_data = res_data[:cut]

# 绘制图形
plt.figure(figsize=(10, 6))  # 设置图形大小
plt.plot(dense_data, marker='o', linestyle='-', color='b', label='croped Loss', markersize = 2)
plt.plot(res_data, marker='o', linestyle='-', color='r', label='not_croped Loss', markersize = 2)

# 添加标题和标签
plt.title('Loss over Epochs', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)


# 添加网格
plt.grid(True, linestyle='--', alpha=0.6)

# 显示图例
plt.legend()

# 显示图形
plt.tight_layout()
plt.show()