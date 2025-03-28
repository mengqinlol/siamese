import matplotlib.pyplot as plt

# 读取数据
file_path = 'loss.txt'  # 替换为你的文件路径
with open(file_path, 'r') as file:
    dense_data = [float(line.strip()) for line in file]

file_path = 'res_loss.txt'  # 替换为你的文件路径
with open(file_path, 'r') as file:
    res_data = [float(line.strip()) for line in file]
# 绘制图形
plt.figure(figsize=(10, 6))  # 设置图形大小
plt.plot(dense_data, marker='o', linestyle='-', color='b', label='DenseNet Loss', markersize = 2)
plt.plot(res_data, marker='o', linestyle='-', color='r', label='ResNet Loss', markersize = 2)

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