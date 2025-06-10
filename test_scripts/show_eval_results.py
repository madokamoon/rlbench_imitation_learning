import matplotlib.pyplot as plt
import re
import numpy as np


# 原始CSV数据
csv_data = """klweight,default,smooth_attention,onlymask
1,0.8,0.82
2,0.72
3,0.78
5,0.64,0.8
6,0.7
7,0.64
10,0.62,0.88,0.8
11,0.72,0.82
17,0.72,0.84
25,0.54,0.88
36,0.56
37,null,0.74
46,0.62
47,0.7"""

# 解析数据
lines = csv_data.strip().split('\n')
header = lines[0].split(',')
data_rows = [line.split(',') for line in lines[1:]]

# 初始化变量
klweight = []
default = []
smooth_attention = []
onlymask = []

# 填充数据
for row in data_rows:
    klweight.append(int(row[0]))
    
    # 处理default列
    if len(row) > 1:
        default.append(None if row[1] == "null" else float(row[1]) if row[1] else None)
    else:
        default.append(None)
    
    # 处理smooth_attention列
    if len(row) > 2:
        smooth_attention.append(None if row[2] == "null" else float(row[2]) if row[2] else None)
    else:
        smooth_attention.append(None)
    
    # 处理onlymask列
    if len(row) > 3:
        onlymask.append(None if row[3] == "null" else float(row[3]) if row[3] else None)
    else:
        onlymask.append(None)

# 输出结果验证
print("klweight =", klweight)
print("default =", default)
print("smooth_attention =", smooth_attention)
print("onlymask =", onlymask)



# 创建图表
plt.figure(figsize=(12, 7))

# 绘制三条折线
plt.plot(klweight, default, marker='o', linestyle='-', label='Default', color='blue')
plt.plot(klweight, smooth_attention, marker='s', linestyle='--', label='Smooth Attention', color='red')
plt.plot(klweight, onlymask, marker='^', linestyle='-.', label='Only Mask', color='green')

# 添加标签和标题
plt.xlabel('KL Weight', fontsize=12)
plt.ylabel('Success Rate', fontsize=12)
plt.title('Success Rates for Different Methods with Various KL Weights', fontsize=14)

# 添加网格和图例
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)

# 设置x轴刻度为所有的klweight值
plt.xticks(klweight)

# 设置y轴范围为0-1
plt.ylim(0, 1)

# 保存图像
plt.tight_layout()
plt.savefig('method_comparison.png', dpi=300, bbox_inches='tight')
plt.show()