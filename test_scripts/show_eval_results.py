import matplotlib.pyplot as plt
import re
import numpy as np

# 解析数据
lines = [
    "act_policy,push_button,policy_last.ckpt,kl_weight1,2025-06-09-11-52-37,False,100,50,0.8,0.18,52,50,0.036,1.0,0.075",
    "act_policy,push_button,policy_last.ckpt,kl_weight2,2025-06-09-12-02-48,False,100,50,0.72,0.26,51,49,0.037,1.0,0.072",
    "act_policy,push_button,policy_last.ckpt,kl_weight3,2025-06-09-12-12-56,False,100,50,0.78,0.22,51,50,0.038,1.0,0.073",
    "act_policy,push_button,policy_last.ckpt,kl_weight5,2025-06-09-12-23-16,False,100,50,0.64,0.34,52,50,0.044,1.0,0.074",
    "act_policy,push_button,policy_last.ckpt,kl_weight6,2025-06-09-12-33-41,False,100,50,0.7,0.26,53,49,0.04,1.0,0.073",
    "act_policy,push_button,policy_last.ckpt,kl_weight7,2025-06-09-12-44-04,False,100,50,0.64,0.34,53,47,0.04,1.0,0.074",
    "act_policy,push_button,policy_last.ckpt,kl_weight11,2025-06-09-13-38-46,False,100,50,0.72,0.26,51,49,0.039,1.0,0.073",
    "act_policy,push_button,policy_last.ckpt,kl_weight36,2025-06-09-13-49-51,False,100,50,0.56,0.42,55,51,0.045,1.0,0.072",
    "act_policy,push_button,policy_last.ckpt,kl_weight46,2025-06-09-14-31-52,False,100,50,0.62,0.38,53,51,0.042,1.0,0.072",
    "act_policy,push_button,policy_last.ckpt,kl_weight47,2025-06-09-14-13-59,False,100,50,0.7,0.28,52,49,0.039,1.0,0.074"
]

data = []
for line in lines:
    parts = line.strip().split(',')
    
    # 提取kl_weight值
    kl_weight_str = parts[3]
    match = re.search(r'kl_weight(\d+)', kl_weight_str)
    kl_weight = int(match.group(1)) if match else 0
    success_rate = float(parts[8])
    
    data.append((kl_weight, success_rate))

# 按kl_weight排序
data.sort(key=lambda x: x[0])

# 提取数据
kl_weights = [item[0] for item in data]
success_rates = [item[1] for item in data]

# 创建图表
plt.figure(figsize=(10, 6))
plt.plot(kl_weights, success_rates, marker='o', linestyle='-')
plt.xlabel('kl_weight')
plt.ylabel('success_rate')
plt.title('kl_weight_success_rate')
plt.grid(True)
plt.xticks(kl_weights)
plt.ylim(0.5, 0.85)  # 调整y轴范围以更好地显示差异
plt.savefig('kl_weight_success_rate.png', dpi=300, bbox_inches='tight')