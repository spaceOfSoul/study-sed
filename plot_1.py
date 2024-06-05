import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 데이터 정의
data = {
    'Models': ['Baseline', 'Model A', 'Model B', 'Model C', 'Model D', 'Model E', 'Model F', 'Model G'],
    'Parameter': [1112420, 1135082, 925162, 915867, 888586, 881610, 653467, 629588],
    'F1-score': [34.97, 34.14, 33.01, 37.06, 34.77, 36.03, 34.76, 28.2]
}

# 데이터프레임 생성
df = pd.DataFrame(data)

# 모델별 색상 지정
colors_parameter = ['blue', 'deepskyblue', 'dodgerblue', 'red', 'orange', 'darkorange', 'crimson', 'lightblue']
colors_f1_score = ['blue', 'deepskyblue', 'dodgerblue', 'red', 'orange', 'darkorange', 'crimson', 'lightblue']

# 첫 번째 그래프: Parameter
fig1, ax1 = plt.subplots(figsize=(10, 5))
bars = ax1.bar(df['Models'], df['Parameter'], color=colors_parameter)
ax1.set_xlabel('Models', fontsize=14)
ax1.set_ylabel('Parameter', fontsize=14)
ax1.set_ylim(600000, 1200000)
ax1.set_yticks(np.arange(600000, 1200001, 100000))
ax1.tick_params(axis='x', labelsize=12)
ax1.tick_params(axis='y', labelsize=12)

for bar in bars:
    height = bar.get_height()
    ax1.annotate('{}'.format(height),
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3),  # 3 points vertical offset
                 textcoords="offset points",
                 ha='center', va='bottom',
                 fontsize=10)

plt.tight_layout()
plt.show()

# Second plot: F1-score
fig2, ax2 = plt.subplots(figsize=(10, 5))
bars = ax2.bar(df['Models'], df['F1-score'], color=colors_f1_score)
ax2.set_xlabel('Models', fontsize=20)
ax2.set_ylabel('F1-score (%)', fontsize=20)
ax2.set_ylim(25, 40)
ax2.tick_params(axis='x', labelsize=16)
ax2.tick_params(axis='y', labelsize=16)

for bar in bars:
    height = bar.get_height()
    ax2.annotate('{}'.format(height),
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3),  # 3 points vertical offset
                 textcoords="offset points",
                 ha='center', va='bottom',
                 fontsize=20)

plt.tight_layout()
plt.show()