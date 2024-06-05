import matplotlib.pyplot as plt
import numpy as np

baseline_data = {
    "Class": ["Alarm bell", "Blender", "Cat", "Dishes", "Dog", "Electric shaver", "Frying", "Running water", "Speech", "Vacuum cleaner"],
    "F1-score": [25.1, 35.0, 40.8, 28.3, 27.7, 40.7, 24.9, 34.1, 50.1, 43.0]
}

model_c_data = {
    "Class": ["Alarm bell", "Blender", "Cat", "Dishes", "Dog", "Electric shaver", "Frying", "Running water", "Speech", "Vacuum cleaner"],
    "F1-score": [16.8, 31.5, 48.0, 29.7, 24.8, 49.0, 34.2, 37.7, 50.8, 48.0]
}


classes = baseline_data["Class"]

baseline_f1 = baseline_data["F1-score"]
model_c_f1 = model_c_data["F1-score"]

bar_width = 0.35
index = np.arange(len(classes))

fig, ax = plt.subplots(figsize=(12, 8))
bars1 = ax.bar(index, baseline_f1, bar_width, label='Baseline', color='blue')
bars2 = ax.bar(index + bar_width, model_c_f1, bar_width, label='Model C', color='red')

ax.set_xlabel('Class', fontsize=20)
ax.set_ylabel('F1-score (%)', fontsize=20)
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=20)
ax.legend(fontsize=20)
ax.tick_params(axis='y', labelsize=18)
for bar in bars1:
    height = bar.get_height()
    ax.annotate('{}'.format(height),
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=13)

for bar in bars2:
    height = bar.get_height()
    ax.annotate('{}'.format(height),
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=13)

plt.tight_layout()
plt.show()