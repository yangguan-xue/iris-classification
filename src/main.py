import matplotlib
matplotlib.use('Agg')  # 重要：在无图形界面的服务器上必须设置
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np
import joblib
import os
import time

print("=== Iris数据集决策树分类实验 (AutoDL版本) ===")
start_time = time.time()

# 设置工作目录
work_dir = '/root/iris_experiment'
if not os.path.exists(work_dir):
    os.makedirs(work_dir)
os.chdir(work_dir)

print(f"工作目录: {os.getcwd()}")

# 1. 数据加载
print("\n1. 加载数据...")
iris = load_iris()
X, y = iris.data, iris.target
feature_names = [name.replace(' ', '_') for name in iris.feature_names]  # 替换空格
class_names = iris.target_names

print(f"数据形状: {X.shape}")
print(f"特征: {feature_names}")
print(f"类别: {list(class_names)}")

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

# 2. 模型训练
print("\n2. 训练模型...")
train_start = time.time()

dt_model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42,
    criterion='gini'
)

dt_model.fit(X_train, y_train)
train_time = time.time() - train_start

print(f"训练完成! 耗时: {train_time:.3f}秒")

# 保存模型
model_path = 'decision_tree_model.pkl'
joblib.dump(dt_model, model_path)
print(f"模型已保存: {model_path}")

# 3. 预测与评估
print("\n3. 模型评估...")
y_pred = dt_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=class_names))

# 4. 可视化
print("\n4. 生成可视化...")
plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 4.1 决策树
plot_tree(dt_model, 
          feature_names=feature_names,
          class_names=class_names,
          filled=True,
          rounded=True,
          ax=axes[0,0])
axes[0,0].set_title('Decision Tree Structure', fontsize=14, fontweight='bold')

# 4.2 特征重要性
importance = dt_model.feature_importances_
sorted_idx = np.argsort(importance)
axes[0,1].barh(range(len(importance)), importance[sorted_idx])
axes[0,1].set_yticks(range(len(importance)))
axes[0,1].set_yticklabels([feature_names[i] for i in sorted_idx])
axes[0,1].set_title('Feature Importance', fontsize=14, fontweight='bold')

# 4.3 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            ax=axes[1,0])
axes[1,0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')

# 4.4 准确率展示
metrics = ['Accuracy']
values = [accuracy]
axes[1,1].bar(metrics, values, color=['skyblue'])
axes[1,1].set_ylim(0, 1)
axes[1,1].set_title('Model Performance', fontsize=14, fontweight='bold')
for i, v in enumerate(values):
    axes[1,1].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('results.png', dpi=150, bbox_inches='tight')
print("可视化已保存: results.png")

# 5. 保存详细结果
print("\n5. 保存详细结果...")
results = {
    'accuracy': accuracy,
    'train_time_seconds': train_time,
    'total_time_seconds': time.time() - start_time,
    'feature_importance': dict(zip(feature_names, importance)),
    'model_parameters': dt_model.get_params()
}

# 保存为文本文件
with open('experiment_results.txt', 'w') as f:
    f.write("=== 实验结果 ===\n")
    f.write(f"准确率: {accuracy:.4f}\n")
    f.write(f"训练时间: {train_time:.3f}秒\n")
    f.write(f"总运行时间: {results['total_time_seconds']:.3f}秒\n")
    f.write("\n特征重要性:\n")
    for feat, imp in results['feature_importance'].items():
        f.write(f"  {feat}: {imp:.4f}\n")

# 保存预测结果表格
results_df = pd.DataFrame({
    'true_label': y_test,
    'predicted_label': y_pred,
    'is_correct': y_test == y_pred
})
results_df.to_csv('predictions.csv', index=False)

print("详细结果已保存!")

# 6. 最终总结
total_time = time.time() - start_time
print(f"\n=== 实验完成 ===")
print(f"总运行时间: {total_time:.3f}秒")
print(f"生成的文件:")
for file in ['decision_tree_model.pkl', 'results.png', 'experiment_results.txt', 'predictions.csv']:
    if os.path.exists(file):
        print(f"  - {file} ({os.path.getsize(file)} bytes)")

print("\n可以查看结果!")
