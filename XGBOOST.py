import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

# 读取数据
data = pd.read_csv('111.csv')

# 打印数据检查
print(data.dtypes)

# 剔除 `time_class_t-0` 到 `time_class_t-6` 列
columns_to_remove = ['time_class_t-0', 'time_class_t-1', 'time_class_t-2', 'time_class_t-3', 'time_class_t-4', 'time_class_t-5', 'time_class_t-6']
data = data.drop(columns=columns_to_remove)

# 检查数据列
print(data.columns)

# 分离特征和标签
X = data.drop(['label'], axis=1)
y = data['label']

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练XGBoost模型
# 转换为DMatrix格式
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 设置参数
params = {
    'objective': 'binary:logistic',  # 二分类任务
    'max_depth': 3,                  # 树的最大深度
    'eta': 0.1,                      # 学习率
    'eval_metric': 'logloss'         # 评估指标
}

# 训练模型
bst = xgb.train(params, dtrain, num_boost_round=100)

# 预测
preds_proba = bst.predict(dtest)
preds = [1 if x >= 0.5 else 0 for x in preds_proba]  # 将概率转换为二分类结果

# 评估模型
accuracy = accuracy_score(y_test, preds)
report = classification_report(y_test, preds)
conf_matrix = confusion_matrix(y_test, preds)
sensitivity = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])
specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
auc = roc_auc_score(y_test, preds_proba)
fpr, tpr, _ = roc_curve(y_test, preds_proba)

print(f'准确率: {accuracy}')
print(f'分类报告:\n{report}')
print(f'混淆矩阵:\n{conf_matrix}')
print(f'灵敏度: {sensitivity}')
print(f'特异性: {specificity}')
print(f'AUC: {auc}')

# 输出所有评估指标
print(f'Test Accuracy: {accuracy}')
print(f'Sensitivity: {sensitivity}')
print(f'Specificity: {specificity}')
print(f'AUC: {auc}')

