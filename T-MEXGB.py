import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import statsmodels.formula.api as smf
import shap
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv('111.csv')

# 剔除 `time_class_t-0` 到 `time_class_t-6` 列
columns_to_remove = ['time_class_t-1', 'time_class_t-2', 'time_class_t-3', 'time_class_t-4', 'time_class_t-5',
                     'time_class_t-6']
data = data.drop(columns=columns_to_remove)

# 确保 'time_class_t-0' 列的分类是 0, 1, 2, 3
data['time_class_t-0'] = data['time_class_t-0'].round().astype(int)

# 确保分类只有 0, 1, 2, 3
data = data[data['time_class_t-0'].isin([0, 1, 2, 3])]

# 打印数据检查
print(data['time_class_t-0'].value_counts())
print(data.dtypes)

# 将时间分类列转换为分类变量并独热编码
time_class_dummies = pd.get_dummies(data['time_class_t-0'], prefix='time_class')
data = pd.concat([data, time_class_dummies], axis=1)

# 确保独热编码后的时间分类特征不用于模型训练
print(data.columns)

# 分离特征和标签
# 移除 'time_class_t-0' 及其独热编码特征
X = data.drop(['label', 'time_class_t-0'] + [col for col in data.columns if col.startswith('time_class_')], axis=1)
y = data['label']

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 将数据合并为一个DataFrame以便拟合混合效应模型
train_data = X_train.copy()
train_data['label'] = y_train

# 使用时间分类变量作为分组变量，但不包括在固定效应中
time_class_train = data.loc[X_train.index, 'time_class_t-0']

# 检查train_data中的列
print(train_data.columns)

# 构建混合效应模型公式，不包括 time_class_t-0 作为固定效应特征
fixed_effects = ' + '.join(X.columns)
formula = f'label ~ {fixed_effects}'

# 打印公式
print(formula)

# 尝试拟合混合效应模型，将 time_class_t-0 作为随机效应
try:
    mixed_model = smf.mixedlm(formula, train_data, groups=time_class_train)
    mixed_model_fit = mixed_model.fit()
    print(mixed_model_fit.summary())
except KeyError as e:
    print(f"KeyError: {e}")
except Exception as e:
    print(f"Exception: {e}")

# 训练XGBoost模型
# 转换为DMatrix格式
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 设置参数
params = {
    'objective': 'binary:logistic',  # 二分类任务
    'max_depth': 3,  # 树的最大深度
    'eta': 0.1,  # 学习率
    'eval_metric': 'logloss'  # 评估指标
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

import numpy as np


# 定义一个函数来移除离群点
def remove_outliers(df, shap_values, column, z_thresh=1.5):
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    mask = z_scores < z_thresh
    return df[mask], shap_values[mask]


# 创建SHAP解释器
explainer = shap.TreeExplainer(bst)

# 计算SHAP值
shap_values = explainer.shap_values(X_test)

# 分别计算每个时间类别的数据集的预测结果及SHAP值
for time_class in [0, 1, 2, 3]:
    class_indices = X_test.index[data.loc[X_test.index, 'time_class_t-0'] == time_class]
    if len(class_indices) > 0:
        X_class = X_test.loc[class_indices]
        y_class = y_test.loc[class_indices]
        dclass = xgb.DMatrix(X_class)
        preds_class_proba = bst.predict(dclass)
        preds_class = [1 if x >= 0.5 else 0 for x in preds_class_proba]

        accuracy_class = accuracy_score(y_class, preds_class)
        report_class = classification_report(y_class, preds_class)
        conf_matrix_class = confusion_matrix(y_class, preds_class)
        sensitivity_class = conf_matrix_class[1, 1] / (conf_matrix_class[1, 0] + conf_matrix_class[1, 1])
        specificity_class = conf_matrix_class[0, 0] / (conf_matrix_class[0, 0] + conf_matrix_class[0, 1])
        auc_class = roc_auc_score(y_class, preds_class_proba)

        print(f'\n时间类别 {time_class} 的评估结果:')
        print(f'准确率: {accuracy_class}')
        print(f'分类报告:\n{report_class}')
        print(f'混淆矩阵:\n{conf_matrix_class}')
        print(f'灵敏度: {sensitivity_class}')
        print(f'特异性: {specificity_class}')
        print(f'AUC: {auc_class}')

        # 计算SHAP值
        shap_values_class = explainer.shap_values(X_class)

        # SHAP可视化
        #shap.summary_plot(shap_values_class, X_class, show=False)
        #plt.title(f'SHAP Summary Plot for Time Class {time_class}')
        #plt.show()

        # 去除 'flow_upstream_t-4' 特征的离群点
        X_class_no_outliers, shap_values_class_no_outliers = remove_outliers(X_class, shap_values_class,
                                                                             'mixture_downstream_t-1')

        # 对横坐标数据进行缩放
        X_class_no_outliers_scaled = X_class_no_outliers.copy()
        X_class_no_outliers_scaled['mixture_downstream_t-1'] = X_class_no_outliers_scaled['mixture_downstream_t-1']

        # 绘制去除离群点后的 SHAP 依赖图，使用缩放后的横坐标
        shap.dependence_plot(
            'mixture_downstream_t-1',
            shap_values_class_no_outliers,
            X_class_no_outliers_scaled,
            interaction_index=None,
            show=False
        )

        # 添加标题并显示图形
        plt.title(f'mixture_downstream_t-1 in Time Class {time_class}')
        plt.show()





