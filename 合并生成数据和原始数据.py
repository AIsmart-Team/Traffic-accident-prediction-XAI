import numpy as np

# 加载原始事故数据
original_accident_features = np.load('normalized_accident_features.npy')

# 加载生成的事故数据
generated_accident_data = np.load('generated_accident_data.npy')

# 合并数据
all_accident_data = np.vstack((original_accident_features, generated_accident_data))
print(f"合并后的事故数据形状: {all_accident_data.shape}")

# 保存合并后的数据
np.save('all_accident_data.npy', all_accident_data)
print("合并后的事故数据已保存到 'all_accident_data.npy'.")


