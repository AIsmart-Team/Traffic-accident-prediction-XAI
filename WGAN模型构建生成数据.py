import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# 读取标准化后的数据
accident_features_normalized = np.load('normalized_accident_features.npy')

# 定义生成器
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# 定义判别器
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.model(x)

# 梯度惩罚函数
def compute_gradient_penalty(discriminator, real_data, fake_data):
    alpha = torch.rand(real_data.size(0), 1)
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates.requires_grad_(True)
    disc_interpolates = discriminator(interpolates)
    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones_like(disc_interpolates),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# 超参数
z_dim = 100  # 噪声维度
feature_dim = accident_features_normalized.shape[1]  # 特征维度
lr = 0.0002  # 学习率
b1 = 0.5  # Adam优化器的beta1参数
b2 = 0.999  # Adam优化器的beta2参数
n_epochs = 2000  # 迭代次数
lambda_gp = 10  # 梯度惩罚项的系数

# 实例化生成器和判别器
generator = Generator(z_dim, feature_dim)
discriminator = Discriminator(feature_dim)

# 定义优化器
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

# 准备数据
accident_features_tensor = torch.tensor(accident_features_normalized, dtype=torch.float32)
data_loader = DataLoader(TensorDataset(accident_features_tensor), batch_size=64, shuffle=True)

# 定义训练函数
def train_WGAN_GP(generator, discriminator, optimizer_G, optimizer_D, data_loader, z_dim, epochs=1000, lambda_gp=10):
    for epoch in range(epochs):
        for i, (real_data,) in enumerate(data_loader):
            real_data = real_data
            batch_size = real_data.size(0)

            # 训练判别器
            optimizer_D.zero_grad()
            z = torch.randn(batch_size, z_dim)
            fake_data = generator(z)
            real_validity = discriminator(real_data)
            fake_validity = discriminator(fake_data.detach())
            d_loss = -(torch.mean(real_validity) - torch.mean(fake_validity))
            gradient_penalty = compute_gradient_penalty(discriminator, real_data, fake_data)
            d_loss += lambda_gp * gradient_penalty
            d_loss.backward(retain_graph=True)  # 保留计算图
            optimizer_D.step()

            # 训练生成器
            if i % 5 == 0:
                optimizer_G.zero_grad()
                gen_validity = discriminator(fake_data)
                g_loss = -torch.mean(gen_validity)
                g_loss.backward()
                optimizer_G.step()

        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/{epochs}] | D Loss: {d_loss.item()} | G Loss: {g_loss.item()}")

# 训练WGAN-GP
train_WGAN_GP(generator, discriminator, optimizer_G, optimizer_D, data_loader, z_dim, n_epochs, lambda_gp)

# 定义标准化器
scaler = StandardScaler()
scaler.fit(accident_features_normalized)

# 生成更多事故数据
def generate_accident_data(generator, z_dim, num_samples):
    generator.eval()
    z = torch.randn(num_samples, z_dim)
    generated_data = generator(z).detach().numpy()
    generated_data = scaler.inverse_transform(generated_data)  # 反标准化
    return generated_data

# 生成1000个事故数据样本
generated_accident_data = generate_accident_data(generator, z_dim, 261)
print(generated_accident_data[:5])

# 保存生成的数据
np.save('generated_accident_data.npy', generated_accident_data)

print("Generated data saved successfully.")









































