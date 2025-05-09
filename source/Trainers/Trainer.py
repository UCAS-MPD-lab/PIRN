#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
# from tqdm import tqdm 
import torch.optim as optim
from torch.autograd import grad
from tqdm.notebook import tqdm 


class Trainer:
    def __init__(self, model, optimizer, criterions, device="cuda"):
        """
        初始化 Trainer。
        
        参数:
        - model: 模型实例。
        - optimizer: 优化器实例。
        - criterions: 损失函数字典，键为损失名称，值为损失函数实例。
        - device: 设备（如 "cuda" 或 "cpu"）。

        示例:
        >>> # 定义模型
        >>> class MyModel(nn.Module):
        ...     def __init__(self):
        ...         super(MyModel, self).__init__()
        ...         self.fc1 = nn.Linear(1, 128)
        ...         self.fc2 = nn.Linear(128, 128)
        ...         self.fc3 = nn.Linear(128, 1)
        ...     def forward(self, x):
        ...         out1 = self.fc1(x)  # 第一个输出
        ...         out2 = self.fc2(out1)  # 第二个输出
        ...         return {"u": out1, "Fv": out2}  # 返回一个字典
    
        >>> # 定义损失函数
        >>> criterions = {
        ...     "u": nn.MSELoss(),  # 第一个损失函数
        ...     "Fv": nn.MSELoss(),  # 第二个损失函数
        ... }
    
        >>> # 初始化 Trainer
        >>> model = MyModel()
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        >>> trainer = Trainer(model, optimizer, criterions, device="cuda")
    
        >>> # 准备数据加载器
        >>> # 假设 targets_u 和 targets_Fv 是目标值
        >>> targets = {"u": targets_u, "Fv": targets_Fv}
        >>> train_dataset = TensorDataset(inputs, targets)
        >>> train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    
        >>> # 开始训练
        >>> trainer.fit(train_loader, val_loader, num_epochs=50, early_stop_patience=5, save_path="best_model.pth")
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterions = criterions  # 损失函数字典
        self.device = device
        self.train_losses = []  # 记录训练损失
        self.val_losses = []  # 记录验证损失

    def _train_step(self, inputs, targets):
        """
        训练步骤。
        
        参数:
        - inputs: 输入数据。
        - targets: 目标值字典，键为损失名称，值为目标值。
        
        返回:
        - total_loss: 总损失值。
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # 前向传播
        preds = self.model(inputs)  # 假设模型返回一个字典，键为损失名称，值为预测值

        
        # 计算各项损失
        total_loss = 0.0
        for name, criterion in self.criterions.items():
            loss = criterion(preds[name], targets[name])
            total_loss += loss
        
        # 反向传播
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()

    def _val_step(self, inputs, targets):
        """
        验证步骤。
        
        参数:
        - inputs: 输入数据。
        - targets: 目标值字典，键为损失名称，值为目标值。
        
        返回:
        - total_loss: 总损失值。
        """
        self.model.eval()
        with torch.no_grad():
            preds = self.model(inputs)  # 假设模型返回一个字典，键为损失名称，值为预测值
            
            # 计算各项损失
            total_loss = 0.0
            for name, criterion in self.criterions.items():
                loss = criterion(preds[name], targets[name])
                total_loss += loss
            
            return total_loss.item()

    def fit(self, train_loader, val_loader, num_epochs=50, early_stop_patience=5, save_path="best_model.pth"):
        """
        训练模型。
        
        参数:
        - train_loader: 训练数据加载器。
        - val_loader: 验证数据加载器。
        - num_epochs: 训练轮数。
        - early_stop_patience: 早停耐心值。
        - save_path: 模型保存路径。
        
        """
        self.train_losses = []  # reset
        self.val_losses = []
        best_val_loss = float('inf')
        early_stop_counter = 0
        
        for epoch in tqdm(range(num_epochs), desc="Training", total=num_epochs):
            # 训练阶段
            train_loss = 0.0
            for batch in train_loader:
                inputs = batch[0].to(self.device)  # 输入数据
                targets = {name: target.to(self.device) for name, target in zip(self.criterions.keys(), batch[1:])}  # 目标值
                train_loss += self._train_step(inputs, targets)
            
            # 验证阶段
            val_loss = 0.0
            for batch in val_loader:
                inputs = batch[0].to(self.device)  # 输入数据
                targets = {name: target.to(self.device) for name, target in zip(self.criterions.keys(), batch[1:])}  # 目标值
                val_loss += self._val_step(inputs, targets)
            
            # 计算平均损失
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # 早停和保存模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= early_stop_patience:
                    tqdm.write(f"Early stopping at epoch {epoch+1}")
                    break
            
            tqdm.write(f"Epoch {epoch+1}/{num_epochs} | "
                       f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            # print(f"Epoch {epoch+1}/{num_epochs} | "
            #       f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    def plot_loss(self, save=False):
        """
        绘制训练和验证损失曲线。
        """
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        if save==True:
            plt.savefig('loss.png')
        plt.show()



def split_data(inputs, targets, train_ratio=0.8, batch_size=32, shuffle=True):
    """
    将数据集分为训练集和验证集，并返回对应的 DataLoader。

    参数:
    - inputs: 输入数据，形状为 (num_samples, input_features)。
    - targets: 目标值字典，键为目标值名称，值为目标值张量。
    - train_ratio: 训练集的比例（默认为 0.8）。
    - batch_size: DataLoader 的批量大小（默认为 32）。
    - shuffle: 是否打乱数据（默认为 True）。

    返回:
    - train_loader: 训练集的 DataLoader。
    - val_loader: 验证集的 DataLoader。

    示例:
    >>> # 生成随机数据
    >>> inputs = torch.from_numpy(np.random.rand(100, 1).astype(np.float32))  # 100 个样本，每个样本 1 个特征
    >>> targets = {
    ...     "u": torch.from_numpy(np.random.rand(100, 1).astype(np.float32)),  # 目标值 u
    ...     "Fv": torch.from_numpy(np.random.rand(100, 1).astype(np.float32)),  # 目标值 Fv
    ... }

    >>> # 划分数据集
    >>> train_loader, val_loader = split_data(inputs, targets, train_ratio=0.8, batch_size=10)

    >>> # 打印划分结果
    >>> print(f"训练集大小: {len(train_loader.dataset)}")
    >>> print(f"验证集大小: {len(val_loader.dataset)}")
    训练集大小: 80
    验证集大小: 20
    """
    # 检查输入数据和目标值的样本数是否匹配
    num_samples = len(inputs)
    for name, target in targets.items():
        assert len(target) == num_samples, f"目标值 '{name}' 的样本数与输入数据不匹配"

    # 将输入数据和目标值打包为 TensorDataset
    dataset = TensorDataset(inputs, *targets.values())

    # 计算训练集和验证集的大小
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    # 使用 random_split 划分数据集
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loader, val_loader


# 定义 PINN 训练器
class Burges_1D_Trainer:
    def __init__(self, model, u, alpha, optimizer, lambda_pde=1.0, device='cuda'):
        """
        参数:
            model: PINN 模型
            u: 物理参数（速度）
            alpha: 物理参数（传热系数）
            optimizer: 求解器
            lambda_pde: 物理方程loss系数
            device: 训练设备，例如 'cpu' 或 'cuda'
        """
        self.device = device
        self.model = model.to(device)
        self.u = u
        self.alpha = alpha
        self.optimizer = optimizer
        self.loss_func = nn.MSELoss()
        self.lambda_pde = lambda_pde
        self.pde_loss = []
        self.bc_loss = []
        self.loss = []

    def pde_residual(self, x):
        """
        计算 PDE 残差:
            u * phi_x - alpha * phi_xx
        参数:
            x: 输入的 collocation 点 (需要计算梯度)
        """
        # 将 x 设置到对应设备并开启梯度
        x = x.to(self.device)
        x.requires_grad_(True)
        phi = self.model(x)
        # 计算一阶导数 phi_x
        phi_x = torch.autograd.grad(phi, x,
                                  grad_outputs=torch.ones_like(phi),
                                  retain_graph=True,
                                  create_graph=True)[0]
        # 计算二阶导数 phi_xx
        phi_xx = torch.autograd.grad(phi_x, x,
                                   grad_outputs=torch.ones_like(phi_x),
                                   retain_graph=True,
                                   create_graph=True)[0]
        # 一维稳态 Burgers 方程: u * phi_x - alpha * phi_xx = 0
        residual = self.u * phi_x - self.alpha * phi_xx
        return residual

    def train(self, collocation_points, boundary_points, boundary_values, epochs=10000, save_path='best.pth'):
        """
        训练函数
        参数:
            collocation_points: 用于计算 PDE 残差的点（内部点）
            boundary_points: 边界点
            boundary_values: 对应的边界条件值
            epochs: 训练的迭代次数
        """
        collocation_points = collocation_points.to(self.device)
        boundary_points = boundary_points.to(self.device)
        boundary_values = boundary_values.to(self.device)

        best_loss = float('inf')
        for epoch in tqdm(range(epochs), desc="Training Progress"):
            self.optimizer.zero_grad()
            # 计算 collocation 点处的 PDE 残差
            res = self.pde_residual(collocation_points)
            pde_loss = self.loss_func(res, torch.zeros_like(res))
            self.pde_loss.append(pde_loss)
            # 计算边界条件损失
            u_b = self.model(boundary_points)
            bc_loss = self.loss_func(u_b, boundary_values)
            self.bc_loss.append(bc_loss)
            # 总损失
            loss = self.lambda_pde * pde_loss + bc_loss
            self.loss.append(loss)
            loss.backward()
            self.optimizer.step()
            if loss < best_loss:
                best_loss = loss
                torch.save(self.model.state_dict(), save_path)
            if epoch % 100 == 0:
                tqdm.write(f"Epoch {epoch+1}/{epochs}, Total Loss: {loss.item():.4e} | PDE Loss: {pde_loss.item():.4e} | BC Loss: {bc_loss.item():.4e}")
    
    def plot_loss(self):
            """
            绘制训练和验证损失曲线。
            """
            import matplotlib.pyplot as plt

            loss = [tensor.detach().cpu().numpy() for tensor in self.loss]
            plt.figure(figsize=(10, 5))
            plt.plot(loss, label='Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()


class NavierStokes_2D_Trainer:
    def __init__(self, model, nu, optimizer, lambda_pde=1.0, device='cuda'):
        """
        参数:
            model: PINN 模型
            nu: 物理参数（动量方程中的粘度）
            rho: 物理参数（流体密度）
            optimizer: 求解器
            lambda_pde: 物理方程loss系数
            device: 训练设备，例如 'cpu' 或 'cuda'
        """
        self.device = device
        self.model = model.to(device)
        self.nu = nu  # 动量方程中的粘度
        self.optimizer = optimizer
        self.loss_func = nn.MSELoss()
        self.lambda_pde = lambda_pde
        self.pde_loss = []
        self.bc_loss = []
        self.loss = []

    def pde_residual(self, x, y):
        """
        计算 2D Navier-Stokes 方程的残差:
            连续性方程: u_x + v_y = 0
            动量方程 (x 方向): u_x + u * u_x + v * u_y = - p_x + nu * (u_xx + u_yy)
            动量方程 (y 方向): v_x + u * v_x + v * v_y = - p_y + nu * (v_xx + v_yy)
        参数:
            x: 输入的空间坐标 x
            y: 输入的空间坐标 y
        """
        # 将 x 和 y 设置到对应设备并开启梯度
        x = x.to(self.device)
        y = y.to(self.device)
        x.requires_grad_(True)
        y.requires_grad_(True)
        
        # 得到模型的预测值
        u, v, p = self.model(x, y)
        
        # 计算各类偏导数
        u_x = grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        u_y = grad(u, y, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
        
        v_x = grad(v, x, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
        v_y = grad(v, y, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
        
        p_x = grad(p, x, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
        p_y = grad(p, y, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
        
        # 计算二阶导数 u_xx, u_yy
        u_xx = grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
        u_yy = grad(u_y, y, grad_outputs=torch.ones_like(u_y), retain_graph=True, create_graph=True)[0]
        
        v_xx = grad(v_x, x, grad_outputs=torch.ones_like(v_x), retain_graph=True, create_graph=True)[0]
        v_yy = grad(v_y, y, grad_outputs=torch.ones_like(v_y), retain_graph=True, create_graph=True)[0]
        
        # 计算各项残差
        continuity_residual = u_x + v_y  # 连续性方程残差
        momentum_x_residual = u_x + u * u_x + v * u_y + p_x - self.nu * (u_xx + u_yy)  # 动量方程 x 方向残差
        momentum_y_residual = v_x + u * v_x + v * v_y + p_y - self.nu * (v_xx + v_yy)  # 动量方程 y 方向残差
        
        return continuity_residual, momentum_x_residual, momentum_y_residual

    def train(self, collocation_points, boundary_points, boundary_values_u, boundary_values_v, boundary_values_p=None, epochs=10000, save_path='best_model.pth'):
        """
        训练函数
        参数:
            collocation_points: 用于计算 PDE 残差的点（内部点）
            boundary_points: 边界点
            boundary_values_u: 边界条件 u
            boundary_values_v: 边界条件 v
            boundary_values_p: 边界条件 p（可以为 None，表示没有 p 的边界条件）
            epochs: 训练的迭代次数
        """
        collocation_points = collocation_points.to(self.device)
        boundary_points = boundary_points.to(self.device)
        boundary_values_u = boundary_values_u.to(self.device)
        boundary_values_v = boundary_values_v.to(self.device)
    
        # 只对非 None 的 p 值进行操作
        if boundary_values_p is not None:
            boundary_values_p = boundary_values_p.to(self.device)
    
        best_loss = float('inf')
        for epoch in tqdm(range(epochs), desc="Training Progress"):
            self.optimizer.zero_grad()
            
            # 合并 collocation_points 为 (x, y) 输入
            x_collocation, y_collocation = collocation_points[:, 0], collocation_points[:, 1]
            
            # 计算 PDE 残差
            continuity_residual, momentum_x_residual, momentum_y_residual = self.pde_residual(x_collocation, y_collocation)
            
            # 计算 PDE 损失
            pde_loss = self.loss_func(continuity_residual, torch.zeros_like(continuity_residual)) + \
                       self.loss_func(momentum_x_residual, torch.zeros_like(momentum_x_residual)) + \
                       self.loss_func(momentum_y_residual, torch.zeros_like(momentum_y_residual))
            self.pde_loss.append(pde_loss)
            
            # 计算边界条件损失
            u_b, v_b, p_b = self.model(boundary_points[:, 0], boundary_points[:, 1])  # 将 boundary_points 传递给模型
            bc_loss_u = self.loss_func(u_b, boundary_values_u)
            bc_loss_v = self.loss_func(v_b, boundary_values_v)
            
            # 只有在 boundary_values_p 非 None 时才计算压力损失
            if boundary_values_p is not None:
                bc_loss_p = self.loss_func(p_b, boundary_values_p)
                bc_loss = bc_loss_u + bc_loss_v + bc_loss_p
            else:
                bc_loss = bc_loss_u + bc_loss_v
            
            self.bc_loss.append(bc_loss)
            
            # 总损失
            loss = self.lambda_pde * pde_loss + bc_loss
            self.loss.append(loss)
            loss.backward()
            self.optimizer.step()
            
            # 保存模型
            if loss < best_loss:
                best_loss = loss
                torch.save(self.model.state_dict(), save_path)
            
            if epoch % 100 == 0:
                tqdm.write(f"Epoch {epoch+1}/{epochs}, Total Loss: {loss.item():.4e} | PDE Loss: {pde_loss.item():.4e} | BC Loss: {bc_loss.item():.4e}")
    
    def plot_loss(self):
        """
        绘制训练和验证损失曲线。
        """
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(self.loss, label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()



