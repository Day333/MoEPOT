import torch
import torch.nn as nn
import torch.nn.functional as F
# 对每个batch分别选取专家，同时专家网络选取为卷积的神经网络
class ConvFeatureExtractor(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ConvFeatureExtractor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
        )

    def forward(self, x):
        return self.conv(x)

class GlobalTopKGating(nn.Module):
    def __init__(self, input_dim, num_experts, top_k=2, initial_temperature=2.0, is_finetune=False): 
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.temperature = initial_temperature
        self.min_temperature = 0.5
        self.temperature_decay = 0.99
        
        if is_finetune: 
            self.temperature = 0.5
            
        # ==========================================
        # 改进点 2: 全局特征提取扩展为 Avg + Max + Std
        # ==========================================
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 拼接后的通道数会变成 3 倍
        gate_input_dim = input_dim * 3
        
        # res MoE
        self.gate = nn.Sequential(
            # 第一层将 3 * input_dim 压缩回 input_dim
            nn.Conv2d(gate_input_dim, input_dim, 1), 
            nn.BatchNorm2d(input_dim),
            nn.GELU(),
            ChannelAttention(input_dim),
            nn.Conv2d(input_dim, input_dim // 2, 1),
            nn.BatchNorm2d(input_dim // 2),
            nn.GELU(),
            nn.Conv2d(input_dim // 2, num_experts, 1)
        )
    
    def update_temperature(self):
        self.temperature = max(
            self.min_temperature,
            self.temperature * self.temperature_decay
        )
    
    def forward(self, x):
        # ==========================================
        # 改进点 2: 提取并拼接三种统计特征
        # ==========================================
        avg_feat = self.avg_pool(x)  # [B, C, 1, 1]
        max_feat = self.max_pool(x)  # [B, C, 1, 1]
        
        # 计算空间维度的标准差: x 形状为 [B, C, H, W]
        # flatten(2) 得到 [B, C, H*W]，然后在空间维度求 std
        std_feat = x.flatten(2).std(dim=-1, keepdim=True).unsqueeze(-1)  # [B, C, 1, 1]
        
        # 拼接全局特征 -> [B, 3*C, 1, 1]
        global_feat = torch.cat([avg_feat, max_feat, std_feat], dim=1)
        
        gating_scores = self.gate(global_feat).squeeze(-1).squeeze(-1)  # [B, num_experts]
        
        # 选择top-k专家
        top_k_values, top_k_indices = torch.topk(gating_scores, self.top_k, dim=1)  # [B, top_k]
        top_k_values = F.softmax(top_k_values / self.temperature, dim=1)
        
        return top_k_indices, top_k_values

class Expert(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, dilation=1):
        super(Expert, self).__init__()
        
        if kernel_size == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0),
                nn.GELU()
            )
        else:
            padding = (kernel_size - 1) * dilation // 2
            self.conv = nn.Sequential(
                # DW Conv
                nn.Conv2d(input_channels, input_channels, kernel_size=kernel_size, 
                          stride=1, padding=padding, dilation=dilation, groups=input_channels),
                nn.GELU(),
                # PW Conv
                nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0),
                nn.GELU()
            )

    def forward(self, x):
        return self.conv(x)

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.GELU(),
            nn.Conv2d(channels // reduction, channels, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out

class MoEImage(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, 
                 num_experts, shared_experts_num=2, top_k=4 ,is_finetune=False):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.num_experts = num_experts
        self.shared_experts_num = shared_experts_num
        self.top_k = top_k
        self.is_finetune = is_finetune

        # 基础网络组件
        self.feature_extractor = ConvFeatureExtractor(input_channels, hidden_channels)
        # self.gating = GlobalTopKGating(hidden_channels, num_experts, top_k, is_finetune=self.is_finetune)
        
        # res moe
        self.gating = GlobalTopKGating(hidden_channels * 2, num_experts, top_k, is_finetune=is_finetune)
        
        # 共享专家网络
        self.shared_experts = nn.ModuleList([
            Expert(hidden_channels, output_channels) 
            for _ in range(shared_experts_num)
        ])
        
        # 专家网络
        # self.experts = nn.ModuleList([
        #     Expert(hidden_channels, output_channels) 
        #     for _ in range(num_experts)
        # ])

        # ==========================================
        # 改进点：为 16 个专家分配 4 种不同的感受野配置
        # ==========================================
        # 定义基础的 4 种配置 (kernel_size, dilation)
        base_configs = [
            (1, 1),  # 1x1 极小尺度 (通道交互)
            (3, 1),  # 3x3 局部细节 (小尺度)
            (5, 1),  # 5x5 平滑区域 (中尺度)
            (3, 2)   # 3x3 dilated=2 (大尺度/长程依赖)
        ]
        
        # 自动循环分配给 num_experts 个专家
        # 比如 16 个专家，就会是 4个(1,1), 4个(3,1), 4个(5,1), 4个(3,2)
        expert_configs = [base_configs[i % len(base_configs)] for i in range(num_experts)]
        
        # 专家网络实例化
        self.experts = nn.ModuleList([
            Expert(hidden_channels, output_channels, kernel_size=k, dilation=d) 
            for k, d in expert_configs
        ])

    def freeze_feature_and_gating(self, freeze=True):
        """
        冻结或解冻特征提取器和门控网络的参数
        Args:
            freeze (bool): True表示冻结参数，False表示解冻参数
        """
        for param in self.feature_extractor.parameters():
            param.requires_grad = not freeze
        for param in self.gating.parameters():
            param.requires_grad = not freeze
            
    # def forward(self, x):
    # res moe
    def forward(self, x, temporal_residual=None):
        features = self.feature_extractor(x)
        
        if temporal_residual is None:
            # 如果没传残差，可以用一个全零占位，或者对 features 做个简单偏移
            gate_input = torch.cat([features, torch.zeros_like(features)], dim=1)
        else:
            # 提取残差特征并与当前特征拼接
            res_feat = self.feature_extractor(temporal_residual)
            gate_input = torch.cat([features, res_feat], dim=1)
        
        top_k_indices, top_k_values = self.gating(gate_input)
        
        # 1. 共享专家的输出
        shared_output = torch.zeros_like(x)
        for expert in self.shared_experts:
            shared_output += expert(features) / self.shared_experts_num
        
        # 2. 专家网络的输出
        output = torch.zeros_like(x)
        
        # 获取专家分配
        # top_k_indices, top_k_values = self.gating(features)
        
        # 对每个专家处理数据
        for expert_idx in range(self.num_experts):
            mask = (top_k_indices == expert_idx)
            weights = top_k_values * mask
            expert_output = self.experts[expert_idx](features)
            output += expert_output * weights.sum(dim=1).view(-1, 1, 1, 1)
        
        # 更新专家使用统计，finetune阶段不在计算门控损失
        if self.training and not self.is_finetune:
            # 计算平衡损失
            loss_gate = self.compute_balance_loss(top_k_values, top_k_indices)
            # 更新温度
            self.gating.update_temperature()
        else:
            loss_gate = 0

        # 最终输出是共享专家和专家网络输出的组合
        return shared_output + output, loss_gate

    def compute_balance_loss(self, gates, indices):
        """计算负载均衡损失"""
        importance = torch.zeros(self.num_experts, device=gates.device)
        for i in range(self.num_experts):
            mask = (indices == i)
            importance[i] = (gates * mask).sum()
        
        # 计算理想负载
        ideal_load = gates.sum() / self.num_experts
        # 计算负载均衡损失
        balance_loss = torch.pow(importance - ideal_load, 2).mean() # 理论上的CV还应该除以理想负载
            
        return balance_loss

# Example usage
if __name__ == "__main__":
    # 初始化模型
    model = MoEImage(
        input_channels=3,
        hidden_channels=16,
        output_channels=3,
        num_experts=4,
        top_k=2
    )

    # 模拟训练过程
    model.train()
    x = torch.randn(8, 3, 32, 32)
    
    for epoch in range(10):
        output, loss_gate = model(x)
        print(f"Epoch {epoch}, Gate Loss: {loss_gate.item():.4f}")
        print(f"Current temperature: {model.gating.temperature:.4f}")
