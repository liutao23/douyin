#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Tao Liu
# @Unit    : Nanjing University of Science and Technology
# @File    : basicmodel.py
# @Time    : 2025/5/16 23:22

import torch
import torch.nn as nn
from timm import create_model
from mamba_ssm import Mamba

# 建立时序依赖的
class MambaBlock(nn.Module):
    def __init__(self, dim, num=8):
        super().__init__()
        self.layer = nn.ModuleList([
            nn.Sequential(
                Mamba(
                    d_model=dim,
                    d_state=16,
                    d_conv=4,
                    expand=2
                ),
                # nn.Dropout(0.1),  # 每层后接Dropout
                # nn.LayerNorm(dim)  # 每层后接LayerNorm
            ) for _ in range(num)
        ])
    def forward(self,x):
        for layer in self.layer:
            x = layer(x)
            pass
        return x


class MyModel(nn.Module):
    def __init__(self, num_classes=6, in_chans=3, time_steps=5, mamba_dim=512, mamba_num=8):
        super().__init__()
        # 时间步数
        self.time_steps = time_steps
        # 彩色三通道图像
        self.in_channels = in_chans

        # 第一部分：一个从图像中提取特征的backbone
        self.convnext = create_model(
            'convnext_base',
            pretrained=True,
            in_chans=self.in_channels,
            num_classes=0  # 暂时不设置分类头
        )

        # 第二部分：在特征之间引入时序依赖（mamba）帧间inter
        self.temporal_block = nn.Sequential(*[MambaBlock(dim=mamba_dim) for _ in range(mamba_num)])

        # 由于维度太大了，我们压缩一下维度
        self.conv_compress = nn.Conv2d(
            in_channels=2048,
            out_channels=mamba_dim,
            kernel_size=1,  # 1x1卷积压缩通道
            bias=False
        )

        # 池化方法（平均，最大，自适应） 2D池化
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # 自定义分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # 对最后一个维度（L=4）池化，输出形状 [2, 512, 1]
            nn.Flatten(),  # 展平为 [2, 512]
            nn.Dropout(0.4),
            nn.Linear(mamba_dim, num_classes)  # 输入维度=512
        )

        # 冻结骨干网络参数（可选）
        # for param in self.convnext.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        """
        输入格式: [B, T, C, H, W] -- [2, 5, 3, 231, 231]
        输出格式: [B, num_classes] --[2, 6]的概率分布
        """
        # 把时间和batch维度合并，送入convnext提取特征
        B, T, C, H, W = x.shape # [2, 5, 3, 231, 231]
        x = x.view(B*T,C,H,W) # [10, 3, 231, 231]

        # 送入convnext模型提取图像特征
        x = self.convnext.forward_features(x) # [10, 1024, 7, 7]

        # 根据文心一言的模型提示，显式地把相邻两帧的特征图拼接起来(可以不要）
        x = x.view(B, T, x.shape[1], 7, 7) # [2, 5, 1024, 7, 7]
        # 压缩之前
        x = torch.cat([x[:, :T - 1], x[:, 1:]], dim=2) # [2, 4, 2048, 7, 7]

        # 压缩一下维度，适合我们电脑，压缩到合适的维度512
        B, T_new, C, H, W = x.shape  # [2, 4, 2048, 7, 7]
        x = x.view(B * T_new, C, H, W)  # [8, 2048, 7, 7]

        x = self.conv_compress(x)  # [8, 512, 7, 7]
        # # 压缩之后
        x = x.view(B, T_new, x.shape[1], x.shape[2], x.shape[3])  # [2, 4, 512, 7, 7]

        # 先池化，再展平，保持mamba的维度是512 [2, 4, 512]
        # 2. 池化（可选三种方法）：例如使用自适应平均池化将空间维度降为 1x1
        x = self.pool(x)  # 输出形状: [2, 4, 512, 1, 1]
        x = x.view(B, T_new, x.shape[2])  # 输出形状: [2, 4, 512]

        # 送入mamba，建立时序依赖
        x = self.temporal_block(x) # [2, 4, 512]

        # 分类
        x = x.permute(0, 2, 1)
        x = self.classifier(x) #[2,6]

        # 转换为概率分布
        return torch.softmax(x, dim=1)

# 使用示例
if __name__ == "__main__":
    # 参数配置
    BATCH_SIZE = 2
    TIME_STEPS = 5
    CHANNELS = 3
    HEIGHT = 231
    WIDTH = 231

    # 初始化模型
    model = MyModel(
        num_classes=6,
        in_chans=CHANNELS,
        time_steps=TIME_STEPS).cuda()

    # 测试输入
    dummy_input = torch.randn(BATCH_SIZE, TIME_STEPS, CHANNELS, HEIGHT, WIDTH).cuda()

    # 前向传播
    output = model(dummy_input)

    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")  # 应输出 torch.Size([8, 6])