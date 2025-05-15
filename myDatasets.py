#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Tao Liu
# @Unit    : Nanjing University of Science and Technology
# @File    : myDatasets.py
# @Time    : 2025/5/15 22:44

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torchvision import transforms
import numpy as np

class MyDataset(Dataset):
    # 初始化函数 它定义了我们数据集需要的一些参数
    def __init__(self, root_dir=r'C:\demo\WaterPaper\Datasets\MMEW',):

        # 数据预处理（图像增强标准化）
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # 将PIL图像或numpy数组转换为张量
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
        ])
        # 拼接第一个文件夹的路径，因为他有5张图片比较简单
        # C:\demo\WaterPaper\Datasets\MMEW\Macro_Expression
        root_dir = os.path.join(root_dir, 'Macro_Expression')

        # 获取Macro_Expression的所有子文件夹，30个人
        # ['S01','S02',...,'S30']
        person_dir = sorted(os.listdir(root_dir))

        self.img_list = []
        # 遍历每个人，并且返回他不同的表情的图片
        for file in person_dir: #anger
            # C:\demo\WaterPaper\Datasets\MMEW\Macro_Expression\S01\anger
            ['anger', 'disgust', 'fear', ..., 'surprise']
            emotion_dir = os.listdir(os.path.join(root_dir, file))

            for emtion in emotion_dir:
                # C:\demo\WaterPaper\Datasets\MMEW\Macro_Expression\S01\anger\S01-07-001.jpg
                img_dir = os.path.join(root_dir,file,emtion)
                sub_img_list = [os.path.join(root_dir,file,emtion,img) for img in os.listdir(img_dir)]
            self.img_list.append((sub_img_list, emtion))
            pass
        pass

    # 获取数据集的长度
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # im_list 是接受的五张图片路径的列表
        # emo 是对应表情描述的字符串
        im_list, emo = self.img_list[idx]
        im_tensor_list = []
        for im_path in im_list:
            im = Image.open(im_path).convert('RGB')
            # [B, C, H ,W]
            im_tensor = self.transform(im)
            im_tensor_list.append(im_tensor)
            pass

        #增加时间维度，对这五张图片进行张量的拼接
        emtion_img = torch.stack(im_tensor_list, dim=1).permute(1,0,2,3)
        return emtion_img, emo

if __name__=='__main__':

    Mydata = MyDataset(root_dir=r'C:\demo\WaterPaper\Datasets\MMEW')
    print(Mydata.img_list[0])
    myloader = DataLoader(Mydata, 1)

    for data in myloader:
        print(data)
