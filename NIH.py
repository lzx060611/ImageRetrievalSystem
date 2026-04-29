import os
import torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import numpy  as np
import random
import warnings
warnings.filterwarnings("ignore")

class NIH_dataset(Dataset):
    """
        初始化数据集
        """
    def __init__(self,img_root,img_list,tag_path=None,transform=None):
        super().__init__()
        self.img_root=img_root
        self.transform=transform
        with open(img_list,'r',encoding='utf-8')as f:
            self.img_names=[line.strip() for line in f]#每行是一个图像名

        if tag_path is not None:
            #读取多标签
            self.tags=[]
            with open (tag_path,'r')as f:
                for line in f:#按行读取
                    tag_list=[int(x) for x in line.strip().split(",")[1:]]#按空格分割0/1标签
                    self.tags.append(torch.tensor(tag_list,dtype=torch.float))#转化为张量读入
            #校验一次图像与标签数是否匹配
            assert len(self.img_names) == len(self.tags),"图像数量与标签数量不匹配"


    def __getitem__(self, idx):
        """读取单张图像和对应路径"""
        img_name=self.img_names[idx]
        img_tag=self.tags[idx]
        img_path=os.path.join(self.img_root,img_name) # 拼接完整图像路径


        #读取图像(转为RGB格式，避免通道不匹配)
        image=Image.open(img_path).convert("L")

        if self.transform is not None:
            image = self.transform(image)

        #返回图像路径
          
        return image,img_path,img_tag
    def __len__(self):
        return len(self.img_names)

    
if __name__ == "__main__":
    IMG_ROOT = "./images_001/images"  # 你的图片根目录
    train_LIST = "images_001/Hashsystem_dataset/train_ids.txt"  # 你的训练集文件
    train_label="images_001/Hashsystem_dataset/train_labels.txt" #训练集标签"

    NIH_data=NIH_dataset(IMG_ROOT,train_LIST,train_label)
    print(len(NIH_data))

