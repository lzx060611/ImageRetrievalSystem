import os
import torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import numpy  as np
import random

class NIH_dataset(Dataset):
    """
        初始化数据集
        :param img_root: 图像所在的根目录(Flickr文件夹路径)
        :param img_list_path: 图像名列表的txt文件路径(如TrainImagelist.txt)
        :param tag_path: 多标签txt文件路径(如Train_Tags81.txt)
        :param transform: 图像预处理变换(如Resize、ToTensor等)
        :param indices: 提取想要的图像索引
        """
    def __init__(self,img_root,img_list,tag_path=None,transform=None,indices=None):
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
                    tag_list=[int(x) for x in line.strip().split()]#按空格分割0/1标签
                    self.tags.append(torch.tensor(tag_list,dtype=torch.float))#转化为张量读入
            #校验一次图像与标签数是否匹配
            assert len(self.img_names) == len(self.tags),"图像数量与标签数量不匹配"

        if indices is not None:
            indices=list(indices)#避免出现结构错误
            self.img_names=[self.img_names[i] for i in indices]
            self.tags=[self.tags[i] for i in indices ]

    def __getitem__(self, idx):
        """读取单张图像和对应路径"""
        img_name=self.img_names[idx]
        img_path=os.path.join(self.img_root,img_name) # 拼接完整图像路径


        #读取图像(转为RGB格式，避免通道不匹配)
        image=Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        #返回图像路径
          
        return image,img_path
    def __len__(self):
        return len(self.img_names)
    
if __name__ == "__main__":
    img_root="images_001\\images"
    img_list="images_001\\train_val_list.txt"

    NIH_data=NIH_dataset(img_root,img_list)
    print(len(NIH_data))

