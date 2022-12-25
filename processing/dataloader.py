import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import random
import cv2
import torchvision


class MyDataset(Dataset):
    def __init__(self, path, image_size, type="train"):
        super(MyDataset, self).__init__()

        self.noise = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), lambda x: x + torch.randn_like(x)*0.1])
        self.path = path
        self.image_size = image_size
        self.type = "train"
        self.data_folder = os.path.join(*self.path.split(os.sep)[:-1])

        self.df = self.get_df(self.path)
        # self.info_pd["super_class_id"] = self.info_pd["super_class_id"] - 1
        # self.classes = self.df["super_class_id"].unique().tolist()

    def __getitem__(self, idx):
        item = self.df.loc[idx, ["class_id", "super_class_id", "path"]]
        class_id, super_class_id, image_path = item.to_list()
        img = self.load_img_from_path(image_path)
        img_noised = self.noise(img)
        return img, img_noised

    def get_random_image_from_class(self, super_class_id):
        df = self.df.loc[self.df["super_class_id"] == super_class_id, ["class_id", "super_class_id", "path"]]
        *_, image_path = df.iloc[random.randint(0, len(df) - 1)].to_list()
        return self.load_img_from_path(image_path)

    def load_img_from_path(self, image_path):
        img = cv2.imread(self.path + image_path)
        # (250, 250) image size
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        img = np.float32(img.transpose((2, 0, 1)) / 255)
        return img

    def get_df(self, data_dir='../../../../../Downloads/Stanford_Online_Products/'):
        train_df = pd.read_csv(f'{data_dir}Ebay_{self.type}.txt', sep=" ", header=None)
        train_df.columns = ["image_id", "class_id", "super_class_id", "path"]
        train_df = train_df.iloc[1:, :]
        return train_df

    def __len__(self):
        return len(self.df)


class DenoisingSelfSupervised(Dataset):
    def __init__(self, path, image_size, type="train", transforms=None):
        super(DenoisingSelfSupervised, self).__init__()

        self.transforms = transforms
        self.path = path
        self.image_size = image_size
        self.type = "train"
        self.data_folder = os.path.join(*self.path.split(os.sep)[:-1])

        self.df = self.get_df(self.path)
        # self.info_pd["super_class_id"] = self.info_pd["super_class_id"] - 1
        # self.classes = self.df["super_class_id"].unique().tolist()

    def __getitem__(self, idx):
        item = self.df.loc[idx, ["class_id", "super_class_id", "path"]]
        class_id, super_class_id, image_path = item.to_list()
        return self.load_img_from_path(image_path),

    def get_random_image_from_class(self, super_class_id):
        df = self.df.loc[self.df["super_class_id"] == super_class_id, ["class_id", "super_class_id", "path"]]
        *_, image_path = df.iloc[random.randint(0, len(df) - 1)].to_list()
        return self.load_img_from_path(image_path)

    def load_img_from_path(self, image_path):
        img = cv2.imread(self.path + image_path)
        # (250, 250) image size
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        img = self.transforms(image=img)['image']
        img = np.float32(img.transpose((2, 0, 1)) / 255)
        return img

    def get_df(self, data_dir='../../../../../Downloads/Stanford_Online_Products/'):
        train_df = pd.read_csv(f'{data_dir}Ebay_{self.type}.txt', sep=" ", header=None)
        train_df.columns = ["image_id", "class_id", "super_class_id", "path"]
        train_df = train_df.iloc[1:, :]
        return train_df

    def __len__(self):
        return len(self.df)

