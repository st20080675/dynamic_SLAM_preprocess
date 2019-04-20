import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os
import numpy as np
import pickle
from PIL import Image
import random


class VOCData():
    def __init__(self, args):
        self.max_size = [385, 385]

        self.data_transforms = transforms.Compose([
            transforms.Resize(args.input_size),
            transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])

        self.dataloaders = {"data": torch.utils.data.DataLoader(
            VOCDataset(args, self.data_transforms),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4)}


class VOCDataset(Dataset):

    def __init__(self, args, transform=None):
        self.args = args
        self.list_file = args.data_file
        self.no_bg = args.no_bg
        self.data_dir = args.data_dir

        self.input_size = args.input_size
        self.num_classes = args.num_classes
        self.transform = transform
        # self.need_mask = args.need_mask_flag
        self.file_list = self._get_file_list()

    def _get_file_list(self):
        with open(os.path.join(self.data_dir, self.list_file),
                  "r") as f:
            all_files = f.readlines()
            return [item.split()[0] for item in all_files]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):

        # if os.path.exists(os.path.join(self.data_dir,
        #                                "JPEGImages",
        #                                self.file_list[idx]+".jpg")):
        img_name = os.path.join(self.data_dir,
                                self.file_list[idx])
        # else:
        #     img_name = os.path.join(self.data_dir,
        #                             "images",

        img = Image.open(img_name)

        if self.args.colorgray == "gray":
            temp_rgb = np.array(img)
            temp_gray = np.array(img.convert('L'))
            temp_rgb[:, :, 0] = temp_gray
            temp_rgb[:, :, 1] = temp_gray
            temp_rgb[:, :, 2] = temp_gray
            img = Image.fromarray(temp_rgb, mode='RGB')

        if self.args.origin_size:
            img_array = np.array(img).astype(np.float32)

        else:
            img_array = np.array(img.resize(self.input_size)).astype(np.float32)

        img_ts = self.transform(img)

        return {"input": img_ts, "img": img_array, "raw": np.array(img)}

