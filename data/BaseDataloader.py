
import torch
import torchaudio.transforms as T
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os



class ImageDataset(Dataset):
    """Basic image dataloader for image classification."""
    def __init__(self, input_path, target_path=None, image_size=224):
        self.img_list = self.get_input_list(input_path)
        self.label_list = None
        if target_path is not None:
            self.label_list = self.get_target_list(target_path)
            assert len(self.img_list) == len(self.label_list), \
            f"The sample number of input: {len(self.img_list)} and the sample number of target {len(self.label_list)} are not matching"
        self._transform = T.Resize([image_size, image_size])

    def get_input_list(self, input_path):
        df = pd.read_csv(input_path)
        return df["Id"].values

    def get_target_list(self, target_path):
        df = pd.read_csv(target_path)
        return df["Label"].values

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image_path = self.img_list[idx]
        image = read_image(image_path)
        image = self._transform(image)
        if self.label_list is not None:
            label = self.label_list[idx]
            return image, label
        return image


