from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from imutils import paths
from PIL import Image
from tqdm import tqdm
import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import time
from tqdm import tqdm
from utils import utils
import torch.distributed as dist
from torchvision.datasets import CIFAR10, Flowers102, OxfordIIITPet
from torchvision.transforms import ToTensor

from utils.EEGBaseDataset import EEGBaseDataset

class Flowers102Dataset(EEGBaseDataset):
    def __init__(self,root="./data/",subset="train", preprocessing_fn=None) -> None:
        super().__init__(root=root, preprocessin_fn=preprocessing_fn,subset=subset)


        self.dataset = Flowers102(root=root,download=True,split =subset, transform=self.preprocessin_fn)
        string_classes = self.dataset.classes

        for _, index in self.dataset:
            label = string_classes[index]
            self.labels.append(index)
            if index not in self.class_id_to_str:
                self.class_id_to_str[index] = label
            if label not in self.class_str_to_id:
                self.class_str_to_id[label] = index

        self.isDataTransformed = False
        self.isImageFeaturesExtracted = False


    def getOriginalImage(self, idx):
        Images, labels_ = self.dataset[idx]
        return Images
    
    def getImagePath(self, idx):
        return None
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label =  self.dataset[idx]
        EEG = []
        Image_features = []

        LabelClassName = self.class_id_to_str[label] 
        LabelClasId = self.class_str_to_id[LabelClassName]

        # if self.preprocessin_fn is not None:
        #     image = self.preprocessin_fn(image)

        if self.isDataTransformed:
            EEG = self.EEGs[idx]

        if len(self.image_features)==len(self):
            Image_features = self.image_features[idx]

        return EEG,{"ClassName": LabelClassName, "ClassId": LabelClasId},image, idx, Image_features


class OxfordIIITPetDataset(EEGBaseDataset):
    """
    subset is either trainval or test
    """
    def __init__(self,root="./data/",subset="trainval", preprocessing_fn=None) -> None:
        super().__init__(root=root, preprocessin_fn=preprocessing_fn,subset=subset)


        self.dataset = OxfordIIITPet(root=root,download=True,split=subset,target_types="category", transform=self.preprocessin_fn)
        string_classes = self.dataset.classes

        for _, index in self.dataset:
            label = string_classes[index]
            self.labels.append(index)
            if index not in self.class_id_to_str:
                self.class_id_to_str[index] = label
            if label not in self.class_str_to_id:
                self.class_str_to_id[label] = index

        self.isDataTransformed = False
        self.isImageFeaturesExtracted = False


    def getOriginalImage(self, idx):
        Images, labels_ = self.dataset[idx]
        return Images
    
    def getImagePath(self, idx):
        return None
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label =  self.dataset[idx]
        EEG = []
        Image_features = []

        LabelClassName = self.class_id_to_str[label] 
        LabelClasId = self.class_str_to_id[LabelClassName]

        # if self.preprocessin_fn is not None:
        #     image = self.preprocessin_fn(image)

        if self.isDataTransformed:
            EEG = self.EEGs[idx]

        if len(self.image_features)==len(self):
            Image_features = self.image_features[idx]

        return EEG,{"ClassName": LabelClassName, "ClassId": LabelClasId},image, idx, Image_features



class TorchDatasets(EEGBaseDataset):

    def __init__(self) -> None:
        super().__init__()

        pass