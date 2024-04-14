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
# custom dataset class
class DomainnetDataset(Dataset):
    def __init__(self, images_path, preprocessin_fn = None, filter_label=None, test_split=0.2, random_seed=43, subset="train"):

        image_paths = list(paths.list_images(images_path))
        self.str_labels = []
        self.images = []
        self.EEGs = []

        for img_path in tqdm(image_paths):
            label = img_path.split(os.path.sep)[-2]

            if filter_label is not None:
                if label == filter_label: 
                    self.images.append(img_path.replace("\\", "/"))
                    self.str_labels.append(label)
            else:
                self.images.append(img_path.replace("\\", "/"))
                self.str_labels.append(label)

        lb = LabelEncoder()
        self.labels = lb.fit_transform(self.str_labels)

        self.class_str_to_id = {}
        for intlab, strlab in zip(self.labels,self.str_labels):
            if strlab not in self.class_str_to_id:
                self.class_str_to_id[strlab] = intlab

        self.int_to_str_labels = {y: x for x, y in self.class_str_to_id.items()}

        # print("self.str_labels", len(self.str_labels))
        # print("self.images", len(self.images))
        # print("self.labels", len(self.labels))
        # print("self.str_labels", len(self.str_labels))
        # print("self.class_str_to_id", len(self.class_str_to_id))
        # print("self.int_to_str_labels", len(self.int_to_str_labels))

        # Create a StratifiedShuffleSplit instance
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_split, random_state=random_seed)

        # Get the indices for the splits
        for train_index, test_index in sss.split(self.images, self.labels):
            self.train_images = [self.images[i] for i in train_index]
            self.train_labels = [self.labels[i] for i in train_index]
            self.test_images = [self.images[i] for i in test_index]
            self.test_labels = [self.labels[i] for i in test_index]

        
        print(len(set(self.test_labels)), len(set(self.train_labels)))
        assert len(set(self.test_labels))==len(set(self.train_labels))

        if subset == "train":
            self.images = self.train_images
            self.labels = self.train_labels
        elif subset == "test":
            self.images = self.test_images
            self.labels = self.test_labels

        # self.images_train, self.images_test, self.str_labels_train, self.str_labels_test= train_test_split(
        #     self.images, 
        #     self.str_labels, 
        #     test_size=test_split, 
        #     random_state=random_seed)
        
        # print(f"Total Images: {len(self.images)}")
        # if subset=="train":
        #     self.str_labels = self.images_train
        #     self.images = self.images_train
        # elif subset=="test":
        #     self.str_labels = self.images_test
        #     self.images = self.images_test
        # else:
        #     raise Exception("subset should either be train or test")
        
        print(f"Subset Images: {len(self.images)}")

        self.preprocessin_fn = preprocessin_fn
        self.isDataTransformed = False

    def getOriginalImage(self, idx):
        ImagePath = self.images[idx][:]
        imageOriginal = Image.open(ImagePath).convert('RGB')
        return imageOriginal
    
    def getImagePath(self, idx):
        ImagePath = self.images[idx][:]
        return ImagePath

    @torch.no_grad()
    def extract_features(self,model, data_loader, use_cuda=True, multiscale=False):
        metric_logger = utils.MetricLogger(delimiter="  ")
        features = None
        # for samples, index in metric_logger.log_every(data_loader, 10):
        for EEG,labels,image, index, Image_features in metric_logger.log_every(data_loader, 10):
            samples = image
            samples = samples.cuda(non_blocking=True)
            index = index.cuda(non_blocking=True)
            if multiscale:
                feats = utils.multi_scale(samples, model)
            else:
                feats = model(samples).clone()

            # init storage feature matrix
            if dist.get_rank() == 0 and features is None:
                features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
                if use_cuda:
                    features = features.cuda(non_blocking=True)
                print(f"Storing features into tensor of shape {features.shape}")

            # get indexes from all processes
            y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
            y_l = list(y_all.unbind(0))
            y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
            y_all_reduce.wait()
            index_all = torch.cat(y_l)

            # share features between processes
            feats_all = torch.empty(
                dist.get_world_size(),
                feats.size(0),
                feats.size(1),
                dtype=feats.dtype,
                device=feats.device,
            )
            output_l = list(feats_all.unbind(0))
            output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
            output_all_reduce.wait()

            # update storage feature matrix
            if dist.get_rank() == 0:
                if use_cuda:
                    features.index_copy_(0, index_all, torch.cat(output_l))
                else:
                    features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
        
        for f in features:
            self.EEGs.append(f.cpu().numpy())
        #return features 
    
    def transformEEGDataDino(self, model, device, pass_eeg=False,preprocessor=None, min_time=0,max_time=460, do_z2_score_norm=False):
        print(f"Transforming Image data to dino features EEG, pass_eeg is {pass_eeg}")
        # assert pass_eeg==False
        model = model.to(device)

        for i, image_path in tqdm(enumerate(self.images), total=len(self.images)):
            t0 = time.perf_counter()
            model_inputs = None
            if pass_eeg:
                eeg = self.subsetData[i]["eeg"].float()
                eeg = eeg.cpu().numpy()
                eeg = self.resizeEEGToImageSize(eeg)
                if do_z2_score_norm:
                    fmean = np.mean(eeg)
                    fstd = np.std(eeg)
                    eeg = (eeg - fmean)/fstd
                model_inputs = torch.from_numpy(eeg)
            else:
                model_inputs = self.getOriginalImage(i)
                if self.preprocessin_fn is not None:
                    model_inputs = self.preprocessin_fn(model_inputs)
                else:
                    if preprocessor is not None:
                        model_inputs = preprocessor(model_inputs)

            with torch.no_grad():
                feats = model(model_inputs.unsqueeze(0).to(device))
                dino_f = feats.cpu().numpy()
                dino_f = dino_f.reshape(128, -1)
                dino_f = dino_f[:,min_time:max_time]
                self.EEGs.append(torch.from_numpy(dino_f).float())

        self.isDataTransformed = True
        print("Transforming Image data to dino EEG features (done)")

    
    def __len__(self):
        return len(self.images)

    
    def __getitem__(self, idx):
        # ImagePath = self.images[idx][:]
        # image = Image.open(ImagePath).convert('RGB')
        image =  self.getOriginalImage(idx)
        EEG = []
        Image_features = []

        LabelClassName = self.int_to_str_labels[self.labels[idx]] 
        LabelClasId = self.class_str_to_id[LabelClassName]

        if self.preprocessin_fn is not None:
            image = self.preprocessin_fn(image)

        if self.isDataTransformed:
            EEG = self.EEGs[idx]


        return EEG,{"ClassName": LabelClassName, "ClassId": LabelClasId},image, idx, Image_features