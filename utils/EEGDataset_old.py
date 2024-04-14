import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from torch.utils.data import Dataset

class EEGDataset(Dataset):
    # Constructor
    def __init__(self, eeg_signals_path, 
                 eeg_splits_path, subset='train',
                 subject=1, 
                 time_low=20,time_high=480, 
                 model_type="cnn", 
                 imagesRoot="./data/images/imageNet_images", 
                 preprocessin_fn=None,
                 Transform_EEG2Image_Shape=False,
                 convert_image_to_tensor=False,
                 perform_dinov2_global_Transform=False,
                 dinov2Config=None):
        # Load EEG signals

        assert subset=='train' or subset=='val' or subset=='test'


        self.Transform_EEG2Image_Shape = Transform_EEG2Image_Shape
        self.convert_image_to_tensor = convert_image_to_tensor
        self.perform_dinov2_global_Transform = perform_dinov2_global_Transform
        self.dinov2Config = dinov2Config

        self.time_low = time_low
        self.time_high = time_high
        self.model_type = model_type
        self.imagesRoot = imagesRoot

        splits = torch.load(eeg_splits_path)
        subset_indexes = splits["splits"][0][f"{subset}"]

        loaded = torch.load(eeg_signals_path)

        self.subsetData = []
        self.labels = []
        self.images = []

        self.class_labels = loaded["labels"]
        image_names = loaded['images']

        EEGSelectedImageNetClasses = []
        for imageP in image_names:
            class_folder_name = imageP.split("_")[0]
            EEGSelectedImageNetClasses.append(class_folder_name)

        self.class_labels_names = {}
        self.class_id_to_str = {}
        self.class_str_to_id = {}

        lines = []
        with open(f"{imagesRoot}/labels.txt") as f:
            lines = f.readlines()
            for i,line in enumerate(lines):
                line = line.strip()
                line = line.split(" ")
                imagenetDirName = line[0]
                imagenetClassId = line[1]
                imagenetClassName = line[-1]
                if imagenetDirName in EEGSelectedImageNetClasses:
                    indexOfClass = self.class_labels.index(imagenetDirName)
                    self.class_labels_names[imagenetDirName] = {"ClassId": int(indexOfClass), "ClassName": imagenetClassName}
                    self.class_id_to_str[int(indexOfClass)]= imagenetClassName
                    self.class_str_to_id[imagenetClassName]= int(indexOfClass)

        for sub_idx in subset_indexes:
            if subject!=0:
                if loaded['dataset'][sub_idx]['subject']==subject:
                    self.subsetData.append(loaded['dataset'][sub_idx])
                    self.labels.append(loaded["dataset"][sub_idx]['label'])
                    self.images.append(image_names[loaded["dataset"][sub_idx]['image']])
            else:
                sub_idx = int(sub_idx)
                self.subsetData.append(loaded['dataset'][sub_idx])
                self.labels.append(loaded["dataset"][sub_idx]['label'])
                self.images.append(image_names[loaded["dataset"][sub_idx]['image']])

        # Compute size
        self.size = len(self.subsetData)

        self.preprocessin_fn = None
        if preprocessin_fn is not None:
            self.preprocessin_fn = preprocessin_fn

        self.trans = transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x)
        self.transform = transforms.Compose([
                    transforms.ToTensor()
                ])
        
        
        if self.perform_dinov2_global_Transform:
            self.geometric_augmentation_global = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        self.dinov2Config.crops.global_crops_size, scale=self.dinov2Config.crops.global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                    ),
                    transforms.RandomHorizontalFlip(p=0.5),
                ]
            )

        self.isDataTransformed = False


    def __len__(self):
        return self.size
    
    def getOriginalImage(self, idx):
        class_folder_name = self.images[idx].split("_")[0]
        ImagePath = f"{self.imagesRoot}/{class_folder_name}/{self.images[idx]}.JPEG"
        imageOriginal = Image.open(ImagePath).convert('RGB')
        return imageOriginal
    
    def getImagePath(self, idx):
        class_folder_name = self.images[idx].split("_")[0]
        ImagePath = f"{self.imagesRoot}/{class_folder_name}/{self.images[idx]}.JPEG"
        return ImagePath
    

    def resizeEEGToImageSize(self,input_data=None,imageShape=(224,224),index=None):

        if input_data is None:
            if index is None:
                raise Exception("Need either input data or give dataset index")
            else:
                input_data = self.subsetData[index]["eeg"].float().t()
                input_data = input_data.cpu().numpy().T
        # print("EE sample size",self.subsetData[i]["eeg"].size())
        # eeg = self.subsetData[i]["eeg"].T
        if not self.isDataTransformed:
            input_data = input_data[self.time_low:self.time_high,:]


        # print(input_data.shape)
        IMG_H, IMG_W  = imageShape[0], imageShape[1]
        # EEG input_data is assumed to be a numpy array of shape (128, 460)

        # Repeat each channel until we reach IMG_H channels
        repeated_data = np.repeat(input_data, (IMG_H // input_data.shape[0])+1, axis=0)
        repeated_data = np.repeat(repeated_data, (IMG_W // repeated_data.shape[1])+1, axis=1)

        # print("repeated_data: ",repeated_data.shape)

        # If we have more than IMG_H channels, slice the array down to IMG_H
        if repeated_data.shape[0] > IMG_H:
            repeated_data = repeated_data[:IMG_H, :]

        # print("reduced height repeated_data: ",repeated_data.shape)

        # Now we have an array of shape (IMG_H, 460). We need to slice the time series data to get IMG_H x IMG_W.

        # If we have more than IMG_W time series data points, slice the array down to IMG_W
        if repeated_data.shape[1] > IMG_W:
            start_index = np.random.randint(0, repeated_data.shape[1]-IMG_W)
            repeated_data = repeated_data[:, start_index:start_index+IMG_W]
            # print(f"start:{start_index} end: {start_index+224}")
            # repeated_data = repeated_data[:, :IMG_W]d

        # print("reduced width repeated_data: ",repeated_data.shape)

        # Now we have an array of shape (IMG_H, IMG_W). We need to repeat this for 3 color channels to get IMG_HxIMG_Wx3.

        # Repeat the 2D array along a new third dimension
        output_data = np.repeat(repeated_data[np.newaxis, :, :], 3, axis=0)

        # set two other channels to zeros, keep only 0th channel
        # output_data[1, : ,: ] = 0.
        # output_data[2, : ,: ] = 0.


        # output_data = output_data.clip(min=4,max=100)
        # minVal = np.min(output_data)
        # output_data = minVal +  output_data
        # maxVal = np.max(output_data)
        # output_data = maxVal -  output_data # invert
        # output_data = output_data.clip(min=0)

        """
        Z2-score Normlization  https://arxiv.org/pdf/2210.01081.pdf
        """
        fmean = np.mean(output_data)
        fstd = np.std(output_data)
        output_data = (output_data - fmean)/fstd
        # output_data = output_data.clip(min=0) # to limit the pixel range

        # doesnt work, gives blank image
        # minVal = np.min(output_data)
        # maxVal = np.max(output_data)
        # output_data = (output_data - minVal)/(maxVal - minVal)
        
        # print("increased c: ",output_data.shape)

        return output_data
    
    def transformEEGData(self, resnet_model, resnet_to_eeg_model, device, isVIT=False):
        print("Transforming EEG data")
        for i, image_path in enumerate(self.images):

            class_folder_name = image_path.split("_")[0]
            ImagePath = f"{self.imagesRoot}/{class_folder_name}/{self.images[i]}.JPEG"

            image = Image.open(ImagePath).convert('RGB')
            if self.preprocessin_fn is not None:
                image = self.preprocessin_fn(image)

            # eeg, label, image, idxs = data
            with torch.no_grad():
                if isVIT:
                    features = resnet_model(image.unsqueeze(0).to(device)).last_hidden_state[:, 0]
                else:
                    features = resnet_model(image.unsqueeze(0).to(device))
                    features = features.view(-1, features.size(1))
                outputs = resnet_to_eeg_model(features)
                self.subsetData[i]["eeg"] = outputs
                # print("FC features shape", outputs.view(128,-1).size(), "original eeg shape: ", eeg.reshape(128,-1).size())
        self.isDataTransformed = True
        print("Transforming EEG data (done)")


    def transformEEGDataDino(self, model, device, pass_eeg=True,preprocessor=None, min_time=0,max_time=460):
        print(f"Transforming EEG data to dino featueres EEG, pass_eeg is {pass_eeg}")
        for i, image_path in enumerate(self.images):

            model_inputs = None
            if pass_eeg:
                eeg = self.subsetData[i]["eeg"].float()
                eeg = eeg.cpu().numpy()
                eeg = self.resizeEEGToImageSize(eeg)
                model_inputs = torch.from_numpy(eeg)
            else:
                class_folder_name = image_path.split("_")[0]
                ImagePath = f"{self.imagesRoot}/{class_folder_name}/{self.images[i]}.JPEG"
                model_inputs = Image.open(ImagePath).convert('RGB')

                if self.preprocessin_fn is not None:
                    model_inputs = self.preprocessin_fn(model_inputs)
                else:
                    if preprocessor is not None:
                        model_inputs = preprocessor(model_inputs)

            # eeg, label, image, idxs = data
            with torch.no_grad():
                feats = model(model_inputs.unsqueeze(0).to(device))
                dino_f = feats.cpu().numpy()
                dino_f = dino_f.reshape(128, -1)
                dino_f = dino_f[:,min_time:max_time]
                self.subsetData[i]["eeg"] = torch.from_numpy(dino_f)
                # print("FC features shape", outputs.view(128,-1).size(), "original eeg shape: ", eeg.reshape(128,-1).size())
        self.isDataTransformed = True
        print("Transforming EEG data to dino EEG features (done)")


    def __getitem__(self, i):
        eeg = self.subsetData[i]["eeg"].float().t()
        # print("EE sample size",self.subsetData[i]["eeg"].size())
        # eeg = self.subsetData[i]["eeg"].T
        if not self.isDataTransformed:
            eeg = eeg[self.time_low:self.time_high,:]

        if self.Transform_EEG2Image_Shape:
            eeg = eeg.cpu().numpy().T
            eeg = self.resizeEEGToImageSize(eeg)
            eeg = torch.from_numpy(eeg)

        class_folder_name = self.images[i].split("_")[0]
        ImagePath = f"{self.imagesRoot}/{class_folder_name}/{self.images[i]}.JPEG"

        label = self.class_labels_names[class_folder_name]

        image = Image.open(ImagePath).convert('RGB')
        if self.preprocessin_fn is not None:
            # image = self.preprocessin_fn(image, eeg, i, self, local_crops_to_remove=2)
            image = self.preprocessin_fn(image)
        else:
            if self.perform_dinov2_global_Transform:
                image = self.geometric_augmentation_global(image)
            if self.convert_image_to_tensor:
                image = self.transform(image)

        return eeg, label, image, i