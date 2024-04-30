import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import Dataset
import cv2
from utils import utils
import torch.distributed as dist
class EEGDataset(Dataset):
    # Constructor
    def __init__(self, eeg_signals_path, 
                 eeg_splits_path, subset='train',
                 subject=1, 
                 exclude_subjects=[],
                 filter_channels=[],
                 time_low=20,time_high=480, 
                 model_type="cnn", 
                 imagesRoot="./data/images/imageNet_images",
                 apply_norm_with_stds_and_means=False,
                 apply_channel_wise_norm=False,
                 preprocessin_fn=None,
                 Transform_EEG2Image_Shape=False,
                 convert_image_to_tensor=False,
                 perform_dinov2_global_Transform=False,
                 dinov2Config=None,
                 inference_mode=True,
                 onehotencode_label=False,
                 data_augment_eeg=False,
                 add_channel_dim_to_eeg=False):
        # Load EEG signals

        assert subset=='train' or subset=='val' or subset=='test'


        self.Transform_EEG2Image_Shape = Transform_EEG2Image_Shape
        self.convert_image_to_tensor = convert_image_to_tensor
        self.perform_dinov2_global_Transform = perform_dinov2_global_Transform
        self.dinov2Config = dinov2Config
        self.apply_norm_with_stds_and_means = apply_norm_with_stds_and_means
        self.apply_channel_wise_norm = apply_channel_wise_norm
        self.filter_channels = filter_channels
        self.inference_mode = inference_mode
        self.onehotencode_label = onehotencode_label
        self.data_augment_eeg = data_augment_eeg
        self.add_channel_dim_to_eeg = add_channel_dim_to_eeg

        self.time_low = time_low
        self.time_high = time_high
        self.model_type = model_type
        self.imagesRoot = imagesRoot

        # splits = torch.load(eeg_splits_path)
        # subset_indexes = splits["splits"][0][f"{subset}"]

        loaded = torch.load(eeg_signals_path)

        self.subsetData = []
        self.labels = []
        self.images = []
        self.image_features = []

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
                    self.class_labels_names[imagenetDirName] = {"ClassId": int(indexOfClass), "ClassName": imagenetClassName, "imagenetClassId": imagenetClassId}
                    self.class_id_to_str[int(indexOfClass)]= imagenetClassName
                    self.class_str_to_id[imagenetClassName]= int(indexOfClass)
        
        self.std = 0
        self.mean = 0
        cnt = 0
        
        for i in range(len(loaded["dataset"])):
            self.mean += loaded['dataset'][i]["eeg"].mean()
            self.std  += loaded['dataset'][i]["eeg"].std()
            cnt +=1
            self.subsetData.append(loaded['dataset'][i])
            self.labels.append(loaded["dataset"][i]['label'])
            self.images.append(image_names[loaded["dataset"][i]['image']])

        self.mean = self.mean/cnt
        self.std = self.std/cnt

        # Compute size
        self.size = len(self.subsetData)

        self.preprocessin_fn = None
        if preprocessin_fn is not None:
            self.preprocessin_fn = preprocessin_fn

        self.trans = transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x)
        self.transform = transforms.Compose([
                    transforms.Resize((224,224)),
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
        self.image_features_extracted = False

        if self.apply_channel_wise_norm:

            self.transformEEGDataToChannelWiseNorm()


    def __len__(self):
        return self.size

    def generate_eeg_noise_data(self, num_channels, num_samples, sampling_rate=1000, frequency=40, amplitude = 0.5):
        # Step 1: Generate Gaussian noise
        gaussian_noise = np.random.normal(0, 1, size=(num_channels, num_samples))
        # Example: Add a sinusoidal oscillation to each channel
        time = np.arange(num_samples) / sampling_rate
        eeg_data = gaussian_noise + amplitude * np.sin(2 * np.pi * frequency * time)

        return eeg_data
    
    def transformToEEGNoisyData(self):
        print("Transforming EEG data to noisy data")
        for i, image_path in enumerate(self.images):
            eeg_noisy_data = self.generate_eeg_noise_data(num_channels=128,num_samples=500,sampling_rate=1000)
            self.subsetData[i]["eeg"] = torch.from_numpy(eeg_noisy_data).float()
        self.isDataTransformed = True
        print("Transforming EEG data (done)")
    
    def getOriginalImage(self, idx):
        class_folder_name = self.images[idx].split("_")[0]
        ImagePath = f"{self.imagesRoot}/{class_folder_name}/{self.images[idx]}.JPEG"
        imageOriginal = Image.open(ImagePath).convert('RGB')
        return imageOriginal
    
    def getImagePath(self, idx):
        class_folder_name = self.images[idx].split("_")[0]
        ImagePath = f"{self.imagesRoot}/{class_folder_name}/{self.images[idx]}.JPEG"
        return ImagePath
    
    @torch.no_grad()
    def extract_features(self,model, data_loader, use_cuda=True, multiscale=False, replace_eeg=True, passEEG=False):
        metric_logger = utils.MetricLogger(delimiter="  ")
        features = None
        # for samples, index in metric_logger.log_every(data_loader, 10):
        for EEG,labels,image, index, Image_features in metric_logger.log_every(data_loader, 10):
            samples = image
            # if passEEG:
            #     samples = EEG
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
        
        for idx, f in enumerate(features):
            # img_features = f.cpu().numpy()
            if replace_eeg:
                self.subsetData[idx]["eeg"] = f.cpu().numpy()
            else:
                # print(f"feature shape: {img_features.shape}")
                self.image_features.append(f.cpu().numpy())

        
        self.image_features_extracted = True

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
            
        
        #Randonmly select 32 channels
        #NumOfChannelsToKeep = np.random.randint(min_channels, 128)
        #random_indices = np.random.choice(128, size=min_channels, replace=False) # replace false means, once channel is picked up wont be considered for selection again. 
        #repeated_data = repeated_data[random_indices]

        # print("reduced width repeated_data: ",repeated_data.shape)

        # Now we have an array of shape (IMG_H, IMG_W). We need to repeat this for 3 color channels to get IMG_HxIMG_Wx3.

        # Repeat the 2D array along a new third dimension
        output_data = np.repeat(repeated_data[np.newaxis, :, :], 3, axis=0)

        """
        Z2-score Normlization  https://arxiv.org/pdf/2210.01081.pdf
        """
        #fmean = np.mean(output_data)
        #fstd = np.std(output_data)
        #output_data = (output_data - fmean)/fstd

        # print("increased c: ",output_data.shape)

        return output_data
    
    def ExtractImageFeatures(self, preprocsessor, model, device):
        for img_idx in tqdm(range(len(self.images)), total=len(len(self.images))):
            class_folder_name = self.images[img_idx].split("_")[0]
            ImagePath = f"{self.imagesRoot}/{class_folder_name}/{self.images[img_idx]}.JPEG"
            image = Image.open(ImagePath).convert('RGB')
            with torch.no_grad():
                inputs = preprocsessor(images=image, return_tensors="pt", do_rescale=False)
                inputs = inputs.to(device)
                outputs = model(**inputs)
                last_hidden_states = outputs[0]
                features = last_hidden_states[:,0,:] # batch size, 257=CLS_Token+256,features_length
                features = features.reshape(features.size(0), -1)
                self.image_features.append(features)
        self.image_features_extracted = True
    

    
    def transformEEGDataLSTM(self, lstm_model,device, replaceEEG=True):
        lstm_model.to(device)
        lstm_model.eval()
        if not replaceEEG:
            self.image_features = [i for i in range(len(self))]

        for i in range(len(self)):
            eeg, label,image,i, image_features = self[i]
            with torch.no_grad():
                lstm_output = lstm_model(eeg.unsqueeze(0).to(device))
                if replaceEEG:
                    self.subsetData[i]["eeg"] = lstm_output
                else:
                    self.image_features[i] = lstm_output
    
    @torch.no_grad()
    def transformEEGDataLSTMByList(self, model, data_loader):
        metric_logger = utils.MetricLogger(delimiter="  ")
        # for samples, index in metric_logger.log_every(data_loader, 10):
        image_features = []
        image_labels = []
        for EEG,labels,image, index, img_feat in metric_logger.log_every(data_loader, 10):
            samples = EEG
            # print(samples.shape)
            samples = samples.cuda(non_blocking=True)

            features, cls_ = model(samples)

            for idx, feat in enumerate(features):
                image_features.append(feat.cpu().numpy())
                image_labels.append(data_loader.dataset.dataset.getLabelbyIndex(idx))

        # print(f"Image features: {len(image_features)} Labels :{len(image_labels)} ")
        return image_features,image_labels


    def transformEEGData(self, resnet_model, resnet_to_eeg_model, device, isVIT=False):
        print("Transforming EEG data")
        resnet_to_eeg_model = resnet_to_eeg_model.to(device)
        resnet_model = resnet_model.to(device)
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

                # outputs = outputs.cpu().numpy()
                # outputs = outputs.reshape(128, -1)
                # outputs = outputs[:,self.time_low:self.time_high]
                self.subsetData[i]["eeg"] = outputs
                # print("output shape : ", outputs.shape)
                # self.subsetData[i]["eeg"] = outputs
                # print("FC features shape", outputs.view(128,-1).size(), "original eeg shape: ", eeg.reshape(128,-1).size())
        self.isDataTransformed = True
        print("Transforming EEG data (done)")


    def transformEEGDataAE(self, eeg_ae_model, device):
        """
        For Autoencoder Model
        """

        print_done = False
        print("Transforming EEG data with AE model")
        for i, image_path in enumerate(self.images):

            eeg = self.subsetData[i]["eeg"].float().t()
            eeg = eeg[self.time_low:self.time_high,:]

            # eeg, label, image, idxs = data
            with torch.no_grad():  

                eeg_reshaped = eeg.reshape(1, -1) # 1 batch

                # features = features.reshape(features.size(0), -1)
                encoder_out = eeg_ae_model.encoder(eeg_reshaped.to(device))
                decoder_out = eeg_ae_model.decoder(encoder_out)
                reformed_eeg = decoder_out.reshape(eeg.size())
                self.subsetData[i]["eeg"] = reformed_eeg
                # print("FC features shape", outputs.view(128,-1).size(), "original eeg shape: ", eeg.reshape(128,-1).size())

                if not print_done:
                    print(eeg.size(), reformed_eeg.size())
                    print_done = True
        
        
        self.isDataTransformed = True
        print("Transforming EEG data (done)")


    def transformEEGDataDino(self, model, device, pass_eeg=True,preprocessor=None, min_time=0,max_time=460,do_z2_score_norm=False, keep_features_flat=False):
        # print(f"Transforming EEG data to dino features EEG, pass_eeg is {pass_eeg}")
        model = model.to(device)
        for i, image_path in tqdm(enumerate(self.images), total=len(self.images)):
        # for i, image_path in enumerate(self.images):
            
            model_inputs = None
            if pass_eeg:
                eeg = self.subsetData[i]["eeg"].float()
                eeg = eeg.cpu().numpy()
                eeg = self.resizeEEGToImageSize(eeg)

                if do_z2_score_norm:
                    """Z2-score Normlization  https://arxiv.org/pdf/2210.01081.pdf """
                    fmean = np.mean(eeg)
                    fstd = np.std(eeg)
                    eeg = (eeg - fmean)/fstd
                model_inputs = torch.from_numpy(eeg)
                # model_inputs = self.preprocessin_fn(model_inputs)
            else:
                class_folder_name = image_path.split("_")[0]
                ImagePath = f"{self.imagesRoot}/{class_folder_name}/{self.images[i]}.JPEG"
                model_inputs = Image.open(ImagePath).convert('RGB')

                if self.preprocessin_fn is not None:
                    model_inputs = self.preprocessin_fn(model_inputs)
                else:
                    if preprocessor is not None:
                        model_inputs = preprocessor(model_inputs)

            with torch.no_grad():
                dino_f = model(model_inputs.unsqueeze(0).to(device))
                if not keep_features_flat:
                    dino_f = dino_f.cpu().numpy()
                    dino_f = dino_f.reshape(128, -1)
                    dino_f = dino_f[:,min_time:max_time]
                    self.subsetData[i]["eeg"] = torch.from_numpy(dino_f)
                else:
                    self.subsetData[i]["eeg"] = dino_f
                # print("FC features shape", outputs.view(128,-1).size(), "original eeg shape: ", eeg.reshape(128,-1).size())
        self.isDataTransformed = True
        # print("Transforming EEG data to dino EEG features (done)")

    
    def normlizeEEG(self, EEG, ch_index, class_index=None):
        ChannelEEG = EEG[: , ch_index]
        ChanneLMean = ChannelEEG.mean()
        ChanneLStd = ChannelEEG.std()
        NormlizedChannelEEG = (ChannelEEG-ChanneLMean)/ChanneLStd
        # NormlizedChannelEEG[NormlizedChannelEEG > 0] = 0
        EEG[:,ch_index] = NormlizedChannelEEG
        return EEG
    

    def transformEEGDataToChannelWiseNorm(self):
        print("Transforming data to channel wise norm across labels")
        label_wise_data = {}
        channel_wise_mean_std = {}
        for i in range(len(self)):
            eeg, label, image, i ,img_f = self[i]
            if not label["ClassId"] in label_wise_data:
                label_wise_data[label["ClassId"]] = {"images":[], "eeg":[], "original_indexes": []}
            label_wise_data[label["ClassId"]]["images"].append(image)
            label_wise_data[label["ClassId"]]["eeg"].append(eeg.numpy())
            label_wise_data[label["ClassId"]]["original_indexes"].append(i)

        
        for label, labelData in label_wise_data.items():
            # print(label, len(labelData["eeg"]))
            channel_wise_mean_std = {}
            eegs = labelData["eeg"]
            timeSpan, channels = eegs[0].shape
            for eeg in eegs:
                for ch in range(channels):
                    ch_mean = eeg[:, ch].mean()
                    ch_std = eeg[:, ch].std()

                    if ch not in channel_wise_mean_std:
                        channel_wise_mean_std[ch] = {
                            "stds": [],
                            "means": [],
                            "mean": 0,
                            "std": 0
                        }
                    
                    channel_wise_mean_std[ch]["stds"].append(ch_std)
                    channel_wise_mean_std[ch]["means"].append(ch_mean)

            
            for idxes in labelData["original_indexes"]:

                eeg = self.subsetData[idxes]["eeg"].float()
                eeg = eeg.cpu().numpy()
                for ch in range(channels):
                    meanStd = np.array(channel_wise_mean_std[ch]["stds"]).mean()
                    meanMean= np.array(channel_wise_mean_std[ch]["means"]).mean()
                    eeg[:,ch] = (eeg[:,ch]-meanMean)/meanStd
                self.subsetData[i]["eeg"] = torch.from_numpy(eeg)


        # for ch in range(128):
        #     meanStd = np.array(channel_wise_mean_std[ch]["stds"]).mean()
        #     meanMean= np.array(channel_wise_mean_std[ch]["means"]).mean()
        #     channel_wise_mean_std[ch]["mean"] = meanMean
        #     channel_wise_mean_std[ch]["std"] = meanStd

        
        # for i, image_path in tqdm(enumerate(self.images), total=len(self.images)):
        #     eeg = self.subsetData[i]["eeg"].float()
        #     eeg = eeg.cpu().numpy()
        #     for ch in range(128):
        #         eeg[:,ch] = (eeg[:,ch]-channel_wise_mean_std[ch]["mean"])/channel_wise_mean_std[ch]["std"]
        #     self.subsetData[i]["eeg"] = torch.from_numpy(eeg)

        print("Transforming data to channel wise norm across labels (done)")


    
    def getLabelbyIndex(self, i):
        class_folder_name = self.images[i].split("_")[0]
        label = self.class_labels_names[class_folder_name]
        if not self.inference_mode:
            label = label["ClassId"]
            if self.onehotencode_label:
                onehot = [0 for i in range(len(self.class_labels_names))]
                onehot[label] = 1
                label = onehot
                label = np.array(label)
                label = torch.from_numpy(label)
        return label

    def __getitem__(self, i):
        eeg = self.subsetData[i]["eeg"].float()

        # print("original shape", eeg.size())
        
        if self.Transform_EEG2Image_Shape:
            eeg = eeg.cpu().numpy().T
            eeg = self.resizeEEGToImageSize(eeg)
            eeg = torch.from_numpy(eeg)
        else:
            eeg = eeg.t()

        if not self.isDataTransformed:
            if len(self.filter_channels)>0:
                eeg = eeg.cpu().numpy()
                eeg_zeros = np.zeros((self.time_high-self.time_low, len(self.filter_channels)), dtype=eeg.dtype)
                # print(f"eeg shape: {eeg.shape} eeg zeros shape: {eeg_zeros.shape}")
                for ch_idx, ch in enumerate(self.filter_channels):
                    # print(f" sliced: {eeg[self.time_low:self.time_high,ch].shape}")
                    eeg_zeros[:,ch_idx] = eeg[self.time_low:self.time_high,ch]
                    if self.apply_channel_wise_norm:
                        eeg_zeros = self.normlizeEEG(EEG=eeg_zeros,ch_index=ch_idx,class_index=None)
                eeg = torch.from_numpy(eeg_zeros).t()
                # print(f"final EEG : {eeg.size()}")
            else:
                # if self.apply_channel_wise_norm:
                #     for ch_idx in range(eeg.shape[-1]):
                #         eeg = self.normlizeEEG(EEG=eeg,ch_index=ch_idx,class_index=None)
                eeg = eeg[self.time_low:self.time_high,:]

        
        if self.apply_norm_with_stds_and_means:
            eeg = (eeg-self.mean)/ self.std
        
        if self.data_augment_eeg:
            channel_norm_eeg = eeg
            for idx_channel in range(32):
                channel_index = np.random.randint(0, channel_norm_eeg.size(-1))
                channel_norm_eeg = self.normlizeEEG(channel_norm_eeg,ch_index=channel_index, class_index=None)

            z2Scoring = eeg
            fmean = z2Scoring.mean()
            fstd = z2Scoring.std()
            z2Scoring = (z2Scoring - fmean)/fstd
            
            # eeg_fft = torch.from_numpy(np.fft.fft(eeg.cpu().numpy()))
            eeg = torch.stack((eeg,channel_norm_eeg,z2Scoring))

        if self.add_channel_dim_to_eeg:
            eeg = eeg.unsqueeze(0)

        # print("final shape", eeg.size())

        class_folder_name = self.images[i].split("_")[0]
        ImagePath = f"{self.imagesRoot}/{class_folder_name}/{self.images[i]}.JPEG"

        label = self.class_labels_names[class_folder_name]

        if not self.inference_mode:
            label = label["ClassId"]
            if self.onehotencode_label:
                onehot = [0 for i in range(len(self.class_labels_names))]
                onehot[label] = 1
                label = onehot
                label = np.array(label)
                label = torch.from_numpy(label)

        image = Image.open(ImagePath).convert('RGB')
        if self.preprocessin_fn is not None:
            # image = self.preprocessin_fn(image, eeg, i, self, local_crops_to_remove=2)
            image = self.preprocessin_fn(image)
        else:
            if self.perform_dinov2_global_Transform:
                image = self.geometric_augmentation_global(image)
            if self.convert_image_to_tensor:
                image = self.transform(image)
                # print(image.size())
        if self.image_features_extracted==True and len(self.image_features)==len(self.images):
            image_features = self.image_features[i]
        else:
            image_features = []

        return eeg, label,image,i, image_features