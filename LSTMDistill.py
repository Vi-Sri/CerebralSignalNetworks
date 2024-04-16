import torch
import argparse
import os
import torch.nn as nn
import uuid
import faiss
import json
from utils.PerilsEEGDataset import EEGDataset
import time
from torch.autograd import Variable
from torchvision import transforms, datasets
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from utils import utils


class Paramters:
    ce_loss_weight = 0.25
    soft_target_loss_weight = 0.75
    alpha = 1
    temperature = 2


def loss_fn_kd(student_logits, labels, teacher_logits, params):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha

    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    alpha = params.alpha
    T = params.temperature
    # KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
    #                          F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
    #           F.cross_entropy(outputs, labels) * (1. - alpha)

    #Soften the student logits by applying softmax first and log() second
    soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1).to(device)
    soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1).to(device)

    # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
    soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (T**2)

    # ce_loss = F.cross_entropy(nn.functional.softmax(student_logits, dim=-1), nn.functional.softmax(teacher_logits, dim=-1))

    mse_loss = F.mse_loss(student_logits, teacher_logits)
                        
    # Weighted sum of the two losses
    loss = params.soft_target_loss_weight * soft_targets_loss + params.ce_loss_weight * mse_loss

    # KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1), F.softmax(teacher_outputs/T, dim=1))

    return loss



class FLAGS:
    num_workers = 4
    dist_url = "env://"
    local_rank = 0
    batch_size = 4

def initDinoV2Model(model= "dinov2_vits14"):
    dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", model)
    return dinov2_vits14

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=2, out_features=384):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.n_layer = n_layers
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, out_features)
    
    def forward(self, x):
        batch_size = x.size(0)
        lstm_init = (torch.zeros(self.n_layer, batch_size, self.hidden_size), torch.zeros(self.n_layer, batch_size, self.hidden_size))
        if x.is_cuda: lstm_init = (lstm_init[0].cuda(), lstm_init[0].cuda())
        lstm_init = (Variable(lstm_init[0], volatile=x.volatile), Variable(lstm_init[1], volatile=x.volatile))

        # Forward LSTM and get final state
        x = self.lstm(x, lstm_init)[0][:,-1,:]
        # x = F.softmax(self.fc(x))
        x = self.fc(x)

        return x
        # h0 = torch.zeros(self.n_layer, x.size(0), self.hidden_size)
        # c0 = torch.zeros(self.n_layer, x.size(0), self.hidden_size)
        # lstm_out, hidden_out = self.lstm(x, (h0, c0))
        # # out = self.fc(lstm_out[:, -1, :])
        # return lstm_out 


if __name__=="__main__":

    SUBJECT = 1
    BATCH_SIZE = 8
    learning_rate = 0.0001
    EPOCHS = 50
    SaveModelOnEveryEPOCH = 100
    EEG_DATASET_PATH = "./data/eeg/theperils/spampinato-1-3RAW_with_mean_std.pth"
    # EEG_DATASET_SPLIT = "./data/eeg/block_splits_by_image_all.pth"

    LSTM_INPUT_FEATURES = 128 # should be image features output.
    LSTM_HIDDEN_SIZE = 460  # should be same as sequence length
    selectedDataset = "imagenet40"


    utils.init_distributed_mode(FLAGS)

    transform_image = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256, antialias=True),       
        transforms.CenterCrop(224),  
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  
    ])

    dataset = EEGDataset(subset="train",
                         eeg_signals_path=EEG_DATASET_PATH,
                         eeg_splits_path=None, 
                         subject=SUBJECT,
                         time_low=0,
                         imagesRoot="./data/images/imageNet_images",
                         time_high=480,
                         exclude_subjects=[],
                         convert_image_to_tensor=False,
                         apply_channel_wise_norm=True,
                         preprocessin_fn=transform_image)


    eeg, label,image,i, image_features =  dataset[0]

    temporal_length,channels = eeg.size()
    print(temporal_length,channels)

    LSTM_INPUT_FEATURES = channels
    LSTM_HIDDEN_SIZE = temporal_length  # should be same as sequence length


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dinov2_model = initDinoV2Model(model="dinov2_vits14").to(device)
    dinov2_model = dinov2_model.eval()
    dinov2_model.to(device)

    data_loader_train = DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=True)
    dataset.extract_features(model=dinov2_model, data_loader=data_loader_train, replace_eeg=False)

    generator1 = torch.Generator().manual_seed(123)
    # HAR_train_ds, HAR_val_ds, HAR_test_ds = torch.utils.data.random_split(HAR_data, [0.7, 0.2, 0.1], generator=generator1)
    
    train_ds, val_ds = torch.utils.data.random_split(dataset, [0.8, 0.2], generator=generator1)
    data_loader_train = DataLoader(train_ds, batch_size=FLAGS.batch_size, shuffle=True)
    data_loader_val = DataLoader(val_ds, batch_size=FLAGS.batch_size, shuffle=True)

    # train_ds.dataset.extract_features(model=dinov2_model, data_loader=data_loader_train, replace_eeg=False)
    # val_ds.dataset.extract_features(model=dinov2_model, data_loader=data_loader_val, replace_eeg=False)

    eeg, label,image,i, image_features = next(iter(data_loader_train)) 
    outs = dinov2_model(image.to(device))
    features_length = outs.size(-1)
    print(outs.size())

    model = LSTMModel(input_size=LSTM_INPUT_FEATURES,hidden_size=LSTM_HIDDEN_SIZE, out_features=features_length)
    model.to(device)
    # model

    output = model(eeg.to(device))
    print(output.size())    


    # criterion = CustomLoss()
    opt = torch.optim.Adam(lr=0.001, params=model.parameters())
    # criterion = 

    epoch_losses = []
    val_epoch_losses = []
    for EPOCH in range(50):

        batch_losses = []
        val_batch_losses = []

        model.train()

        for data in data_loader_train:
            eeg, label,image,i, image_features = data

            image_features = torch.from_numpy(np.array(image_features)).to(device)
            # print(image_features.size())

            opt.zero_grad()
            lstm_output = model(eeg.to(device))
            # dinov2_out = dinov2_model(image.to(device))

            # loss = criterion(image_features, lstm_output)
            loss = loss_fn_kd(student_logits=lstm_output,labels=None,teacher_logits=image_features, params=Paramters)
            batch_losses.append(loss.item())

            loss.backward()
            opt.step()

        model.eval()

        for data in data_loader_val:
            eeg, label,image,i, image_features = data

            with torch.no_grad():
                image_features = torch.from_numpy(np.array(image_features)).to(device)
                lstm_output = model(eeg.to(device))
                # loss = criterion(image_features, lstm_output)
                loss = loss_fn_kd(student_logits=lstm_output,labels=None,teacher_logits=image_features, params=Paramters)
                val_batch_losses.append(loss.item())
        
        batch_losses = np.array(batch_losses)
        val_batch_losses = np.array(val_batch_losses)
        val_epoch_loss= val_batch_losses.mean()
        epoch_loss = batch_losses.mean()
        epoch_losses.append(epoch_loss)
        val_epoch_losses.append(val_epoch_loss)

        print(f"EPOCH {EPOCH} train_loss: {epoch_loss} val_loss: {val_epoch_loss}")










    