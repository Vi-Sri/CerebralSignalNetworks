from utils.PerilsEEGDataset import EEGDataset
from utils.Utilities import Utilities, evaluate
import torch
from torch.utils.data import DataLoader
from models.lstm import Model
import os
import argparse
from torchvision import transforms, datasets
from utils import utils
import numpy as np
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F


class HyperParams:
    learning_rate=0.001
    T=0.5
    soft_target_loss_weight=0.25
    ce_loss_weight=0.75
    warmup_teacher_temp = 1.5
    teacher_temp = 0.22
    warmup_teacher_temp_epochs = 50
    alpha = 0.5
    beta = 0.5


class Parameters:
    ce_loss_weight = 0.95
    mse_loss_weight = 0.20
    soft_target_loss_weight = 0.05
    alpha = 0.5
    teacher_temp = 0.05
    student_temp=0.1

class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cosine_similarity = nn.CosineSimilarity()

    def forward(self, student_outputs, teacher_outputs):
        loss = 1 - self.cosine_similarity(student_outputs, teacher_outputs).mean()
        return loss

class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        HyperParams.T = self.teacher_temp_schedule[epoch]

        student_out = student_output / self.student_temp
        # student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        # teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        loss = torch.sum(-teacher_out * F.log_softmax(student_out, dim=-1), dim=-1)
        total_loss += loss.mean()

        # total_loss = 0
        # n_loss_terms = 0
        # for iq, q in enumerate(teacher_out):
        #     for v in range(len(student_out)):
        #         if v == iq:
        #             # we skip cases where student and teacher operate on the same view
        #             continue
        #         loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
        #         total_loss += loss.mean()
        #         n_loss_terms += 1
        # total_loss /= n_loss_terms

        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)   

class FeatureDistributionLoss(nn.Module):
    def __init__(self, nepochs, warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs):
        super().__init__()
        self.mse = nn.MSELoss()

        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))


    def forward(self, student_outputs, teacher_outputs, epoch, label, pred_label=None):
        overall_loss = 0


        HyperParams.T = self.teacher_temp_schedule[epoch]

        teacher_logits_with_T = nn.functional.softmax(teacher_outputs/HyperParams.T, dim=-1)
        student_logits_with_T = nn.functional.softmax(student_outputs/HyperParams.T, dim=-1)
        # student_logits_without_T = nn.functional.softmax(student_outputs, dim=-1)

        term1 = HyperParams.alpha * F.cross_entropy(pred_label,label) 
        term2 = HyperParams.beta *  F.cross_entropy(teacher_logits_with_T,student_logits_with_T)

        overall_loss +=term1
        overall_loss +=term2

        # ce_loss_lables = F.cross_entropy(soft_prob,soft_targets)
        # ce_loss = F.cross_entropy(pred_label, label)
        # overall_loss = HyperParams.soft_target_loss_weight * ce_loss_lables + HyperParams.ce_loss_weight * ce_loss


        return overall_loss



def initDinoV2Model(model= "dinov2_vits14"):
    dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", model)
    return dinov2_vits14

if __name__=="__main__":

    parser = argparse.ArgumentParser('CNN Exercise.')
    parser.add_argument('--learning_rate',
                        type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=100,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size',
                        type=int, default=16,
                        help='Batch size. Must divide evenly into the dataset sizes.')
    parser.add_argument('--log_dir',
                        type=str,
                        default='./logs/DinoV2LstmDistillv2sdsad/',
                        help='Directory to put logging.')
    parser.add_argument('--gallery_subject',
                        type=int,
                        default=1,
                        choices=[0,1,2,3,4,5,6],
                        help='Subject Data to train')
    parser.add_argument('--query_subject',
                        type=int,
                        default=1,
                        choices=[0,1,2,3,4,5,6],
                        help='Subject Data to train')
    parser.add_argument('--eeg_dataset',
                        type=str,
                        default="./data/eeg/theperils/spampinato-1-IMAGE_BLOCK_RAW_with_mean_std.pth",
                        help='Dataset to train')
    parser.add_argument('--images_root',
                        type=str,
                        default="./data/images/imageNet_images",
                        help='Dataset to train')
    parser.add_argument('--eeg_dataset_split',
                        type=str,
                        default="./data/eeg/block_splits_by_image_all.pth",
                        help='Dataset split')
    parser.add_argument('--mode',
                        type=str,
                        default="train",
                        help='type of mode train or test')
    parser.add_argument('--custom_model_weights',
                        type=str,
                        default="",
                        help='custom model weights')
    parser.add_argument('--search_gallery',
                        type=str,
                        default="train",
                        help='dataset in which images will be searched')
    parser.add_argument('--query_gallery',
                        type=str,
                        default="test",
                        help='dataset in which images will be searched')
    parser.add_argument('--topK',
                        type=int,
                        default=5,
                        help='Top-k paramter, defaults to 5')
    parser.add_argument('--gallery_tranformation_type',
                        type=str,
                        default="eeg2eeg",
                        choices=["img", "img2eeg", "eeg", "eeg2eeg"],
                        help='type of tansformation needed to be done to create search gallery')
    parser.add_argument('--query_tranformation_type',
                        type=str,
                        default="eeg2eeg",
                        choices=["img", "img2eeg", "eeg", "eeg2eeg"],
                        help='type of tansformation needed to be done to create query instances')
    parser.add_argument('--hyperprams',
                        type=str,
                        default="{'ce_loss_weight': 0.50, 'soft_target_loss_weight':0.50,'alpha': 1,'temperature':2}",
                        help='hyper params for training model, pass dict tpye in string format')
    parser.add_argument('--seed', default=43, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")


    Utilities_handler = Utilities()
    
    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS)

    utils.init_distributed_mode(FLAGS)

    os.makedirs(FLAGS.log_dir, exist_ok=True)

    SUBJECT = FLAGS.gallery_subject
    BATCH_SIZE = FLAGS.batch_size
    learning_rate = FLAGS.learning_rate
    EPOCHS = FLAGS.num_epochs
    SaveModelOnEveryEPOCH = 100
    EEG_DATASET_PATH = FLAGS.eeg_dataset
    validation_frequency = 5
    # EEG_DATASET_SPLIT = "./data/eeg/block_splits_by_image_all.pth"

    hyperprams = eval(FLAGS.hyperprams)
    if 'alpha' in hyperprams:
        Parameters.alpha = hyperprams["alpha"]
    if 'ce_loss_weight' in hyperprams:
        Parameters.ce_loss_weight = hyperprams["ce_loss_weight"]
    if 'soft_target_loss_weight' in hyperprams:
        Parameters.soft_target_loss_weight = hyperprams["soft_target_loss_weight"]
    if 'temperature' in hyperprams:
        Parameters.temperature = hyperprams["temperature"]



    transform_image = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256, antialias=True),       
        transforms.CenterCrop(224),  
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  
    ])



    dataset = EEGDataset(eeg_signals_path="./data/eeg/theperils/spampinato-1-IMAGE_RAPID_RAW_with_mean_std.pth", 
                        eeg_splits_path=None,
                        preprocessin_fn=transform_image, 
                        time_low=20, 
                        time_high=480,
                        apply_channel_wise_norm=False,
                        apply_norm_with_stds_and_means=False)
    
    eeg, label,image,i, image_features = dataset[0]
    print(eeg.shape)
    # Utilities_handler.plotSampleEEGChannels(eeg_data=[eeg], channels_to_plot=[0])
    # Utilities_handler.plotSampleEEGChannels(eeg_data=[eeg], channels_to_plot=[26])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dinov2_model = initDinoV2Model(model="dinov2_vits14").to(device)
    dinov2_model = dinov2_model.eval()
    dinov2_model.to(device)

    data_loader_train = DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=True)
    dataset.extract_features(model=dinov2_model, data_loader=data_loader_train, replace_eeg=False)

    generator1 = torch.Generator().manual_seed(43)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [0.8, 0.2], generator=generator1)
    data_loader_train = DataLoader(train_ds, batch_size=FLAGS.batch_size, shuffle=True)
    data_loader_val = DataLoader(val_ds, batch_size=FLAGS.batch_size, shuffle=False)

    eeg, label,image,i, image_features = next(iter(data_loader_train))
    # print("label len ", len(label), label)
    outs = dinov2_model(image.to(device))
    features_length = outs.size(-1)
    print(outs.size())

    gallery_features, query_features = [], []
    gallery_labels, query_labels = [], []

    for idx,data in enumerate(data_loader_train):
        eeg, label,image,i, image_features = data
        for fidx, fet in enumerate(image_features):
            gallery_features.append(fet)
            gallery_labels.append(data_loader_train.dataset.dataset.getLabelbyIndex(i[fidx]))
    
    for idx,data in enumerate(data_loader_val):
        eeg, label,image,i, image_features = data
        for fidx, fet in enumerate(image_features):
            query_features.append(fet)
            query_labels.append(data_loader_val.dataset.dataset.getLabelbyIndex(i[fidx]))

    del dinov2_model
    torch.cuda.empty_cache()
    
    print("Evaluating DINOv2")
    Recall_Total, Precision_Total = evaluate(FLAGS=FLAGS,gallery_features=gallery_features, query_features=query_features,
    gallery_labels=gallery_labels,query_labels=query_labels,dataset=dataset)

    
    LSTM_model = Model(input_size=96,lstm_size=96,lstm_layers=2,output_size=features_length, include_top=True)
    LSTM_model.to(device)

    lstmout, cls_ = LSTM_model(eeg.to(device))
    print(lstmout.size() , cls_.size())

    opt = torch.optim.RMSprop(lr=learning_rate, params=LSTM_model.parameters())
    criterion_feature_dist = FeatureDistributionLoss(nepochs=EPOCHS, 
                                        warmup_teacher_temp=HyperParams.warmup_teacher_temp, 
                                        teacher_temp=HyperParams.teacher_temp,
                                        warmup_teacher_temp_epochs=HyperParams.warmup_teacher_temp_epochs)
    
    # criterion = CosineSimilarityLoss()
    # criterion = SupervisedContrastiveLoss()
    # criterion = DINOLoss(
    #     out_dim=features_length,
    #     ncrops=4,
    #     warmup_teacher_temp=HyperParams.warmup_teacher_temp,
    #     teacher_temp=HyperParams.teacher_temp,
    #     warmup_teacher_temp_epochs=HyperParams.warmup_teacher_temp_epochs,
    #     nepochs=EPOCHS,
    # ).cuda() 

    epoch_losses = []
    val_epoch_losses = []
    best_val_loss = None
    best_val_loss_epoch = -1

    for EPOCH in range(EPOCHS):

        batch_losses = []
        val_batch_losses = []

        LSTM_model.train()

        for data in data_loader_train:
            eeg, label,image,i, image_features = data
            image_features = torch.from_numpy(np.array(image_features)).to(device)
            # print(image_features.shape, "image features")
            # img_mean, img_std = image_features.mean(dim=-1), image_features.std(dim=-1)
            # eeg = (eeg - img_mean)/img_std
            opt.zero_grad()
            lstm_output, cls_pred = LSTM_model(eeg.to(device))


            # loss = dino_loss(lstm_output,image_features, EPOCH)
            # loss = criterion(lstm_output, image_features, EPOCH, label["ClassId"].to(device))
            # loss = criterion(lstm_output, image_features)
            loss = criterion_feature_dist(lstm_output, image_features, EPOCH, label["ClassId"].to(device), pred_label=cls_pred)
            batch_losses.append(loss.cpu().item())

            loss.backward()
            opt.step()

        batch_losses = np.array(batch_losses)
        epoch_loss = batch_losses.mean()
        epoch_losses.append(epoch_loss)
        
        if EPOCH % validation_frequency==0 and EPOCH>0:
            LSTM_model.eval()

            gallery_features, gallery_labels = dataset.transformEEGDataLSTMByList(model=LSTM_model,
                                                                                  data_loader=data_loader_train)
            query_features, query_labels = dataset.transformEEGDataLSTMByList(model=LSTM_model,
                                                                             data_loader=data_loader_val)

            Recall_Total, Precision_Total = evaluate(FLAGS=FLAGS,gallery_features=gallery_features, query_features=query_features,
                     gallery_labels=gallery_labels,query_labels=query_labels,dataset=dataset)

            for data in data_loader_val:
                eeg, label,image,i, image_features = data

                with torch.no_grad():
                    image_features = torch.from_numpy(np.array(image_features)).to(device)

                    lstm_output, cls_pred  = LSTM_model(eeg.to(device))

                    # loss = criterion(lstm_output, image_features)
                    loss = criterion_feature_dist(lstm_output, image_features, EPOCH, label["ClassId"].to(device), pred_label=cls_pred)
                    # loss = criterion(lstm_output,image_features, EPOCH, label["ClassId"].to(device))
                    # loss = dino_loss(lstm_output,image_features, EPOCH)
                    loss = loss.cpu().item()
                    val_batch_losses.append(loss)
            
            val_batch_losses = np.array(val_batch_losses)
            val_epoch_loss= val_batch_losses.mean()
            val_epoch_losses.append(val_epoch_loss)

            if best_val_loss is None:
                best_val_loss = val_epoch_loss
                best_val_loss_epoch = EPOCH
                torch.save(LSTM_model.state_dict(), f"{FLAGS.log_dir}/lstm_dinov2_best_loss.pth")
            else:
                if val_epoch_loss<best_val_loss:
                    best_val_loss_epoch = EPOCH
                    best_val_loss = val_epoch_loss
                    torch.save(LSTM_model.state_dict(), f"{FLAGS.log_dir}/lstm_dinov2_best_loss.pth")


            print(f"EPOCH {EPOCH} train_loss: {round(epoch_loss,6)} val_loss: {round(val_epoch_loss,6)} T: {HyperParams.T} best val loss: {best_val_loss} on epoch: {best_val_loss_epoch}")
        else:
            print(f"EPOCH {EPOCH} train_loss: {round(epoch_loss,6)} T: {HyperParams.T}")






"""
Losses
# print(f"ce loss logits: {ce_loss_lables.item()}  label loss: {ce_loss.item()}")
# overall_loss += ce_loss
# ce_loss_logits = nn.functional.cross_entropy(student_outputs, teacher_outputs) # is negative 
# soft_targets = nn.functional.softmax(teacher_outputs / HyperParams.T, dim=-1).to(device)
# soft_prob = nn.functional.log_softmax(student_outputs / HyperParams.T, dim=-1).to(device)
# # # softLabelLoss = nn.functional.cross_entropy(soft_prob, soft_targets)
# soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (HyperParams.T**2)
# overall_loss += ce_loss_logits
# HyperParams.T = self.teacher_temp_schedule[epoch]
# soft_targets = nn.functional.softmax(teacher_outputs / HyperParams.T, dim=-1).to(device)
# soft_prob = nn.functional.log_softmax(student_outputs / HyperParams.T, dim=-1).to(device)
# # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
# soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (HyperParams.T**2)
# overall_loss += soft_targets_loss
# mse_loss = self.mse(student_outputs, teacher_outputs)
# overall_loss +=mse_loss
# overall_loss +=ce_loss
"""