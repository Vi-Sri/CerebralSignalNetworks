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

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

class Parameters:
    ce_loss_weight = 0.50
    soft_target_loss_weight = 0.50
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

    mse_loss = F.smooth_l1_loss(student_logits, teacher_logits)
                        
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
        # print(x.size())
        batch_size, timespan, channels = x.size()
        x = x.view(batch_size, channels, timespan)
        lstm_init = (torch.zeros(self.n_layer, batch_size, self.hidden_size), torch.zeros(self.n_layer, batch_size, self.hidden_size))
        if x.is_cuda: lstm_init = (lstm_init[0].cuda(), lstm_init[0].cuda())
        lstm_init = (Variable(lstm_init[0], volatile=x.volatile), Variable(lstm_init[1], volatile=x.volatile))

        # Forward LSTM and get final state
        x = self.lstm(x, lstm_init)[0][:,-1,:]

        # hx0, hx1 =  self.lstm(x, lstm_init)[1]
        # print(hx0.size(), hx1.size())
        # x = F.softmax(self.fc(x))
        x = self.fc(x)

        return x
        # h0 = torch.zeros(self.n_layer, x.size(0), self.hidden_size)
        # c0 = torch.zeros(self.n_layer, x.size(0), self.hidden_size)
        # lstm_out, hidden_out = self.lstm(x, (h0, c0))
        # # out = self.fc(lstm_out[:, -1, :])
        # return lstm_out 


if __name__=="__main__":


    parser = argparse.ArgumentParser('CNN Exercise.')
    parser.add_argument('--learning_rate',
                        type=float, default=0.0001,
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
                        default='./logs/DINOvsDinIE/',
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
                        default="./data/eeg/theperils/spampinato-1-IMAGE_RAPID_RAW_with_mean_std.pth",
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
                        default="./weights/lstm_dinov2_best_loss_lr_0.00001_2.21_loss.pth",
                        help='custom model weights')
    parser.add_argument('--dino_base_model_weights',
                        type=str,
                        default="./models/pretrains/dino_deitsmall8_pretrain_full_checkpoint.pth",
                        help='dino based model weights')
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


    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS)

    output_dir = FLAGS.log_dir + "/output" 
    os.makedirs(FLAGS.log_dir + "/output" , exist_ok=True)

    SUBJECT = FLAGS.gallery_subject
    BATCH_SIZE = FLAGS.batch_size
    learning_rate = FLAGS.learning_rate
    EPOCHS = FLAGS.num_epochs
    SaveModelOnEveryEPOCH = 100
    EEG_DATASET_PATH = FLAGS.eeg_dataset
    # EEG_DATASET_SPLIT = "./data/eeg/block_splits_by_image_all.pth"

    LSTM_INPUT_FEATURES = 128 # should be image features output.
    LSTM_HIDDEN_SIZE = 460  # should be same as sequence length
    selectedDataset = "imagenet40"

    hyperprams = eval(FLAGS.hyperprams)
    if 'alpha' in hyperprams:
        Parameters.alpha = hyperprams["alpha"]
    if 'ce_loss_weight' in hyperprams:
        Parameters.ce_loss_weight = hyperprams["ce_loss_weight"]
    if 'soft_target_loss_weight' in hyperprams:
        Parameters.soft_target_loss_weight = hyperprams["soft_target_loss_weight"]
    if 'temperature' in hyperprams:
        Parameters.temperature = hyperprams["temperature"]

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
                         imagesRoot=FLAGS.images_root,
                         time_high=480,
                         exclude_subjects=[],
                         convert_image_to_tensor=False,
                         apply_channel_wise_norm=False,
                         preprocessin_fn=transform_image)


    eeg, label,image,i, image_features =  dataset[0]


    temporal_length,channels = eeg.size()
    print(temporal_length,channels)

    LSTM_INPUT_FEATURES = channels
    LSTM_HIDDEN_SIZE = temporal_length  # should be same as sequence length

    model = LSTMModel(input_size=LSTM_HIDDEN_SIZE,hidden_size=LSTM_INPUT_FEATURES, out_features=384, n_layers=4)
    if os.path.exists(FLAGS.custom_model_weights):
        model.load_state_dict(torch.load(FLAGS.custom_model_weights))
        print(f"Loaded weights from: {FLAGS.custom_model_weights}")
    model.to(device)

    time_t0 = time.perf_counter()


    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # dinov2_model = initDinoV2Model(model="dinov2_vits14").to(device)
    # dinov2_model = dinov2_model.eval()
    # dinov2_model.to(device)

    data_loader_train = DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=True)
    dataset.transformEEGDataLSTM(lstm_model=model,device=device,replaceEEG=True)
    # dataset.extract_features(model=model, data_loader=data_loader_train, replace_eeg=False)


    generator1 = torch.Generator().manual_seed(43)
    dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2], generator=generator1)

    dataset = dataset.dataset
    test_dataset = test_dataset.dataset
    
    # data_loader_train = DataLoader(train_ds, batch_size=FLAGS.batch_size, shuffle=True)
    # data_loader_val = DataLoader(test_dataset, batch_size=FLAGS.batch_size, shuffle=True)


    gallery_features = []
    query_features = []
    for i in range(len(dataset)):
        gallery_features.append(dataset.subsetData[i]["eeg"].cpu().numpy())
    for i in range(len(test_dataset)):
        query_features.append(test_dataset.subsetData[i]["eeg"].cpu().numpy())

    gallery_features = torch.from_numpy(np.array(gallery_features))
    query_features = torch.from_numpy(np.array(query_features))

    gallery_features = gallery_features.reshape(gallery_features.size(0), -1)
    query_features = query_features.reshape(query_features.size(0), -1)

    d = gallery_features.size(-1)    # dimension
    nb = gallery_features.size(0)    # database size
    nq = query_features.size(0)      # nb of queries
    
    index = faiss.IndexFlatL2(d)   # build the index
    print(index.is_trained)
    index.add(gallery_features)    # add vectors to the index
    print(index.ntotal)

    topK  = FLAGS.topK
    k = FLAGS.topK                          # we want to see 4 nearest neighbors
    D, I = index.search(gallery_features[:5], k) # sanity check
    print(I)
    print(D)
    D, I = index.search(query_features, k)     # actual search
    print(I[:5])                   # neighbors of the 5 first queries
    # print(I[-5:])                # neighbors of the 5 last queries

    class_scores = {"data" :{}, "metadata": {}}
    class_scores["metadata"] = {"flags": FLAGS}
    print_done = False

    
    for query_idx, search_res in enumerate(I):
        # print(search_res)
        labels = []
        test_intlabel = test_dataset.labels[query_idx]
        test_strlabel = test_dataset.class_id_to_str[test_intlabel]

        cosine_similarities = []
        cosine_similarities_labels_int = []
        cosine_similarities_labels_str = []
        cosine_similarities_labels_classid = []
        cosine_similarities_images = []

        test_intlabel = test_dataset.labels[query_idx]
        test_strlabel = test_dataset.class_id_to_str[test_intlabel]

        test_eeg, test_label, test_image, test_idx, img_f = test_dataset[query_idx]
        #originalImage = test_dataset.getOriginalImage(test_idx)
        originalImage = test_dataset.getImagePath(test_idx)

        if test_label["ClassName"] not in class_scores["data"]:
            class_scores["data"][test_label["ClassName"]] = {"TP": 0, 
                                                    "classIntanceRetrival": 0,
                                                    "TotalRetrival": 0,
                                                    "TotalClass": 0, 
                                                    "input_images": [],
                                                    "GroundTruths": [], 
                                                    "Predicted":[], 
                                                    "Topk": {
                                                        "labels": [], 
                                                        "scores": [],
                                                        "images": []
                                                        },
                                                    "Recall": "",
                                                    "Precision": ""
                                                    }
            
        for search_res_idx in search_res:
            intlabel = dataset.labels[search_res_idx]
            strLabel = dataset.class_id_to_str[intlabel]
            cosine_similarities_labels_str.append(strLabel)
            cosine_similarities_labels_int.append(intlabel)
                
        cosine_similarities.append(list(D[query_idx]))
        unique, counts = np.unique(cosine_similarities_labels_str, return_counts=True)
        count = 0
        count_label = ""
        
        for u, c in zip(unique, counts):
            if u==test_strlabel:
                count = c
                count_label = u
        
        classIntanceRetrival = count
        TotalRetrival = topK


        if test_label["ClassName"] in cosine_similarities_labels_str:
            class_scores["data"][test_label["ClassName"]]["TP"] +=1
            class_scores["data"][test_label["ClassName"]]["classIntanceRetrival"] +=classIntanceRetrival
            class_scores["data"][test_label["ClassName"]]["Predicted"].append(test_label["ClassId"])
        else:
            class_scores["data"][test_label["ClassName"]]["Predicted"].append(test_dataset.class_str_to_id[cosine_similarities_labels_str[0]])

            
        class_scores["data"][test_label["ClassName"]]["TotalRetrival"] +=TotalRetrival
        class_scores["data"][test_label["ClassName"]]["TotalClass"] +=1

        class_scores["data"][test_label["ClassName"]]["Topk"]["labels"].append(list(cosine_similarities_labels_str))
        class_scores["data"][test_label["ClassName"]]["Topk"]["scores"].append(list(cosine_similarities))
        class_scores["data"][test_label["ClassName"]]["Topk"]["images"].append(list(cosine_similarities_images))
        
        class_scores["data"][test_label["ClassName"]]["input_images"].append(originalImage)
        class_scores["data"][test_label["ClassName"]]["GroundTruths"].append(test_label["ClassId"])

        TP  = class_scores["data"][test_label["ClassName"]]['TP']
        TotalClass = class_scores["data"][test_label["ClassName"]]['TotalClass']
        classIntanceRetrival = class_scores["data"][test_label["ClassName"]]['classIntanceRetrival']
        TotalRetrival = class_scores["data"][test_label["ClassName"]]['TotalRetrival']

        class_scores["data"][test_label["ClassName"]]["Recall"] = round(((TP*100)/TotalClass), 2)
        class_scores["data"][test_label["ClassName"]]["Precision"] = round(((classIntanceRetrival*100)/TotalRetrival), 2)


    Recall_Total = []
    Precision_Total = []
    for key, cls_data in class_scores["data"].items():
        print(f"Class : {key} Recall: [{cls_data['Recall']}] Precision: [{cls_data['Precision']}]" )
        Recall_Total.append(cls_data["Recall"])
        Precision_Total.append(cls_data["Precision"])

    Recall_Total = np.array(Recall_Total).mean()
    Precision_Total = np.array(Precision_Total).mean()
    print(f"Overall Recall :{Recall_Total} Overall Precision: {Precision_Total}")
    
    
    time_tn = time.perf_counter()
    outputPath = f"{output_dir}/{selectedDataset}_Scores.pth"
    class_scores["metadata"] = {"processing_time": f"{time_tn-time_t0:.2f}s"}
    
    torch.save(class_scores, outputPath)

    with open(f"{output_dir}/{selectedDataset}_Scores.txt", 'w') as f:
        json.dump(class_scores, f, indent=2, cls=NpEncoder)

    pthFiles = [outputPath]
    csv_file = open(f"{output_dir}/{selectedDataset}_.csv", "w")
    csv_file.write(f"srno, label, imagenet_label, Total class images,Total class image Retr, TP,Total Images Retr, Recall, Precision")
    cnt = 1
    for pth in pthFiles:
        class_metrics = torch.load(pth)
        filename = pth.split("train")[-1].split(".")[0]
        filename  = filename[1:]
        for key,val1 in class_metrics.items():
            if key=="data":
                val1 = dict(sorted(val1.items()))
                for classN, classData in val1.items():
                    TP  = classData['TP']
                    TotalClass = classData['TotalClass']
                    classIntanceRetrival = classData['classIntanceRetrival']
                    TotalRetrival = classData['TotalRetrival']
                    Recall = classData['Recall']
                    Precision = classData['Precision']
                    print(f"Class:{classN} TP: [{classData['TP']}] TotalClass: [{classData['TotalClass']}] classIntanceRetrival: [{classData['classIntanceRetrival']}] TotalRetrival: [{classData['TotalRetrival']}] ")
                    csv_file.write(f"\n {cnt}, {filename}, {classN}, {TotalClass},{TotalRetrival},{TP},{classIntanceRetrival},{Recall},{Precision}")
                    cnt +=1
    csv_file.write(f"\n\n,,,,,,,{Recall_Total},{Precision_Total}")                    
    csv_file.close()
    
    print(f"Completed in : {time_tn-time_t0:.2f}")

    