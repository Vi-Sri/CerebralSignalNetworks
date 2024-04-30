from utils.PerilsEEGDataset import EEGDataset
from utils.Utilities import Utilities, NpEncoder
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
import faiss
import json
import time

class HyperParams:
    learning_rate=0.001
    T=0.5
    soft_target_loss_weight=0.25
    ce_loss_weight=0.75
    warmup_teacher_temp = 1.7
    teacher_temp = 0.23
    warmup_teacher_temp_epochs = 50

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


    def forward(self, student_outputs, teacher_outputs, epoch):
        overall_loss = 0
        # distribution loss
        # student_mean, student_std = student_outputs.mean(), student_outputs.std()
        # teacher_mean, teacher_std = teacher_outputs.mean(), teacher_outputs.std()
        # mean_mse = self.mse(student_mean, teacher_mean)
        # mean_std = self.mse(student_std, teacher_std)

        # mse_loss = self.mse(student_outputs, teacher_outputs)
        # student_out = student_outputs / Parameters.student_temp
        # teacher_out = F.softmax(teacher_outputs/Parameters.teacher_temp, dim=-1)
        # # teacher_out = teacher_out.detach().chunk(2)
        # total_loss = 0
        # loss = torch.sum(-teacher_out * F.log_softmax(student_out, dim=-1), dim=-1)
        # total_loss += loss.mean()

        HyperParams.T = self.teacher_temp_schedule[epoch]

        soft_targets = nn.functional.softmax(teacher_outputs / HyperParams.T, dim=-1).to(device)
        soft_prob = nn.functional.log_softmax(student_outputs / HyperParams.T, dim=-1).to(device)

        # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
        soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (HyperParams.T**2)

        overall_loss += soft_targets_loss

        # ce_loss = nn.functional.cross_entropy(soft_targets, soft_prob)

        return overall_loss



def initDinoV2Model(model= "dinov2_vits14"):
    dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", model)
    return dinov2_vits14

if __name__=="__main__":

    parser = argparse.ArgumentParser('CNN Exercise.')
    parser.add_argument('--learning_rate',
                        type=float, default=0.0001,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=50,
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
                        default="./weights/checkpoint0190.pth",
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


    Utilities_handler = Utilities()
    
    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS)

    utils.init_distributed_mode(FLAGS)

    os.makedirs(FLAGS.log_dir, exist_ok=True)
    output_dir = FLAGS.log_dir

    SUBJECT = FLAGS.gallery_subject
    BATCH_SIZE = FLAGS.batch_size
    learning_rate = FLAGS.learning_rate
    EPOCHS = FLAGS.num_epochs
    SaveModelOnEveryEPOCH = 100
    EEG_DATASET_PATH = FLAGS.eeg_dataset
    validation_frequency = 5
    selectedDataset = f"Theperils_sub_{SUBJECT}"
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
                        time_high=480)
    


    
    eeg, label,image,i, image_features = dataset[0]
    print(eeg.shape)
    # Utilities_handler.plotSampleEEGChannels(eeg_data=[eeg], channels_to_plot=[0])
    # Utilities_handler.plotSampleEEGChannels(eeg_data=[eeg], channels_to_plot=[26])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dinov2_model = initDinoV2Model(model="dinov2_vits14").to(device)
    dinov2_model = dinov2_model.eval()
    dinov2_model.to(device)

    data_loader = DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=True)
    # dataset.extract_features(model=dinov2_model, data_loader=data_loader, replace_eeg=False)
    
    eeg, label,image,i, image_features = next(iter(data_loader)) 
    outs = dinov2_model(image.to(device))
    features_length = outs.size(-1)
    print(outs.size())

    embed_dim = 128
    # lstm_embedding_dim = 128

    LSTM_model = Model(input_size=96,lstm_size=embed_dim,lstm_layers=4,output_size=embed_dim, include_top=False)
    loaded_state_dict = torch.load(FLAGS.custom_model_weights)
    loaded_state_dict = {k.replace("backbone.", ""): v for k, v in loaded_state_dict["teacher"].items()}
    print(loaded_state_dict.keys())
    
    LSTM_model.load_state_dict(loaded_state_dict,strict=False)
    print(f"Loaded: {FLAGS.custom_model_weights}")
    LSTM_model.to(device)
    LSTM_model.eval()

    lstmout = LSTM_model(eeg.to(device))
    print(lstmout.size())

    time_t0 = time.perf_counter()

    # dataset.transformEEGDataLSTM(lstm_model=LSTM_model,device=device,replaceEEG=True)

    generator1 = torch.Generator().manual_seed(43)
    ds_train, ds_test = torch.utils.data.random_split(dataset, [0.8, 0.2], generator=generator1)

    data_loader_train = DataLoader(ds_train, batch_size=FLAGS.batch_size, shuffle=False)
    data_loader_test = DataLoader(ds_test, batch_size=FLAGS.batch_size, shuffle=False)

    print(f"data r:: {len(ds_train)}  test {len(ds_test)} {len(dataset)} ")

    gallery_features, gallery_labels = dataset.transformEEGDataLSTMByList(model=LSTM_model,data_loader=data_loader_train)
    query_features, query_labels = dataset.transformEEGDataLSTMByList(model=LSTM_model,data_loader=data_loader_test)

    print(len(gallery_features), len(query_features))


    # print(len(dataset.dataset.subsetData), len(test_dataset.dataset.subsetData))
    # dataset = dataset.dataset
    # test_dataset = test_dataset.dataset
    # data_loader_train = DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=True)
    # data_loader_val = DataLoader(test_dataset, batch_size=FLAGS.batch_size, shuffle=True)

    # gallery_features = []
    # query_features = []
    # # for i in range(len(dataset)):
    # for test_eeg, test_label, test_image, test_idx, img_f in dataset:
    #     gallery_features.append(test_eeg.cpu().numpy())

    # # for i in range(len(test_dataset)):
    # for test_eeg, test_label, test_image, test_idx, img_f in test_dataset:
    #     query_features.append(test_eeg.cpu().numpy())

    
    gallery_features = torch.from_numpy(np.array(gallery_features))
    query_features = torch.from_numpy(np.array(query_features))
    gallery_features = gallery_features.reshape(gallery_features.size(0), -1)
    query_features = query_features.reshape(query_features.size(0), -1)

    # query_features = gallery_features
    print(gallery_features.shape, query_features.shape)
    

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
        # test_intlabel = test_dataset.dataset.labels[query_idx]
        # test_strlabel = test_dataset.dataset.class_id_to_str[test_intlabel]
        test_intlabel = query_labels[query_idx]["ClassId"]
        test_strlabel = dataset.class_id_to_str[test_intlabel]
        test_label = query_labels[query_idx]

        cosine_similarities = []
        cosine_similarities_labels_int = []
        cosine_similarities_labels_str = []
        cosine_similarities_labels_classid = []
        cosine_similarities_images = []


        # test_eeg, test_label, test_image, test_idx, img_f = test_dataset[query_idx]

        # student_mean, student_std = test_eeg.mean(), test_eeg.std()
        # print(img_f)
        # teacher_mean, teacher_std = img_f.mean(), img_f.std()
        # print(f"Student mean {student_mean} std: {student_std}  Teacher mean:{teacher_mean}  std:{teacher_std}")
        #originalImage = test_dataset.getOriginalImage(test_idx)
        # originalImage = test_dataset.dataset.getImagePath(test_idx)
        originalImage = ""

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
            intlabel = gallery_labels[search_res_idx]["ClassId"]
            # intlabel = dataset.dataset.labels[search_res_idx]
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
            class_scores["data"][test_label["ClassName"]]["Predicted"].append(dataset.class_str_to_id[cosine_similarities_labels_str[0]])

            
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