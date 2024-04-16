import torch
import numpy as np
# from PIL import Image
# from torchvision import transforms
# from itertools import cycle
# import matplotlib.patches as mpatches 
# import random
import os

import matplotlib.pyplot as plt

# from sklearn.manifold import TSNE

from utils.Utilities import Utilities
from utils.PerilsEEGDataset import EEGDataset
import faiss 
import mne
from mne.preprocessing import ICA
Utilities_handler = Utilities()

from torch.utils.data import DataLoader
import json
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    

import time

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

# eeg_signals_path="./data/eeg/eeg_signals_raw_with_mean_std.pth", 
# dataset = EEGDataset(subset="train",eeg_signals_path="./data/eeg/eeg_14_70_std.pth", eeg_splits_path="./data/eeg/block_splits_by_image_all.pth", subject=1,preprocessin_fn=None, time_low=20, time_high=480)
dataset = EEGDataset(subset="train",
                         eeg_signals_path=EEG_DATASET_PATH,
                         eeg_splits_path=None, 
                         subject=SUBJECT,
                         time_low=0,
                         imagesRoot="./data/images/imageNet_images",
                         time_high=480,
                         exclude_subjects=[],
                         convert_image_to_tensor=True,
                         apply_channel_wise_norm=True,
                         preprocessin_fn=None)

generator1 = torch.Generator().manual_seed(123)
train_ds, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2], generator=generator1)
data_loader_train = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
data_loader_val = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

def addFeatures(label_wise_data, 
                label_index_To_Add, 
                SamplesToAdd=30, 
                TimeStart=20, 
                TimeEnd=460, 
                SelectedChannels=None,
                ChanneWiseTS=None):

    if SelectedChannels is None:
        SelectedChannels = [2,6,7,10,22,26,28,30,34,33,51,52,55,56,60,71,75,79,106]
    features = []
    labels = []
    str_labels = []

    if ChanneWiseTS is not None:
        TimeStart = 1
        TimeEnd = 2

    for i in range(SamplesToAdd):
        try:
            sample0 = label_wise_data[label_index_To_Add]["eeg"][i]
            # TimeSamplesDim = sample0[TimeStart:TimeEnd,:]
            fechannelsSample = np.zeros((TimeEnd-TimeStart, len(SelectedChannels)))
            # print(f"fechannelsSample {fechannelsSample.shape}")

            for i, ch in enumerate(SelectedChannels):
                # TimeStart, TimeEnd = ChanneWiseTS[str(ch)]
                # print(f"TimeStartEnd {TimeStart, TimeEnd}")
                fechannelsSample[:,i] = sample0[TimeStart:TimeEnd,ch]
            features.append(fechannelsSample)
            labels.append(label_index_To_Add)
            str_labels.append(label_wise_data[label_index_To_Add]["strClass"][i])
        except Exception as e:
            # print(e)
            pass 
            # print(f"Error : {e}")
    return features, labels, str_labels


label_wise_data = {}
label_wise_test_data = {}
for data in dataset:
    eeg, label,image,i, image_features = data
    if not label["ClassId"] in label_wise_data:
        label_wise_data[label["ClassId"]] = {"images":[], "eeg":[], "strClass": []}
    label_wise_data[label["ClassId"]]["images"].append(image)
    label_wise_data[label["ClassId"]]["eeg"].append(eeg.numpy())
    label_wise_data[label["ClassId"]]["strClass"].append(label["ClassName"])

for data in test_dataset:
    eeg, label,image,i, image_features = data
    if not label["ClassId"] in label_wise_test_data:
        label_wise_test_data[label["ClassId"]] = {"images":[], "eeg":[], "strClass": []}
    label_wise_test_data[label["ClassId"]]["images"].append(image)
    label_wise_test_data[label["ClassId"]]["eeg"].append(eeg.numpy())
    label_wise_test_data[label["ClassId"]]["strClass"].append(label["ClassName"])


SelectedChannels__ = []
TimeStart = 20
TimeEnd = 480
ChannelWiseMetrics = {}
MasterFixedChannels = []
for i in range(96):
    for ch in range(96):

        if ch in MasterFixedChannels:
            continue

        FixedChannels = []
        for master_ch in MasterFixedChannels:
            FixedChannels.append(master_ch)

        FixedChannels.append(ch)
        SelectedChannels = FixedChannels

        gallery_features = []
        labels = []
        str_labels = []

        query_features = []
        query_labels = []
        query_str_labels = []

        for i in range(40):
        # for i in ClassesToAdd:
            features__, labels__, str_labels__ = addFeatures(label_wise_data=label_wise_data, 
                                                            label_index_To_Add=i,
                                                            SamplesToAdd=30,
                                                            TimeStart=TimeStart,
                                                            TimeEnd=TimeEnd,
                                                            SelectedChannels=SelectedChannels,
                                                            ChanneWiseTS=None)
            gallery_features += features__
            labels += labels__
            str_labels += str_labels__
            # print(set(str_labels__))

            features__, labels__, str_labels__ = addFeatures(label_wise_data=label_wise_test_data, 
                                                            label_index_To_Add=i,
                                                            SamplesToAdd=30,
                                                            TimeStart=TimeStart,
                                                            TimeEnd=TimeEnd,
                                                            SelectedChannels=SelectedChannels,
                                                            ChanneWiseTS=None)
            
            query_features +=features__
            query_labels += labels__
            query_str_labels += str_labels__


        gallery_features = np.array(gallery_features)
        # print(gallery_features.shape)

        gallery_features = gallery_features.reshape(gallery_features.shape[0], -1)
        # print(gallery_features.shape)

        query_features = np.array(query_features)
        query_features = query_features.reshape(query_features.shape[0], -1)
        # print(query_features.shape)

        

        gallery_features = torch.from_numpy(np.array(gallery_features))
        query_features = torch.from_numpy(np.array(query_features))

        # gallery_features = gallery_features.reshape(gallery_features.size(0), -1)
        # query_features = query_features.reshape(query_features.size(0), -1)

        d = gallery_features.size(-1)    # dimension
        nb = gallery_features.size(0)    # database size
        nq = query_features.size(0)      # nb of queries

        index = faiss.IndexFlatL2(d)   # build the index
        # print(index.is_trained)
        index.add(gallery_features)    # add vectors to the index
        # print(index.ntotal)

        topK  =5
        k = 5                       # we want to see 4 nearest neighbors
        D, I = index.search(gallery_features[:5], k) # sanity check
        # print(I)
        # print(D)
        D, I = index.search(query_features, k)     # actual search
        # print(I[:5])                   # neighbors of the 5 first queries
        # print(I[-5:])                # neighbors of the 5 last queries

        class_scores = {"data" :{}, "metadata": {}}
        # class_scores["metadata"] = {"flags": FLAGS}
        print_done = False


        
        selectedDataset = "imagenet"
        output_dir = f"./output/Dataset_{selectedDataset}"
        os.makedirs(output_dir,exist_ok=True)

        time_t0 = time.perf_counter()

        for query_idx, search_res in enumerate(I):
            # print(search_res)
            labels = []
            test_intlabel = test_dataset.dataset.labels[query_idx]
            test_strlabel = test_dataset.dataset.class_id_to_str[test_intlabel]

            cosine_similarities = []
            cosine_similarities_labels_int = []
            cosine_similarities_labels_str = []
            cosine_similarities_labels_classid = []
            cosine_similarities_images = []

            test_intlabel = test_dataset.dataset.labels[query_idx]
            test_strlabel = test_dataset.dataset.class_id_to_str[test_intlabel]

            test_eeg, test_label, test_image, test_idx, img_f = test_dataset[query_idx]
            #originalImage = test_dataset.getOriginalImage(test_idx)
            originalImage = test_dataset.dataset.getImagePath(test_idx)

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
                class_scores["data"][test_label["ClassName"]]["Predicted"].append(test_dataset.dataset.class_str_to_id[cosine_similarities_labels_str[0]])

                
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
            # print(f"Class : {key} Recall: [{cls_data['Recall']}] Precision: [{cls_data['Precision']}]" )
            Recall_Total.append(cls_data["Recall"])
            Precision_Total.append(cls_data["Precision"])

        Recall_Total = np.array(Recall_Total).mean()
        Precision_Total = np.array(Precision_Total).mean()


        ch_key = ""
        for chk in SelectedChannels:
            # ch_key +=f"{TimeStart}_{TimeEnd}"
            ch_key +=f",{chk}"

        ChannelWiseMetrics[ch_key] = {"Recall": Recall_Total,"Precision": Precision_Total}
        # print(f"TS [{TimeStart}-{TimeEnd}][{ch}] Overall Recall :{Recall_Total} Overall Precision: {Precision_Total}")
        print(f"TS {MasterFixedChannels}[{ch}] Overall Recall :{Recall_Total} Overall Precision: {Precision_Total}")
        time_tn = time.perf_counter()
        # print(f"Completed in : {time_tn-time_t0:.2f}")

    maxRecall = 0
    bestScoreChannel = 0

    for key,val in ChannelWiseMetrics.items():
        if val["Recall"]>maxRecall:
            maxRecall = val["Recall"]
            bestScoreChannel = key

    print(f"best score channel: {bestScoreChannel}  with Scores: {ChannelWiseMetrics[bestScoreChannel]}")
    # print(f"best ts: [{TimeStart}-{TimeEnd}] with Scores: {ChannelWiseMetrics[bestScoreChannel]}")

    # if maxRecall>20.92:
    currentbestChannel = int(bestScoreChannel.split(",")[-1])
    if currentbestChannel not in MasterFixedChannels:
        MasterFixedChannels.append(currentbestChannel)
    else:
        print(f"found no channel better than last iteration. final channels: {MasterFixedChannels}")
        # for MC in MasterFixedChannels:
        #     print(f"{channelMap[MC+1]}", end=",")
        # print()
        break
    # else:
    #     print(f"exiting since no best score is found")
#     break