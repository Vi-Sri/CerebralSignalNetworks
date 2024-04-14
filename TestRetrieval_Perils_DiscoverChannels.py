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


channelMap = Utilities_handler.read_channel_map(input_file="./channelmap.txt")

# eeg_signals_path="./data/eeg/eeg_signals_raw_with_mean_std.pth", 
# dataset = EEGDataset(subset="train",eeg_signals_path="./data/eeg/eeg_14_70_std.pth", eeg_splits_path="./data/eeg/block_splits_by_image_all.pth", subject=1,preprocessin_fn=None, time_low=20, time_high=480)
dataset = EEGDataset(subset="train",
                     eeg_signals_path="./data/eeg/eeg_signals_raw_with_mean_std.pth", 
                     eeg_splits_path="./data/eeg/block_splits_by_image_all.pth", 
                     subject=1,
                     preprocessin_fn=None, 
                     time_low=0, 
                     apply_norm_with_stds_and_means=True,
                     time_high=490)

test_dataset = EEGDataset(subset="test",
                     eeg_signals_path="./data/eeg/eeg_signals_raw_with_mean_std.pth", 
                     eeg_splits_path="./data/eeg/block_splits_by_image_all.pth", 
                     subject=1,
                     preprocessin_fn=None, 
                     time_low=0, 
                     apply_norm_with_stds_and_means=True,
                     time_high=490)

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
for data in dataset:
    eeg, label,image,i, image_features = data
    if not label["ClassId"] in label_wise_data:
        label_wise_data[label["ClassId"]] = {"images":[], "eeg":[], "strClass": []}
    label_wise_data[label["ClassId"]]["images"].append(image)
    label_wise_data[label["ClassId"]]["eeg"].append(eeg.numpy())
    label_wise_data[label["ClassId"]]["strClass"].append(label["ClassName"])

label_wise_test_data = {}
for data in test_dataset:
    eeg, label,image,i, image_features = data
    if not label["ClassId"] in label_wise_test_data:
        label_wise_test_data[label["ClassId"]] = {"images":[], "eeg":[], "strClass": []}
    label_wise_test_data[label["ClassId"]]["images"].append(image)
    label_wise_test_data[label["ClassId"]]["eeg"].append(eeg.numpy())
    label_wise_test_data[label["ClassId"]]["strClass"].append(label["ClassName"])


# SelectedChannels = [51,52,56,71]
# SelectedChannels = [51,52]
# SelectedChannels = [55]
# SelectedChannels = [71]
# SelectedChannels = [51,52,55,56,60,71,75,79, 106]
# SelectedChannels = [118]
# SelectedChannels = [117]
# SelectedChannels = [51,52,55,56,60,71,75,79,106]
# SelectedChannels = [2,6,7,10,22,26,28,30,34,33] # from [Reading into the mind’s eye: Boosting automatic visual recognition with EEG signals]
# SelectedChannels = [2,6,7,10,22,26,28,30,34,33,51,52,55,56,60,71,75,79,106]
SelectedChannels__ = [2,6,7,10,22,26,28,30,34,33,51,52,55,56,60,71,75,79,106]

TimeStart = 20
TimeEnd = 160

# plt.clf()
ClassesToAdd = [0,13,23,29]

ChannelWiseMetrics = {}



# MasterFixedChannels = [33,46,49,71,35,91,48,31,105,51,52,39,105] # manually added channel 71,51,52
# MasterFixedChannels = [71, 39, 35, 79] # manually added channel 
MasterFixedChannels = [28,30,91,107,1,5,99,3,94,78,4,119,47,108] # manually added channel
# MasterFixedChannels = [33] # AF3  #best ts 167_168  with Scores: {'Recall': 18.809, 'Precision': 3.9004999999999996}
# MasterFixedChannels = [34] # AF4 #best ts 346_347  with Scores: {'Recall': 17.884, 'Precision': 3.6717499999999994}
# MasterFixedChannels = [46]  #best ts 429_430  with Scores: {'Recall': 17.67625, 'Precision': 3.535}
# MasterFixedChannels = [71] #FTT9h #best ts 125_126  with Scores: {'Recall': 17.8555, 'Precision': 3.69625}
# MasterFixedChannels = [28,30] #O1, O2 #best ts 231_232  with Scores: {'Recall': 19.18775, 'Precision': 3.8870000000000005}
# MasterFixedChannels = [7,10] FC5,FC6 #best ts  63_64  with Scores: {'Recall': 18.389, 'Precision': 3.72275}
# MasterFixedChannels = [2,6] # F7,F8 #best ts 177_178  with Scores: {'Recall': 19.16225, 'Precision': 3.93225}
# MasterFixedChannels = [2]  # F7 # best ts 261_262  with Scores: {'Recall': 18.137, 'Precision': 3.8950000000000005}
# MasterFixedChannels = [6]  # F8 #best ts  192_193  with Scores: {'Recall': 16.830750000000002, 'Precision': 3.5072500000000004}
# MasterFixedChannels = [33,34] # AF3, AF4  #best ts 49_50  with Scores: {'Recall': 16.360500000000002, 'Precision': 3.401}
# MasterFixedChannels = [22,26] # P7, P8  #best ts 27_28  with Scores: {'Recall': 17.42575, 'Precision': 3.64525}
# MasterFixedChannels = [11,15] # T7, T8  #best ts 322_323  with Scores: {'Recall': 16.512999999999998, 'Precision': 3.3579999999999997}
# MasterFixedChannels = [28] # O1 #best ts 125_126  with Scores: {'Recall': 16.359750000000002, 'Precision': 3.3274999999999997}
# MasterFixedChannels = [30] # O2 #best ts 29_30  with Scores: {'Recall': 16.827499999999997, 'Precision': 3.41075}

# MasterFixedChannels = [28,30,33,34,2,6,46,71]
# MasterFixedChannels = [71]
# MasterFixedChannels = [2,6,7,10,22,26,28,30,34,33] 
# MasterFixedChannels = [2,6,7,10,22,26,28,30,34,33,51,52,55,56,60,71,75,79,106]

# MasterFixedChannels = [33,46,49,71,35,91,48,31,105,51,52,39,105]

ChanneWiseTS = {

    # "2": [261,262],
    # "6": [192,193],
    # "2_6": [177,178],
    # "28": [125,126],
    # "30": [29,30],
    # "28_30": [231,232],
    # "33": [167,168],
    # "34": [346,347],
    # "46": [429,430], 
    # "71": [125,126],

    "2": [250,270],
    "6": [180,200],
    "2_6": [180,200],
    "28": [130,150],
    "30": [20,40],
    "28_30": [240,260],
    "33": [170,190],
    "34": [350,370],
    "46": [440,460], 
    "71": [130,150],

}

MasterFixedChannels = [5]

print("Fixed Channels: ")
for MC in MasterFixedChannels:
    print(f"{channelMap[MC+1]}", end=",")
print()
# for iteration in range(0,480,1):

#     TimeStart = iteration
#     TimeEnd = TimeStart + 1


for i in range(128):

    for ch in range(128):

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
        for MC in MasterFixedChannels:
            print(f"{channelMap[MC+1]}", end=",")
        print()
        break
    # else:
    #     print(f"exiting since no best score is found")
#     break