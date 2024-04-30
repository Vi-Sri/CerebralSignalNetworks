import numpy as np
from sklearn.decomposition import FastICA
import numpy as np
from scipy import signal
import torch
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
import json
import faiss
import time

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def initlogger(name):
    return logging.getLogger(name=name)



def evaluate(FLAGS,gallery_features,query_features,gallery_labels,query_labels,dataset, ):

    time_t0 = time.perf_counter()

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
        # print(f"Class : {key} Recall: [{cls_data['Recall']}] Precision: [{cls_data['Precision']}]" )
        Recall_Total.append(cls_data["Recall"])
        Precision_Total.append(cls_data["Precision"])

    Recall_Total = np.array(Recall_Total).mean()
    Precision_Total = np.array(Precision_Total).mean()
    print(f"Overall Recall :{Recall_Total} Overall Precision: {Precision_Total}")
    

    time_tn = time.perf_counter()

    return Recall_Total, Precision_Total
    # outputPath = f"{output_dir}/Scores.pth"
    # class_scores["metadata"] = {"processing_time": f"{time_tn-time_t0:.2f}s"}
    
    # torch.save(class_scores, outputPath)

    # with open(f"{output_dir}/Scores.txt", 'w') as f:
        # json.dump(class_scores, f, indent=2, cls=NpEncoder)

    # pthFiles = [outputPath]
    # csv_file = open(f"{output_dir}/retreival_.csv", "w")
    # csv_file.write(f"srno, label, imagenet_label, Total class images,Total class image Retr, TP,Total Images Retr, Recall, Precision")
    # cnt = 1
    # for pth in pthFiles:
    #     class_metrics = torch.load(pth)
    #     filename = pth.split("train")[-1].split(".")[0]
    #     filename  = filename[1:]
    #     for key,val1 in class_metrics.items():
    #         if key=="data":
    #             val1 = dict(sorted(val1.items()))
    #             for classN, classData in val1.items():
    #                 TP  = classData['TP']
    #                 TotalClass = classData['TotalClass']
    #                 classIntanceRetrival = classData['classIntanceRetrival']
    #                 TotalRetrival = classData['TotalRetrival']
    #                 Recall = classData['Recall']
    #                 Precision = classData['Precision']
    #                 print(f"Class:{classN} TP: [{classData['TP']}] TotalClass: [{classData['TotalClass']}] classIntanceRetrival: [{classData['classIntanceRetrival']}] TotalRetrival: [{classData['TotalRetrival']}] ")
    #                 # csv_file.write(f"\n {cnt}, {filename}, {classN}, {TotalClass},{TotalRetrival},{TP},{classIntanceRetrival},{Recall},{Precision}")
    #                 cnt +=1
    # csv_file.write(f"\n\n,,,,,,,{Recall_Total},{Precision_Total}")                    
    # csv_file.close()
    
    print(f"Completed in : {time_tn-time_t0:.2f}")

class Utilities:
    def __init__(self) -> None:
        pass

    def read_channel_map(self, input_file):
        lines = []
        channel_map = {}
        with open(input_file) as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = line.strip()
                line = line.split("=")
                SciTerm = line[-1]
                ChanNum = line[0]
                ChanNum = ChanNum.split("-")
                ChanNum_int = int(ChanNum[-1])
                ChanNum = ChanNum[0]
                channel_map[ChanNum_int] = SciTerm
        return channel_map
    

    def load_data_label_wise(self, dataset, model,CustModel,device,process_data_with_model=False):
        label_wise_data = {}
        for data in dataset:
            eeg, label, image,i = data
            test_eeg = None

            imagePath = dataset.getImagePath(i)

            if process_data_with_model:
                with torch.no_grad():
                    features = model(image.unsqueeze(0).to(device))
                    features = features.view(-1, features.size(1))
                    outputs = CustModel(features.to(device))
                test_eeg = outputs.cpu().numpy() 

            if not label["ClassId"] in label_wise_data:
                label_wise_data[label["ClassId"]] = {"images":[], "eeg":[], "pred_eeg":[]}

            label_wise_data[label["ClassId"]]["images"].append(imagePath)
            label_wise_data[label["ClassId"]]["eeg"].append(eeg)
            label_wise_data[label["ClassId"]]["pred_eeg"].append(test_eeg)
        return label_wise_data
    

    def prepareEEGData(self, labelWiseData, convert_to_numpy=True, flatten_eeg=True, isModelPreprocessedData=False):
        eeg_features_ = []
        eeg_labels_ = []
        for label, labeData in labelWiseData.items():
            if isModelPreprocessedData:
                pred_eeg_fet = labeData["pred_eeg"]
            else:
                pred_eeg_fet = labeData["eeg"]
            for eeg in pred_eeg_fet:
                eeg_features_.append(eeg)
                eeg_labels_.append(label)
        if convert_to_numpy:
            eeg_features_  =np.array(eeg_features_, dtype=float)
        if flatten_eeg:
            eeg_features_ = eeg_features_.reshape(eeg_features_.shape[0], -1) 
        return eeg_features_, eeg_labels_
    

    def PrepapreDataforVis(self,dataset,convert_to_numpy=True):

        label_wise_data = {}
        img_f = None
        for data in dataset:
            eeg, label, image, i, img_f = data
            if not label["ClassId"] in label_wise_data:
                label_wise_data[label["ClassId"]] = {"images":[], "eeg":[]}

            label_wise_data[label["ClassId"]]["images"].append(image)
            label_wise_data[label["ClassId"]]["eeg"].append(eeg.cpu().numpy())
            # print(eeg.numpy().shape)

        # Exclude_labels = [0,22,15,14,16]
        Exclude_labels = []

        eeg_features = []
        eeg_labels = []
        eeg_dtype = None
        for label, labeData in label_wise_data.items():
            eeg_fet = labeData["eeg"]
            for eeg in eeg_fet:
                if eeg_dtype is None:
                    eeg_dtype = eeg.dtype
                eeg_features.append(eeg)
                eeg_labels.append(label)
        
        if convert_to_numpy:
            eeg_features  = np.array(eeg_features, dtype=float)

        return label_wise_data, eeg_features, eeg_labels, img_f
    
    def CalcMean(self,dataset,image_size=224):
        label_wise_data = {}
        for data in dataset:
            eeg, label, image, i, img_f = data
            if not label["ClassId"] in label_wise_data:
                label_wise_data[label["ClassId"]] = {"images":[]}
            label_wise_data[label["ClassId"]]["images"].append(image)
        
        label_wise_data_means = {}
        for label, data in tqdm(label_wise_data.items(), total=len(label_wise_data.keys())):
            if not label in label_wise_data_means:
                label_wise_data_means[label] = {"psum": torch.tensor([0.0, 0.0, 0.0]), "psum_sq": torch.tensor([0.0, 0.0, 0.0])}
            for image in data["images"]:
                image = image.unsqueeze(0)
                label_wise_data_means[label]["psum"] += image.sum(axis=[0, 2, 3])
                label_wise_data_means[label]["psum_sq"] += (image**2).sum(axis=[0, 2, 3])

        for label, psumval in label_wise_data_means.items():
            psum  = psumval["psum"]
            psum_sq  = psumval["psum_sq"]
            # pixel count
            count = len(label_wise_data[label]["images"]) * image_size * image_size
            # mean and std
            total_mean = psum / count
            total_var = (psum_sq / count) - (total_mean**2)
            total_std = torch.sqrt(total_var)
            # print(f"label : {label} mean {total_mean} std: {total_std} ")
            label_wise_data_means[label]["mean"] = total_mean.cpu().numpy()
            label_wise_data_means[label]["std"] = total_std.cpu().numpy()

        return label_wise_data, label_wise_data_means
    

    def CalcEEGMean(self,dataset,image_size=224):

        eeg_time = 0
        eeg_channels = 0
        
        label_wise_data = {}
        for data in dataset:
            eeg, label, image, i, img_f = data
            if not label["ClassId"] in label_wise_data:
                label_wise_data[label["ClassId"]] = {"images":[], "eegs": []}
            label_wise_data[label["ClassId"]]["images"].append(image)
            label_wise_data[label["ClassId"]]["eegs"].append(eeg)
        
        label_wise_data = dict(sorted(label_wise_data.items()))
        label_wise_data_means = {}
        for label, data in tqdm(label_wise_data.items(), total=len(label_wise_data.keys())):

            if not label in label_wise_data_means:
                label_wise_data_means[label] = {
                        "image":{
                            "psum": torch.tensor([0.0, 0.0, 0.0]), 
                            "psum_sq": torch.tensor([0.0, 0.0, 0.0])
                        },
                        "eeg":{
                            "psum": torch.zeros(128, dtype=float), 
                            "psum_sq":  torch.zeros(128, dtype=float)
                        }
                    }
            for image in data["images"]:
                image = image.unsqueeze(0)
                # print(image.size())
                label_wise_data_means[label]["image"]["psum"] += image.sum(axis=[0, 2, 3])  # batch, channel, hight, width
                label_wise_data_means[label]["image"]["psum_sq"] += (image**2).sum(axis=[0, 2, 3])

                # break

            for eeg in data["eegs"]:
                eeg = eeg.unsqueeze(0)
                # print(eeg.size())
                if eeg_time==0:
                    eeg_time =  eeg.size(1)
                    eeg_channels =  eeg.size(2)
                label_wise_data_means[label]["eeg"]["psum"] += eeg.sum(axis=[0, 1])  # batch, time, channel
                label_wise_data_means[label]["eeg"]["psum_sq"] += (eeg**2).sum(axis=[0, 1])
                # break

            # break


        for label, psumval in label_wise_data_means.items():
            psum  = psumval["image"]["psum"]
            psum_sq  = psumval["image"]["psum_sq"]
            # pixel count
            count = len(label_wise_data[label]["images"]) * image_size * image_size
            # mean and std
            total_mean = psum / count
            total_var = (psum_sq / count) - (total_mean**2)
            total_std = torch.sqrt(total_var)
            # print(f"label : {label} mean {total_mean} std: {total_std} ")
            label_wise_data_means[label]["image"]["mean"] = total_mean.cpu().numpy()
            label_wise_data_means[label]["image"]["std"] = total_std.cpu().numpy()

            eeg_psum  = psumval["eeg"]["psum"]
            eeg_psum_sq  = psumval["eeg"]["psum_sq"]
            # total eeg points count
            eeg_count = len(label_wise_data[label]["eegs"]) * eeg_time * eeg_channels
            # mean and std
            eeg_total_mean = eeg_psum / eeg_count
            eeg_total_var = (eeg_psum_sq / eeg_count) - (eeg_total_mean**2)
            eeg_total_std = torch.sqrt(eeg_total_var)
            # print(f"label : {label} mean {total_mean} std: {total_std} ")
            label_wise_data_means[label]["eeg"]["mean"] = eeg_total_mean.cpu().numpy()
            label_wise_data_means[label]["eeg"]["std"] = eeg_total_std.cpu().numpy()

        label_wise_data_means = dict(sorted(label_wise_data_means.items()))

        return label_wise_data, label_wise_data_means


    def remove_noise(self,eeg_data, sampling_rate):
        # Define filter parameters
        nyquist_freq = 0.5 * sampling_rate
        low_cutoff = 1.0  # Define your low cutoff frequency
        high_cutoff = 50.0  # Define your high cutoff frequency
        filter_order = 4  # Define your filter order

        # Calculate filter coefficients
        low = low_cutoff / nyquist_freq
        high = high_cutoff / nyquist_freq
        b, a = signal.butter(filter_order, [low, high], btype='band')

        # Apply the filter to each EEG channel
        filtered_data = np.zeros_like(eeg_data)
        for s in range(eeg_data.shape[0]):
            for i in range(eeg_data.shape[-1]):  # assuming eeg_data is (samples,time_points, channels)
                filtered_data[s,:, i] = signal.filtfilt(b, a, eeg_data[s,:, i])
        return filtered_data

    def remove_noise_with_ica(self,eeg_data,n_components=20):
        # Initialize ICA model
        
        # Reshape EEG data to have shape (samples, channels, time_points)
        num_samples, time_points, num_channels = eeg_data.shape

        denoised_zeros = np.zeros(shape=(num_samples, num_channels, n_components), dtype=float)
        
        for i in range(num_samples):
            ica = FastICA(n_components=n_components, random_state=42, max_iter=300)
            # print("fitting: ", eeg_data[i,:, :].shape)
            ica.fit(eeg_data[i,:, :].T)
            denoised_data = ica.transform(eeg_data[i,:, :].T)
            # print("transformed: ", denoised_data.shape)
            denoised_zeros[i,:,:] = denoised_data

        print(denoised_zeros.shape)

        return denoised_zeros
    

    def plotSampleEEGChannels(self, eeg_data, channels_to_plot, saveFigure=False, saveFigurename=None):

        if not saveFigure:
            plt.clf()

        plt.figure().set_size_inches(20,5)
        # for x in range(normalized_data.shape[-1]):

        for chn in channels_to_plot:
            plt.plot(eeg_data[0][:, chn], label=f'Channel :{chn}')

        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Filtering signals')
        plt.legend(ncol=10)
        plt.grid(True)
        if saveFigure:
            plt.savefig(saveFigurename)
        if not saveFigure:
            plt.show()
        plt.close()




# fig = plt.figure(figsize=(8, 6))
# fig.set_size_inches(11,11)
# ax = fig.add_subplot(111, projection="3d")
# fig.add_axes(ax)

# ax.set_title("EEG data")
# ax.view_init(azim=-30, elev=50)

# eeg_labels_np = np.array(eeg_labels)

# c_label = 30
# handles = []
# for chn_c in range(128):
 
#     # print(eeg_labels_np.shape)
#     filtered = np.where(eeg_labels_np == c_label)[0]
#     filtered_tsne = np.take(tsne_channel_wise[chn_c], filtered, 0)
#     # print(filtered_tsne.shape, chn_c)
#     _ = ax.text2D(0.8, 0.05, s=f"n_samples={filtered_tsne.shape[0]}", transform=ax.transAxes)
#     _patch = mpatches.Patch(color=channel_cmaps[chn_c], label=f'channel {chn_c}') 
#     handles.append(_patch)
#     ax.scatter(filtered_tsne[:,0], filtered_tsne[:,1],filtered_tsne[:,2], c=[channel_cmaps[chn_c] for i in range(filtered_tsne.shape[0])], s=50, alpha=0.8)
# plt.legend(handles=handles, loc="lower left", fontsize=10,ncol=4, bbox_to_anchor=(1.0, 0.0),fancybox=True) 