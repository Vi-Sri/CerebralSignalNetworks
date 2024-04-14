import os
import time
import torch
import argparse
import numpy as np
import json
import faiss
from utils.EEGDataset import EEGDataset
# from utils.Caltech101Dataset import Caltech101Dataset
# from utils.CustomModel import CustomModel
# import torchvision.transforms as transforms 

from sklearn.metrics.pairwise import cosine_similarity
from utils.DinoModel import DinoModel, dino_args
from utils.Utilities import initlogger

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    

def initDinoModel(model_to_load, FLAGS, checkpoint_key="teacher", use_only_backbone=True):
    dino_args.pretrained_weights = model_to_load
    dino_args.output_dir = FLAGS.log_dir
    dino_args.checkpoint_key = checkpoint_key
    dino_args.use_cuda = torch.cuda.is_available()
    dinov1_model = DinoModel(dino_args, use_only_backbone=use_only_backbone)
    dinov1_model.eval()
    return dinov1_model


if __name__=="__main__":


    parser = argparse.ArgumentParser('CNN Exercise.')
    parser.add_argument('--learning_rate',
                        type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=100,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size',
                        type=int, default=1,
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
                        default="./data/eeg/eeg_signals_raw_with_mean_std.pth",
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
                        default="./models/dino/localcrops_as_eeg/subject1/checkpoint.pth",
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
    # parser.add_argument('--pass_eeg_to_dino_for_gallery',
    #                     type=bool,
    #                     default=True,
    #                     help='if eeg to be passed to dino to get query or gallery')
    # parser.add_argument('--pass_eeg_to_dino_for_query',
    #                     type=bool,
    #                     default=True,
    #                     help='if eeg to be passed to dino to get query or gallery')

    
    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
    print(FLAGS)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    output_dir = f"{FLAGS.log_dir}/ST_{FLAGS.gallery_tranformation_type}_QT_{FLAGS.query_tranformation_type}/S_{FLAGS.gallery_subject}_Q_{FLAGS.query_gallery}"
    os.makedirs(output_dir,exist_ok=True)

    with open(f'{output_dir}/commandline_args.txt', 'w') as f:
        json.dump(FLAGS.__dict__, f, indent=2)

    # dino_args.pretrained_weights = FLAGS.custom_model_weights
    # dino_args.output_dir = FLAGS.log_dir
    # dino_args.checkpoint_key  ="teacher"
    # dino_args.use_cuda = torch.cuda.is_available()
    # dinov1_model = DinoModel(dino_args)
    # dinov1_model.eval()
    logger = initlogger(__name__)

    dinov1_model = None

    if FLAGS.gallery_tranformation_type=="img":
        # del dinov1_model
        dinov1_model = initDinoModel(model_to_load=FLAGS.dino_base_model_weights,FLAGS=FLAGS,checkpoint_key="teacher", use_only_backbone=False)
    else:
        dinov1_model = initDinoModel(model_to_load=FLAGS.custom_model_weights,FLAGS=FLAGS,checkpoint_key="teacher", use_only_backbone=False)

    SUBJECT = FLAGS.gallery_subject
    BATCH_SIZE = int(FLAGS.batch_size)
    learning_rate = FLAGS.learning_rate
    EPOCHS = FLAGS.num_epochs
    SaveModelOnEveryEPOCH = 100
    EEG_DATASET_PATH = FLAGS.eeg_dataset
    EEG_DATASET_SPLIT = FLAGS.eeg_dataset_split

    LSTM_INPUT_FEATURES = 2048 # should be image features output.
    LSTM_HIDDEN_SIZE = 460  # should be same as sequence length
    selectedDataset = "imagenet40"

    # Load dataset
    # dataset = EEGDataset(subset=FLAGS.search_gallery,eeg_signals_path=EEG_DATASET_PATH,eeg_splits_path=FLAGS.dataset_split, subject=SUBJECT,preprocessin_fn=weights.transforms())
    # test_dataset = EEGDataset(subset=FLAGS.query_gallery,eeg_signals_path=EEG_DATASET_PATH,eeg_splits_path=FLAGS.dataset_split, subject=SUBJECT,preprocessin_fn=weights.transforms())
    
    
    test_dataset = EEGDataset(subset=FLAGS.query_gallery,
                              eeg_signals_path=EEG_DATASET_PATH,
                              eeg_splits_path=EEG_DATASET_SPLIT,
                              subject=FLAGS.query_subject,
                              exclude_subjects=[],
                              preprocessin_fn=dinov1_model.dinov1_transform)
    
    
    dataset = EEGDataset(subset=FLAGS.search_gallery,
                         eeg_signals_path=EEG_DATASET_PATH,
                         eeg_splits_path=EEG_DATASET_SPLIT, 
                         subject=SUBJECT,
                         exclude_subjects=[],
                         preprocessin_fn=dinov1_model.dinov1_transform)
    

    time_t0 = time.perf_counter()

    ## Transform Test Data

    if FLAGS.gallery_tranformation_type=="img":
        """
        Transform images to images features, assuming dino model is baseline model and not EEG trained model
        """
        logger.info("Transforming image to dino base model image features for gallery")
        dataset.transformEEGDataDino(
                            model=dinov1_model,
                            preprocessor=dinov1_model.dinov1_transform,
                            pass_eeg=False,
                            device=device,
                            min_time=20,
                            max_time=490,
                            keep_features_flat=False)
    elif FLAGS.gallery_tranformation_type=="img2eeg":
        """
        Transform images to IMG2EEG features, assuming dino model is EEG finetuned model.
        """
        logger.info("Transforming image to EEG dino model image2eeg features for gallery")
        dataset.transformEEGDataDino(
                            model=dinov1_model,
                            preprocessor=dinov1_model.dinov1_transform,
                            pass_eeg=False,
                            device=device,
                            min_time=20,
                            max_time=490,
                            keep_features_flat=False)
    elif FLAGS.gallery_tranformation_type=="eeg2eeg":
        """
        Transform EEG to EEG2DinoPredEEG features, assuming dino model is EEG finetuned model.
        """
        logger.info("Transforming eeg to EEG dino model eeg2eeg features for gallery")
        dataset.transformEEGDataDino(
                            model=dinov1_model,
                            preprocessor=dinov1_model.dinov1_transform,
                            pass_eeg=True,
                            device=device,
                            min_time=20,
                            max_time=490,
                            keep_features_flat=False)
    elif FLAGS.gallery_tranformation_type=="eeg":
        logger.info("Keeping RAW EEG features for gallery")
        ## keep RAW eeg 
        pass


    del dinov1_model
    if FLAGS.query_tranformation_type=="img":
        dinov1_model = initDinoModel(model_to_load=FLAGS.dino_base_model_weights,FLAGS=FLAGS,checkpoint_key="teacher", use_only_backbone=False)
    else:
        dinov1_model = initDinoModel(model_to_load=FLAGS.custom_model_weights,FLAGS=FLAGS,checkpoint_key="teacher", use_only_backbone=False)

    if FLAGS.query_tranformation_type=="img":
        """
        Transform images to images features, assuming dino model is baseline model and not EEG trained model
        """
        logger.info("Transforming image to dino base model image features for gallery")
        test_dataset.transformEEGDataDino(
                            model=dinov1_model,
                            preprocessor=dinov1_model.dinov1_transform,
                            pass_eeg=False,
                            device=device,
                            min_time=20,
                            max_time=490,
                            keep_features_flat=False)
    elif FLAGS.query_tranformation_type=="img2eeg":
        """
        Transform images to IMG2EEG features, assuming dino model is EEG finetuned model.
        """
        logger.info("Transforming image to EEG dino model image2eeg features for gallery")
        test_dataset.transformEEGDataDino(
                            model=dinov1_model,
                            preprocessor=dinov1_model.dinov1_transform,
                            pass_eeg=False,
                            device=device,
                            min_time=20,
                            max_time=490,
                            keep_features_flat=False)
    elif FLAGS.query_tranformation_type=="eeg2eeg":
        """
        Transform EEG to EEG2DinoPredEEG features, assuming dino model is EEG finetuned model.
        """
        logger.info("Transforming eeg to EEG dino model eeg2eeg features for gallery")
        test_dataset.transformEEGDataDino(
                            model=dinov1_model,
                            preprocessor=dinov1_model.dinov1_transform,
                            pass_eeg=True,
                            device=device,
                            min_time=20,
                            max_time=490,
                            keep_features_flat=False)
    elif FLAGS.query_tranformation_type=="eeg":
        logger.info("Keeping RAW EEG features for gallery")
        ## keep RAW eeg 
        pass



    
    gallery_features = []
    query_features = []
    for i in range(len(dataset)):
        gallery_features.append(dataset.subsetData[i]["eeg"])
    for i in range(len(test_dataset)):
        query_features.append(test_dataset.subsetData[i]["eeg"])

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


