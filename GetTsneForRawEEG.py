import torch
import numpy as np
import matplotlib
import matplotlib.patches as mpatches 
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from utils.Utilities import Utilities
# from utils.PerilsEEGDataset import EEGDataset
from utils.EEGDataset import EEGDataset
# from utils.Caltech101Dataset import Caltech101Dataset
from utils.DinoModel import DinoModel, dino_args

Utilities_handler = Utilities()


search_gallary = "train"
query_gallary = "test"
SUBJECT = 1
EEG_DATASET_PATH = "./data/eeg/eeg_signals_raw_with_mean_std.pth"
dataset_split = "./data/eeg/block_splits_by_image_all.pth"

def prepareEEGData(labelWiseData, convert_to_numpy=True, flatten_eeg=True):
    eeg_features_ = []
    eeg_labels_ = []
    for label, labeData in labelWiseData.items():
        pred_eeg_fet = labeData["eeg"]
        for idx,eeg in enumerate(pred_eeg_fet):
            # print(pred_eeg_fet[idx].shape)
            eeg_features_.append(pred_eeg_fet[idx])
            eeg_labels_.append(label)
    if convert_to_numpy:
        eeg_features_  = np.array(eeg_features_, dtype=float)
    if flatten_eeg:
        eeg_features_ = eeg_features_.reshape(eeg_features_.shape[0], -1) 
    return eeg_features_, eeg_labels_


for subject in range(1,7):

    SUBJECT = subject

    train_dataset = EEGDataset(subset=search_gallary,
                            eeg_signals_path=EEG_DATASET_PATH,
                            eeg_splits_path=dataset_split, 
                            subject=SUBJECT,
                            time_low=20,
                            time_high=480,
                            exclude_subjects=[],
                            apply_norm_with_stds_and_means=False,
                            apply_channel_wise_norm=False,
                            preprocessin_fn=None)

    label_wise_data_test, eeg_features_test, eeg_labels_test, img_feat_test = Utilities_handler.PrepapreDataforVis(train_dataset)
    img_eeg_pred_features_test,eeg_labels_test = prepareEEGData(label_wise_data_test, convert_to_numpy=True, flatten_eeg=True)

    X_tsne_RAW_EEG = TSNE(n_components=3,perplexity=40, init="pca", learning_rate=0.1, n_iter=1000).fit_transform(img_eeg_pred_features_test)

    # print("after tsne")

    # plt.figure().set_size_inches(20,10)
    # plt.clf()

    cmap = matplotlib.colormaps["tab20c"]

    cmap = plt.cm.get_cmap("tab20c", len(label_wise_data_test.keys()))
    
    # cmap_pred = plt.cm.get_cmap("tab20c", len(label_wise_data_test.keys()))

    # print("after cmaps")

    gen_colors = []
    handles = []
    cmaps = []

    for eeg_label in list(label_wise_data_test.keys()):
        cmaps.append(cmap(eeg_label))
        _patch = mpatches.Patch(color=cmap(eeg_label), label=f'Class {eeg_label}') 
        handles.append(_patch)

    # print("after patches")

    for i in range(X_tsne_RAW_EEG.shape[0]):
        colorMap = cmaps[eeg_labels_test[i]]
        gen_colors.append(colorMap)


    # plt.clf()

    ELEVATION = 40
    VIEW_ANGLE = 50

    # Create a new figure
    fig = plt.figure(figsize=(20, 15))

    # print("after setting figure size")

    # Add the first subplot
    ax1 = fig.add_subplot(111, projection='3d') # '121' means 1 row, 2 columns, and use the first cell
    ax1.set_title(f"EEG Subject {SUBJECT} RAW EEG")
    # ax.view_init(azim=90, elev=1)
    ax1.view_init(azim=VIEW_ANGLE, elev=ELEVATION)

    # print("after view init")
    _ = ax1.text2D(0.8, 0.05, s=f"n_samples={X_tsne_RAW_EEG.shape[0]}", transform=ax1.transAxes)

    ax1.scatter(X_tsne_RAW_EEG[:,0], X_tsne_RAW_EEG[:,1], X_tsne_RAW_EEG[:,2], c=gen_colors, s=10, alpha=0.8)
    ax1.legend(handles=handles, loc="best", fontsize=13,fancybox=True,ncol=5)

    # Show the figure
    plt.savefig(f"./output/tsne/SUB_{SUBJECT}_RAW_EEG_features_distribution.png", bbox_inches='tight', pad_inches=0)

    # plt.show()
