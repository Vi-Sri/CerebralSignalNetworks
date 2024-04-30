from utils.PerilsEEGDataset import EEGDataset
from utils.Utilities import Utilities
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
import json
from pathlib import Path

from dino.utils import trunc_normal_

class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cosine_similarity = nn.CosineSimilarity()

    def forward(self, student_outputs, teacher_outputs):
        loss = 1 - self.cosine_similarity(student_outputs, teacher_outputs).mean()
        return loss

class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx, output = 0, torch.empty(0).to(x[0].device)
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(x[start_idx: end_idx]))
            # The output is a tuple with XCiT model. See:
            # https://github.com/facebookresearch/xcit/blob/master/xcit.py#L404-L405
            if isinstance(_out, tuple):
                _out = _out[0]
            # accumulate outputs
            output = torch.cat((output, _out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        return self.head(output)

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

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
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(1)

        # total_loss = 0
        # loss = torch.sum(-teacher_out * F.log_softmax(student_out, dim=-1), dim=-1)
        # total_loss += loss.mean()

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms

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
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, student_outputs, teacher_outputs):
        student_mean, student_std = student_outputs.mean(), student_outputs.std()
        teacher_mean, teacher_std = teacher_outputs.mean(), teacher_outputs.std()
        mean_mse = self.mse(student_mean, teacher_mean)
        mean_std = self.mse(student_std, teacher_std)
        mse_loss = self.mse(student_outputs, teacher_outputs)
        return mean_std*0.4 + mean_mse*0.4 + mse_loss*0.2

class Parameters:
    ce_loss_weight = 0.95
    mse_loss_weight = 0.20
    soft_target_loss_weight = 0.05
    alpha = 0.5
    temperature = 5

def initDinoV2Model(model= "dinov2_vits14"):
    dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", model)
    return dinov2_vits14

if __name__=="__main__":

    parser = argparse.ArgumentParser('CNN Exercise.')

    # parser.add_argument('--arch', default='vit_small', type=str,
    #     choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'] \
    #             + torchvision_archs + torch.hub.list("facebookresearch/xcit:main"),
    #     help="""Name of architecture to train. For quick experiments with ViTs,
    #     we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=8, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=384, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=8, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=200, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-06, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=4, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizi""")

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
    parser.add_argument('--hyperprams',
                        type=str,
                        default="{'ce_loss_weight': 0.50, 'soft_target_loss_weight':0.50,'alpha': 1,'temperature':2}",
                        help='hyper params for training model, pass dict tpye in string format')
    parser.add_argument('--seed', default=43, type=int, help='Random seed.')
    parser.add_argument('--output_dir', default="./output", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=10, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of data loading workers per GPU.')
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
    EEG_DATASET_SPLIT = FLAGS.eeg_dataset_split

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

    dataset = EEGDataset(eeg_signals_path=EEG_DATASET_PATH, 
                        eeg_splits_path=EEG_DATASET_SPLIT,
                        preprocessin_fn=transform_image, 
                        time_low=0, 
                        time_high=495,
                        imagesRoot=FLAGS.images_root,
                        data_augment_eeg=False)

    
    eeg, label,image,i, image_features = dataset[0]
    print(eeg.shape)
    # Utilities_handler.plotSampleEEGChannels(eeg_data=[eeg], channels_to_plot=[0])
    # Utilities_handler.plotSampleEEGChannels(eeg_data=[eeg], channels_to_plot=[26])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # dinov2_model = initDinoV2Model(model="dinov2_vits14").to(device)
    # dinov2_model = dinov2_model.eval()
    # dinov2_model.to(device)

    # data_loader_train = DataLoader(dataset, batch_size=FLAGS.batch_size, shuffle=True)
    # dataset.extract_features(model=dinov2_model, data_loader=data_loader_train, replace_eeg=False)

    generator1 = torch.Generator().manual_seed(43)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [0.8, 0.2], generator=generator1)

    sampler = torch.utils.data.DistributedSampler(train_ds, shuffle=True)
    data_loader_train = torch.utils.data.DataLoader(
        train_ds,
        sampler=sampler,
        batch_size=FLAGS.batch_size_per_gpu,
        num_workers=FLAGS.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(train_ds)} images.")

    # data_loader_train = DataLoader(train_ds, batch_size=FLAGS.batch_size, shuffle=True)
    # data_loader_val = DataLoader(val_ds, batch_size=FLAGS.batch_size, shuffle=True)
    eeg, label,image,i, image_features = next(iter(data_loader_train)) 
    # # print(eeg.shape, "eeg shape")
    # outs = dinov2_model(image.to(device))
    # features_length = outs.size(-1)
    # print(outs.size())
    embed_dim = 128
    # lstm_embedding_dim = 128

    teacher = Model(input_size=96,lstm_size=embed_dim,lstm_layers=4,output_size=embed_dim, include_top=False)
    student = Model(input_size=96,lstm_size=embed_dim,lstm_layers=4,output_size=embed_dim, include_top=False)

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(student, DINOHead(
        embed_dim,
        FLAGS.out_dim,
        use_bn=FLAGS.use_bn_in_head,
        norm_last_layer=FLAGS.norm_last_layer,
    ))
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, FLAGS.out_dim, FLAGS.use_bn_in_head),
    )

    student, teacher = student.cuda(), teacher.cuda()


    student = nn.parallel.DistributedDataParallel(student, device_ids=[FLAGS.gpu], find_unused_parameters=True)
    teacher.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both LSTM network.")

    # print(eeg[0].size(), "eeg sliced")
    lstmout = student(eeg.to(device))
    # print(lstmout.size())

    params_groups = utils.get_params_groups(student)
    
    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        FLAGS.out_dim,
        FLAGS.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        FLAGS.warmup_teacher_temp,
        FLAGS.teacher_temp,
        FLAGS.warmup_teacher_temp_epochs,
        FLAGS.epochs,
    ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if FLAGS.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif FLAGS.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif FLAGS.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    
    # for mixed precision training
    fp16_scaler = None
    if FLAGS.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        FLAGS.lr * (FLAGS.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        FLAGS.min_lr,
        FLAGS.epochs, len(data_loader_train),
        warmup_epochs=FLAGS.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        FLAGS.weight_decay,
        FLAGS.weight_decay_end,
        FLAGS.epochs, len(data_loader_train),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(FLAGS.momentum_teacher, 1,
                                               EPOCHS, len(data_loader_train))
    print(f"Loss, optimizer and schedulers ready.")


    # warmup_teacher_temp = 0.05
    # teacher_temp = 0.05
    # warmup_teacher_temp_epochs = 5

    # dino_loss = DINOLoss(
    #     out_dim=features_length,
    #     ncrops=4,
    #     warmup_teacher_temp=warmup_teacher_temp,
    #     teacher_temp=teacher_temp,
    #     warmup_teacher_temp_epochs=warmup_teacher_temp_epochs,
    #     nepochs=EPOCHS,
    # ).cuda()

    epoch_losses = []
    val_epoch_losses = []
    best_val_loss = None


    GlobalLength = 300
    LocalLength = 200
    GlobalViews = 2
    LocalViews = 4



    timeWindow = None
    for EPOCH in range(EPOCHS):

        batch_losses = []
        val_batch_losses = []

        student.train()

        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Epoch: [{}/{}]'.format(EPOCH, EPOCHS)

        # for data in data_loader_train:
        for it, (eeg, label, images, i, image_features) in enumerate(metric_logger.log_every(data_loader_train, 10, header)):
            # eeg, label,image,i, image_features = data
            # image_features = torch.from_numpy(np.array(image_features)).to(device)
            it = len(data_loader_train) * EPOCH + it  # global training iteration
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_schedule[it]
                if i == 0:  # only the first group is regularized
                    param_group["weight_decay"] = wd_schedule[it]

            # print("eeg size", eeg.size())

            if timeWindow is None:
                timeWindow = eeg.size(1)

            globalViews = []
            localViews = []
            for i in range(GlobalViews):
                start_index = np.random.randint(0, eeg.size(1))
                end_index = start_index + GlobalLength
                if end_index>timeWindow:
                    remain = end_index - timeWindow
                    start_index -= remain
                    end_index = start_index + GlobalLength
                globalViews.append(eeg[:,start_index:end_index,:].cuda(non_blocking=True))

            for i in range(LocalViews):
                start_index = np.random.randint(0, eeg.size(1))
                end_index = start_index + LocalLength
                if end_index>timeWindow:
                    remain = end_index - timeWindow
                    start_index -= remain
                    end_index = start_index + LocalLength
                localViews.append(eeg[:,start_index:end_index,:].cuda(non_blocking=True))


            # move images to gpu
            # print(eeg.shape, "egg shape")  [batch, augs, time, chan]
            # eegs = [e.cuda(non_blocking=True) for e in eeg]
            # eeg = eeg.cuda(non_blocking=True)
            # teacher and student forward passes + compute dino loss
            with torch.cuda.amp.autocast(fp16_scaler is not None):

                student_outputs = []
                teacher_outputs = []
                for gView in globalViews:
                    teacher_output = teacher(gView)  # only the 2 global views pass through the teacher
                    teacher_outputs.append(teacher_output)
                    student_output = student(gView)
                    student_outputs.append(student_output)

                for lView in localViews:
                    student_output = student(lView)
                    student_outputs.append(student_output)
                
                student_outputs = torch.stack(student_outputs, dim=0)
                teacher_outputs = torch.stack(teacher_outputs, dim=0)
                loss = dino_loss(student_outputs, teacher_outputs, EPOCH)

            # student update
            optimizer.zero_grad()
            param_norms = None
            if fp16_scaler is None:
                loss.backward()
                if FLAGS.clip_grad:
                    param_norms = utils.clip_gradients(student, FLAGS.clip_grad)
                utils.cancel_gradients_last_layer(EPOCH, student,
                                                FLAGS.freeze_last_layer)
                optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                if FLAGS.clip_grad:
                    fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                    param_norms = utils.clip_gradients(student, FLAGS.clip_grad)
                utils.cancel_gradients_last_layer(EPOCH, student,
                                                FLAGS.freeze_last_layer)
                fp16_scaler.step(optimizer)
                fp16_scaler.update()

            # EMA update for the teacher
            with torch.no_grad():
                m = momentum_schedule[it]  # momentum parameter
                for param_q, param_k in zip(student.module.parameters(), teacher.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            # logging
            torch.cuda.synchronize()

            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)


        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': EPOCH + 1,
            'args': FLAGS,
            'dino_loss': dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(FLAGS.log_dir, 'checkpoint.pth'))
        if FLAGS.saveckp_freq and EPOCH % FLAGS.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(FLAGS.log_dir, f'checkpoint{EPOCH:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': EPOCH}
        if utils.is_main_process():
            with (Path(FLAGS.log_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # opt.zero_grad()
            # lstm_output = LSTM_model(eeg.to(device))

            # loss = dino_loss(lstm_output,image_features, EPOCH)
            # # loss = criterion(lstm_output, image_features)
            # batch_losses.append(loss.cpu().item())

            # loss.backward()
            # opt.step()

        # batch_losses = np.array(batch_losses)
        # epoch_loss = batch_losses.mean()
        # epoch_losses.append(epoch_loss)
        
        # if EPOCH % validation_frequency==0:
        #     LSTM_model.eval()

        #     for data in data_loader_val:
        #         eeg, label,image,i, image_features = data

        #         with torch.no_grad():
        #             image_features = torch.from_numpy(np.array(image_features)).to(device)
        #             lstm_output = LSTM_model(eeg.to(device))

        #             # print(f"lstm features mean: {torch.mean(lstm_output)} std: {torch.std(lstm_output)} dino mean: {torch.mean(image_features)} std: {torch.std(image_features)}")


        #             # loss = criterion(lstm_output,image_features)
        #             loss = dino_loss(lstm_output,image_features, EPOCH)
        #             loss = loss.cpu().item()
        #             val_batch_losses.append(loss)

        #             if best_val_loss is None:
        #                 best_val_loss = loss
        #             else:
        #                 if loss<best_val_loss:
        #                     best_val_loss = loss
        #                     torch.save(LSTM_model.state_dict(), f"{FLAGS.log_dir}/lstm_dinov2_best_loss.pth")
            
            
        #     val_batch_losses = np.array(val_batch_losses)
        #     val_epoch_loss= val_batch_losses.mean()
        #     val_epoch_losses.append(val_epoch_loss)
        #     print(f"EPOCH {EPOCH} train_loss: {round(epoch_loss,6)} val_loss: {round(val_epoch_loss,6)}")
        # else:
        #     print(f"EPOCH {EPOCH} train_loss: {round(epoch_loss,6)}")