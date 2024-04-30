# CerebralSignalNetworks
An exploration of experiments based on EEG signals used for vision tasks - Medical Image Computing 


## LSTM DINO Kaggle Notebooks
```
https://www.kaggle.com/ajaymin28/eeg-lstm

```

## Train and Evaluate LSTM distill for DINOV2 Features
```
python LstmDistillFromDinoV2Train.py --eeg_dataset=$path --images_root=$path --num_epochs=50 --batch_size=16
python LstmDistillFromDinoV2Eval.py --eeg_dataset=$path --images_root=$path --custom_model_weights=$path
```

## Train LSTM DINO 
```
!python LstmDistillation.py --num_epochs=200 --epochs=200 --warmup_teacher_temp=-0.004 --batch_size=64 --batch_size_per_gpu=64 --log_dir=$path --eeg_dataset=$path  --eeg_dataset_split=$path  --images_root=$path   
```