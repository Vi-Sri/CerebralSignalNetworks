import math
from torchvision import transforms
from PIL import Image
import librosa
import numpy as np
import torch

def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = lr * args.learning_rate_weights
    optimizer.param_groups[1]['lr'] = lr * args.learning_rate_biases

class BarlowTransform:
    def __init__(self):
        self.img_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.eeg_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def transform_image(self, x):
        y = self.img_transform(x)
        return y
    
    def transform_signal(self, x):
        y = convertsignaltomelspectrogram(x, sr=256, n_mels=224, fmin=0.0, fmax=None, duration=10)
        y = self.eeg_transform(y)
        return y
    
def convertsignaltomelspectrogram(signal, sr, n_mels=224, fmin=0.0, fmax=None, duration=10):
    mel_spectrogram = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=n_mels, fmin=fmin, fmax=fmax)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    mel_spectrogram = mel_spectrogram[:, :int(duration * sr / 512)]
    mel_spectrogram = np.stack([mel_spectrogram, mel_spectrogram, mel_spectrogram], axis=-1)
    return mel_spectrogram # shape: (224, 224, 3)
