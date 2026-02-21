import torch
import torch.nn as nn
import torch.utils.data as data
import shutil
import fnmatch
import numpy as np
import cv2
import kornia
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torch.nn.functional as nn_F
from torchvision.utils import save_image
import random
from random import choice
import json
import pickle
import string
import glob
import logging
from PIL import Image, ImageFile, ImageChops, ImageOps
from functools import partial
import re
from collections import Counter
from scipy import ndimage
import pandas as pd
import os

from .config import DATASET_CONFIGS

import nltk
from nltk.corpus import stopwords

nltk.download('words')
from nltk.corpus import words
word_set = set(words.words())

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)

class SquarePad:
	def __call__(self, image):
		w, h = image.size
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, hp, vp)
		return F.pad(image, padding, 0, 'constant')
transform_orig_image = T.Compose([T.Resize((224, 224) , interpolation=Image.BICUBIC), T.ToTensor()])
#transform_orig_image = transforms.Compose([transforms.CenterCrop(224), transforms.Resize(224), transforms.ToTensor(),])

get_tensor = T.Compose([T.ToTensor()])

def build_transforms(img_size=224, is_train=True, normalize=True):

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if not is_train:
        transform = T.Compose([
            T.Resize((img_size, img_size) , interpolation=Image.BICUBIC),
            #T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    else:
        if normalize:
            transform = T.Compose([
                T.RandomResizedCrop(img_size, scale=(0.6, 1.0), interpolation=Image.BICUBIC),
                T.RandomHorizontalFlip(),
                #T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ])
        else:
            transform = T.Compose([
                T.RandomResizedCrop(img_size, scale=(0.6, 1.0), interpolation=Image.BICUBIC),
                T.RandomHorizontalFlip(),
                #T.ToTensor(),
            ])
            
    return transform


class RawImageDataset(data.Dataset): ###

    def __init__(self, src, load_subj_attr_data, exp_num, load_image=True):
        
        self.exp_num = exp_num
        self.load_image = load_image

        if load_subj_attr_data not in DATASET_CONFIGS:
            raise ValueError(f"Dataset '{load_subj_attr_data}' not found in DATASET_CONFIGS.")

        dataset_config = DATASET_CONFIGS[load_subj_attr_data]
        
        # Handle special case for UAF dataset labels based on experiment number
        if load_subj_attr_data == 'uaf_subj_attr' and exp_num not in ['0', '1']:
            labels = ['attractive', 'unattractive', 'happy', 'unhappy', 'trustworthy', 'untrustworthy']
        else:
            labels = dataset_config['labels']

        image_src_dir = f'{src}/{dataset_config["path"]}'
        
        data_path = f'{src}/original_scores/{dataset_config["scores_file"]}'

        with open(data_path, 'r') as file:
            all_scores_data = json.load(file)

        self.images_path = []
        self.attributes = []
        for score_entry in all_scores_data:
            image_name = score_entry['image_name']
            # UAF dataset includes .jpg, others do not. Standardize.
            if not str(image_name).endswith('.jpg'):
                image_name = f"{image_name}.jpg"
            
            image_path = os.path.join(image_src_dir, image_name)

            if os.path.exists(image_path):
                current_attributes = {}
                valid_entry = True
                for attribute in labels:
                    if attribute in score_entry:
                        score_data = score_entry[attribute]
                        # CFD/OMI scores are tuples (score, rank), UAF are just scores
                        current_attributes[attribute] = score_data[0] if isinstance(score_data, list) else score_data
                    else:
                        valid_entry = False
                        break
                
                if valid_entry:
                    self.images_path.append(image_path)
                    self.attributes.append(current_attributes)
                
    def absolute_file_paths(self, directory):
        path = os.path.abspath(directory)
        return [entry.path for entry in os.scandir(path) if entry.is_file()]      
            
    def get_attribute_paths(self, dict, index):
        return [dict[score][index] for score in self.score_changes]
    
    def load_images(self, paths):
        images = []
        for path in paths:
            images.append(get_tensor(Image.open(path).convert("RGB")))
        return images
    
    def __getitem__(self, index):
        if self.exp_num in ['0', '1']:
            image_name = self.images_path[index].split('/')[-1].split('.')[0]
            image_path = self.images_path[index]
            attributes = self.attributes[index]
            if self.load_image:
                image = Image.open(image_path).convert("RGB")
                return get_tensor(image), attributes, image_name, image_path
            else:
                return attributes, image_name, image_path
        elif self.exp_num == '2':
            attract_images = self.load_images(self.get_attribute_paths(self.images_path['Attractiveness'], index))
            dominance_images = self.load_images(self.get_attribute_paths(self.images_path['Dominance'], index))
            trust_images = self.load_images(self.get_attribute_paths(self.images_path['Trustworthiness'], index))
            return attract_images, dominance_images, trust_images

    def __len__(self):
        if self.exp_num in ['0', '1']:
            return len(self.images_path)
        elif self.exp_num == '2':
            return len(self.images_path['Attractiveness']['0.0'])

def get_loader(src, exp_num, batch_size, shuffle=True, num_workers=2, load_subj_attr_data=False, load_image=True, train=True):

    dataset = RawImageDataset(src, load_subj_attr_data, exp_num, load_image)
    sampler = None
    
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                batch_size=batch_size,
                                                sampler=sampler,
                                                shuffle=shuffle,
                                                num_workers=num_workers,
                                                pin_memory=True,
                                                collate_fn=None,
                                                drop_last=train,
                                                )
    return dataset, data_loader

def get_test_loader(src, exp_num, batch_size, workers, shuffle=False, split='test', load_subj_attr_data=False, load_image=True):
    test_dataset, test_loader = get_loader(src, exp_num, batch_size, shuffle, workers, load_subj_attr_data, load_image, train=False) 
    return test_dataset, test_loader

if __name__ == '__main__':

    pass
