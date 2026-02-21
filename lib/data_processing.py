import os
import gc
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from lib import image_caption
from tqdm import tqdm
from PIL import Image
import pickle
import re
import json
import shutil
import argparse
import cv2
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from hycoclip.hycoclip import lorentz as L
from lib import image_caption

import mediapipe as mp

from sklearn.manifold import TSNE
import umap

#####
face_feats = ['face shape', 'forehead', 'eyebrows', 'eyes', 'nose', 'mouth', 'cheeks', 'chin']
shape_attributes = ['oval face', 'round face', 'square face', 'face height', 'face width', 
                    'straight eyebrows', 'angled eyebrows', 'rounded eyebrows', 'thin eyebrows', 'bushy eyebrows',
                    'eye height', 'eye width', 'open eyes', 'closed eyes', 'narrow eyes', 'wide eyes', 'round eyes', 'bags under eyes',
                    'nose length', 'nose width', 'pointy nose', 'round nose', 'big nose',
                    'open mouth', 'closed mouth', 'mouth slightly open', 'round lips', 'narrow lips', 'wide lips', 'big lips',
                    'high cheekbones', 'jawline structure', 'cheekbone prominence',
                    'short chin', 'long chin', 'double chin', 'pointed chin', 'square chin']
skin_attributes = ['fair skin tone', 'medium skin tone', 'dark skin tone', 'pale skin tone', 'skin texture', 'wrinkles', 'acne', 'scars', 'freckles']
hair_attributes = ['bald', 'black hair', 'brown hair', 'blonde hair', 'gray hair', 'straight hair', 'wavy hair', 'beard', 'no beard', 'mustache', 'receding hairline', 'bangs']
age = ['young', 'middle-aged', 'old']
gender = ['female', 'male']
accessories = ['glasses', 'jewelry']
other = ['chubby', 'smiling', 'straight head pose', 'left head pose', 'right head pose', 'symmetry']
ALL_ATTRIBUTES = face_feats + shape_attributes + skin_attributes + hair_attributes + age + gender + other
#####

#####
# https://storage.googleapis.com/mediapipe-assets/documentation/mediapipe_face_landmark_fullsize.png
LEFT_EYE_LANDMARKS = [463, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382, 362]  # Left eye landmarks
RIGHT_EYE_LANDMARKS = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]  # Right eye landmarks
NOSE_LANDMARKS = [168, 193, 122, 196, 3, 236, 198, 131, 49, 48, 219, 59, 218, 237, 44, 19, 274, 457, 438, 289, 439, 278, 279, 360, 420, 456, 248, 419, 351, 417]  # Nose landmarks
MOUTH_LANDMARKS = [0, 267, 269, 270, 409, 306, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39, 37]  # Mouth landmarks
LEFT_EYEBROWS_LANDMARKS = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
RIGHT_EYEBROWS_LANDMARKS = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
#####

######
# Hyperbolic law of cosines: https://math.stackexchange.com/questions/2530948/angle-between-points-in-hyperbolic-space        
def angle_from_sides(a, b, c):
    return torch.acos(((torch.cosh(a) * torch.cosh(b)) - torch.cosh(c)) / (torch.sinh(a) * torch.sinh(b)))
    
def calculate_interior_angles(p1, p2, p3, _curv):
    a = -L.pairwise_dist(p2, p3, _curv)
    b = -L.pairwise_dist(p1, p3, _curv)
    c = -L.pairwise_dist(p1, p2, _curv)
    
    alpha = angle_from_sides(b, c, a)
    beta = angle_from_sides(a, c, b)
    gamma = angle_from_sides(a, b, c)
    
    return alpha, beta, gamma
######

def project_to_poincare_disk(hyperbolic_points):
    # Implement your projection logic (e.g., stereographic projection) here
    # This is a placeholder, use a suitable projection for your embedding space.
    # Example: assume points already in Poincare disk, just limit radius
    radius = torch.clamp(hyperbolic_points.norm(dim=-1, keepdim=True), max=0.99)
    return hyperbolic_points/radius

def tsne_plot(pretrained_model, emb_main, other_embs, labels, available_attributes, save_path):
    os.makedirs(save_path, exist_ok=True)
    
    combined = torch.cat((emb_main, other_embs), 0)
    #tsne = TSNE(n_components=2, random_state=0, perplexity=min(30, other_embs.size(0)))
    #reduced_embeddings = tsne.fit_transform(combined.detach().cpu().numpy())
    
    reduced_embeddings = umap.UMAP(n_components=2, random_state=0).fit_transform(combined.detach().cpu().numpy())
    
    target_reduced = reduced_embeddings[:7]
    other_reduced = reduced_embeddings[7:]
    
    plt.figure(figsize=(18, 12))
    plt.scatter(other_reduced[:, 0], other_reduced[:, 1], label='Objective Face Attributes', alpha=0.7)
    plt.scatter(target_reduced[0, 0], target_reduced[0, 1], color='red', label=labels[0], s=100)
    if len(labels) > 1:
        plt.scatter(target_reduced[1, 0], target_reduced[1, 1], color='green', label=labels[1], s=100)
        plt.scatter(target_reduced[2, 0], target_reduced[2, 1], color='blue', label=labels[2], s=100)
        plt.scatter(target_reduced[3, 0], target_reduced[3, 1], color='yellow', label=labels[3], s=100)
        plt.scatter(target_reduced[4, 0], target_reduced[4, 1], color='orange', label=labels[4], s=100)
        plt.scatter(target_reduced[5, 0], target_reduced[5, 1], color='purple', label=labels[5], s=100)
        plt.scatter(target_reduced[6, 0], target_reduced[6, 1], color='cyan', label=labels[6], s=100)
    
    for i, txt in enumerate(range(1, len(other_reduced) + 1)):
        plt.annotate(txt, (other_reduced[i, 0], other_reduced[i, 1]), textcoords="offset points", xytext=(2,0), ha='left')
    
    table_data = []
    for i in range(0, len(available_attributes), 2):
        val = [f'{i+j+1}. {item}' for j, item in enumerate(available_attributes[i:i + 2])]
        table_data.append(val)
    plt.table(cellText=table_data, loc='right', cellLoc='left', bbox=[1.05, 0, 0.6, 1])
    
    plt.subplots_adjust(right=0.65)
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.title(f'Embedding Visualization: {pretrained_model}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{save_path}/{pretrained_model}__relevant_objective_face_feat_viz.jpg')
    plt.clf()

def preprocess_text(tokenizer, text):
    text_proc = tokenizer.basic_tokenizer.tokenize(text)
    # Convert caption (string) to word ids (with Size Augmentation at training time).
    return image_caption.process_caption_bert(tokenizer, text_proc, False, size_augment=1)

preprocess_image = image_caption.build_transforms(img_size=224, is_train=False)
get_tensor = transforms.Compose([transforms.ToTensor()])

exp_clip_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
def exp_clip_sim(model, image, caption_concept_list):
    logit_scale, image_features, text_features = model(image=exp_clip_transform(image).unsqueeze(0).cuda(), text=caption_concept_list, mode_task="Static_FER")
    similarity = (logit_scale * torch.nn.functional.cosine_similarity(text_features, image_features)).tolist()
    return similarity

def exp_num_2_clip(tokenizer, processor, model, text, images):
    inputs = tokenizer(text, padding=True, return_tensors="pt")
    text_features = model.get_text_features(**inputs)
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    image_features = clip_image_preprocess(images, processor, model)
    return torch.nn.functional.cosine_similarity(text_features, image_features).tolist()

def exp_num_2_hycoclip(tokenizer, model, image_transform_2, text, images):
    images = [im.squeeze(0) for im in images]
    image_features = model.encode_image(image_transform_2(torch.stack(images, 0)).to(model.device), project=True)
    text_tokens = tokenizer(text)
    text_features = model.encode_text(text_tokens, project=True)
    return L.pairwise_inner(text_features, image_features, model.curv.exp()).tolist()[0]

def clip_image_preprocess(image, processor, model):
    inputs = processor(images=image, return_tensors="pt")
    image_features = model.get_image_features(**inputs)
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    return image_features

def metaclip_image_preprocess(image, preprocess, model):
    image = preprocess(image).unsqueeze(0)
    image_features = model.encode_image(image)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features

def visualize_attention_region(visualize, visualization_path, image_name, image, caption_concept_list, preprocess, model, device, mask_name=None):
    if mask_name is not None:
        save_path = f'{visualization_path}/{image_name}/{mask_name}'
    else:
        save_path = f'{visualization_path}/{image_name}'
    os.makedirs(save_path, exist_ok=True)
    image = preprocess(image).unsqueeze(0).to(device)
    logits = model(image, caption_concept_list)
    logits = torch.where(logits > 0.8, logits, torch.tensor(0, dtype=logits.dtype))
    visualize(image, caption_concept_list, logits, save_path=save_path)

def get_coordinates(region_landmarks, width, height):
    coordinates = []
    for landmark in region_landmarks:
        x, y = int(landmark.x * height), int(landmark.y * width)
        coordinates.append((x, y))
    return coordinates

def apply_eye_mask(pil_image, detector, apply_mask):
    np_image = np.array(pil_image)
    width, height = pil_image.size
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(pil_image))
    detection_result = detector.detect(image)
    face_landmarks_list = detection_result.face_landmarks
    
    if face_landmarks_list:
        for mask in apply_mask:
            if mask == 'eyes':
                eye_landmarks_left = [face_landmarks_list[0][idx] for idx in LEFT_EYE_LANDMARKS]
                eye_landmarks_right = [face_landmarks_list[0][idx] for idx in RIGHT_EYE_LANDMARKS]
                left_eye_coordinates = get_coordinates(eye_landmarks_left, width, height)
                right_eye_coordinates = get_coordinates(eye_landmarks_right, width, height)
                left_eye = np.array(left_eye_coordinates, dtype=np.int32)
                right_eye = np.array(right_eye_coordinates, dtype=np.int32)
                cv2.fillPoly(np_image, [left_eye, right_eye], color=(0, 0, 0))
            elif mask == 'nose':
                nose_landmarks = [face_landmarks_list[0][idx] for idx in NOSE_LANDMARKS]
                nose_coordinates = get_coordinates(nose_landmarks, width, height)
                nose = np.array(nose_coordinates, dtype=np.int32)
                cv2.fillPoly(np_image, [nose], color=(0, 0, 0))
            elif mask == 'mouth':
                mouth_landmarks = [face_landmarks_list[0][idx] for idx in MOUTH_LANDMARKS]
                mouth_coordinates = get_coordinates(mouth_landmarks, width, height)
                mouth = np.array(mouth_coordinates, dtype=np.int32)
                cv2.fillPoly(np_image, [mouth], color=(0, 0, 0))
            elif mask == 'eyebrows':
                eyebrows_landmarks_left = [face_landmarks_list[0][idx] for idx in LEFT_EYEBROWS_LANDMARKS]
                eyebrows_landmarks_right = [face_landmarks_list[0][idx] for idx in RIGHT_EYEBROWS_LANDMARKS]
                left_eyebrows_coordinates = get_coordinates(eyebrows_landmarks_left, width, height)
                right_eyebrows_coordinates = get_coordinates(eyebrows_landmarks_right, width, height)
                left_eyebrows = np.array(left_eyebrows_coordinates, dtype=np.int32)
                right_eyebrows = np.array(right_eyebrows_coordinates, dtype=np.int32)
                cv2.fillPoly(np_image, [left_eyebrows, right_eyebrows], color=(0, 0, 0))
        return Image.fromarray(np_image)
    else:
        return None
    
def apply_eye_mask__(pil_image, mtcnn, fa, apply_mask):
    np_image = np.array(pil_image)
    boxes, _ = mtcnn.detect(np_image) # detect faces
    
    if boxes is not None:
        # Detect landmarks
        x1, y1, x2, y2 = map(int, boxes[0])
        face_img = np_image[y1:y2, x1:x2]
        
        h, w, _ = face_img.shape
        if h == 0 or w == 0:
            return None
        
        preds = fa.get_landmarks(face_img)[0]
        
        # https://learnopencv.com/using-facial-landmarks-for-overlaying-faces-with-masks/
        if apply_mask == 'eyes':
            left_eye = preds[[36, 37, 38, 39, 40, 41]]
            right_eye = preds[[42, 43, 44, 45, 46, 47]]
            left_eye = np.array(left_eye, dtype=np.int32)
            right_eye = np.array(right_eye, dtype=np.int32)
            cv2.fillPoly(face_img, [left_eye, right_eye], color=(0, 0, 0))
        elif apply_mask == 'nose':
            nose = preds[[27, 28, 29, 30, 31, 32, 33, 34, 35]]
            nose = np.array(nose, dtype=np.int32)
            cv2.fillPoly(face_img, [nose], color=(0, 0, 0))
        elif apply_mask == 'mouth':
            mouth = preds[[48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]]
            mouth = np.array(mouth, dtype=np.int32)
            cv2.fillPoly(face_img, [mouth], color=(0, 0, 0))
        elif apply_mask == 'eyebrows':
            left_eyebrow = preds[[17, 18, 19, 20, 21]]
            right_eyebrow = preds[[22, 23, 24, 25, 26]]
            left_eyebrow = np.array(left_eyebrow, dtype=np.int32)
            right_eyebrow = np.array(right_eyebrow, dtype=np.int32)
            cv2.fillPoly(face_img, [left_eyebrow, right_eyebrow], color=(0, 0, 0))
        np_image[y1:y2, x1:x2] = face_img
        
        return Image.fromarray(np_image)
    else:
        return None
    
def compare_gt_score_pairs(model_scores, gt_scores):
    if gt_scores[0] > gt_scores[1]:
        if model_scores[0] > model_scores[1]:
            return 1
        else:
            return 0
    elif gt_scores[0] < gt_scores[1]:
        if model_scores[0] < model_scores[1]:
            return 1
        else:
            return 0
    else:
        if model_scores[0] == model_scores[1]:
            return 1
        else:
            return 0

def compare_elements(vali, valj, i, j):
    if vali >= valj:
        return i
    else:
        return j
    
def compare_range(vali, valj, i, j, chosen_idx):
    if i == chosen_idx:
        if vali[0] >= valj[0] or vali[1] >= valj[1]:
            return i
        else:
            return j
    elif j == chosen_idx:
        if valj[0] >= vali[0] or valj[1] >= vali[1]:
            return j
        else:
            return i
        
def pairwise_compare(model_list, gt_list, check_range=False):
    compare_model = []
    compare_gt = []
    n = len(model_list)
    for i in range(n):
        for j in range(i+1, n):
            if i != j:
                if check_range:
                    chosen_idx = compare_elements(gt_list[i], gt_list[j], i, j)
                    compare_gt.append(chosen_idx)
                    compare_model.append(compare_range(model_list[i], model_list[j], i, j, chosen_idx))
                else:
                    compare_model.append(compare_elements(model_list[i], model_list[j], i, j))
                    compare_gt.append(compare_elements(gt_list[i], gt_list[j], i, j))
    match_count = sum(1 for i, j in zip(compare_model, compare_gt) if i == j)
    match_pct = (match_count / len(compare_model)) * 100
    return match_pct

def check_in_range(model_list, gt_list):
    n = len(model_list)
    acc = []
    for i in range(n):
        if gt_list[i] >= model_list[i][0] and gt_list[i] <= model_list[i][1]:
            acc.append(1)
        else:
            acc.append(0)
    return (acc.count(1) / len(acc)) * 100

def calc_acc(score_list):
    if score_list:
        return (f'{score_list.count(1)} / {len(score_list)}, {(score_list.count(1) / len(score_list)) * 100}')
    else:
        return None

def kendall_tau(rank1, rank2):
    """
    Calculates Kendall's Tau rank correlation coefficient.

    Args:
        rank1 (torch.Tensor): The first ranking (list or tensor).
        rank2 (torch.Tensor): The second ranking (list or tensor).

    Returns:
        float: Kendall's Tau coefficient.
    """
    n = len(rank1)
    num_concordant = 0
    num_discordant = 0

    for i in range(n):
        for j in range(i + 1, n):
            if (rank1[i] < rank1[j] and rank2[i] < rank2[j]) or (rank1[i] > rank1[j] and rank2[i] > rank2[j]):
                num_concordant += 1
            else:
                num_discordant += 1

    tau = (num_concordant - num_discordant) / (n * (n - 1) / 2)
    return tau

def normalize_list(data):
    min_val = min(data) - 0.01
    max_val = max(data)
    normalized_data = [(x - min_val) / (max_val - min_val) for x in data]
    return normalized_data

def normalized_item(item, max_val, min_val):
    return (item - min_val) / (max_val - min_val)

def sqrt_mapping(item):
    return 1 - torch.sqrt((1 - item)/2)

torch.pi = torch.acos(torch.zeros(1)).item() * 2 # https://discuss.pytorch.org/t/np-pi-equivalent-in-pytorch/67157/2
def angular_sim(item):
    print(torch.pi)
    print('------')
    return 1 - (torch.arccos(item) / torch.pi)

def weighted_mean(data, weights):
    """
    Calculates the weighted mean of a tensor.

    Args:
    data: The input tensor.
    weights: The weights tensor, with the same shape as data.

    Returns:
    The weighted mean of the input tensor.
    """
    if data.size() != weights.size():
        raise ValueError("Data and weights tensors must have the same size")

    weighted_values = data * weights
    return torch.sum(weighted_values) / torch.sum(weights)

def average_at_index(list1, list2):
    """
    Calculates the average of values at the same index in two lists.

    Args:
        list1: The first list.
        list2: The second list.

    Returns:
        A new list containing the averages, or None if lists are not of equal length.
    """
    if len(list1) != len(list2):
        return None  # Or raise an exception, depending on desired behavior

    averages = []
    for i in range(len(list1)):
        averages.append((list1[i] + list2[i]) / 2)
    return averages

def calculate_mean(num1, num2):
    """
    Calculates the mean of two numbers.

    Args:
        num1: The first number.
        num2: The second number.

    Returns:
        The mean of the two numbers.
    """
    mean = (num1 + num2) / 2
    return mean

def filter_dict_by_value_threshold(input_dict, threshold, greater_than=True):
    """
    Filters a dictionary based on a value threshold.

    Args:
        input_dict (dict): The dictionary to filter.
        threshold: The threshold value to compare against.
        greater_than (bool, optional): If True, keep values greater than or equal to the threshold.
                                     If False, keep values less than the threshold.
                                     Defaults to True.

    Returns:
        dict: A new dictionary containing only the key-value pairs that meet the threshold condition.
    """
    filtered_dict = {}
    for key, value in input_dict.items():
        if greater_than:
            if value >= threshold:
                filtered_dict[key] = value
        else:
            if value < threshold:
                filtered_dict[key] = value
    return filtered_dict

def filter_dict_by_top_n_values(input_dict, top_n_obj_atts):
    return dict(list(input_dict.items())[:top_n_obj_atts])

def clean_sentence(sentence):
    return re.findall(r'"([^"]*)"', sentence)

def pairwise_cosine_similarity_manual(x):
    x_normalized = F.normalize(x, p=2, dim=1)
    similarity_matrix = torch.matmul(x_normalized, x_normalized.transpose(0, 1))
    return similarity_matrix

def calc_semantic_diversity(captions, sentence_model, method='average'):
    embeddings = sentence_model.encode(captions, convert_to_tensor=True)
    similarity_matrix = pairwise_cosine_similarity_manual(embeddings)
    
    num_texts = similarity_matrix.shape[0]
    if num_texts <= 1:
        return 0.0  # No diversity if there's only one text

    # Remove diagonal elements
    mask = ~torch.eye(num_texts, dtype=torch.bool)
    similarities = similarity_matrix[mask]

    if method == 'average':
        return torch.mean(similarities).item()
    elif method == 'variance':
        return torch.var(similarities).item()
    elif method == 'entropy':
        hist = torch.histc(similarities, bins=20)
        hist = hist[hist > 0]
        probs = hist / torch.sum(hist)
        entropy = -torch.sum(probs * torch.log2(probs))
        return entropy.item()
    else:
        raise ValueError("Invalid method")