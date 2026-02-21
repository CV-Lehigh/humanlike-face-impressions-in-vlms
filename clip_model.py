import torch
from lib import data_processing as dp
from lib.strategies import VLMStrategy

import warnings
warnings.filterwarnings("ignore")
import logging.config
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})

device = (torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu"))

def process_exp00(strategy: VLMStrategy, model, labels, available_attributes, top_descriptions, obj_descr_path, **kwargs):
    """Performs a zero-shot experiment to find top descriptive attributes."""
    subj_attributes_text_features = strategy.encode_text(model, labels, **kwargs)
    choice_text_features = strategy.encode_text(model, available_attributes, **kwargs)
    
    # Assuming dp.tsne_plot can handle tensors directly
    # dp.tsne_plot(pretrained_model, subj_attributes_text_features, choice_text_features, labels, available_attributes, f'{obj_descr_path}/visualize_attributes')
    
    for lab, sa_text in zip(labels, subj_attributes_text_features):
        sim = strategy.calculate_similarity(sa_text.unsqueeze(0), choice_text_features, **kwargs)
        sim, indices = torch.sort(sim, descending=True)
        for index, sim_score in zip(indices, sim):
            top_descriptions[lab].update({available_attributes[index]: sim_score.item()})
    return top_descriptions

def process_exp0(strategy: VLMStrategy, model, caption_concept_list, image, image__mask_eyes, image__mask_nose, image__mask_mouth, image__mask_eyebrows, **kwargs):
    """Calculates the change in similarity when facial regions are masked."""
    text_features = strategy.encode_text(model, caption_concept_list, **kwargs)
    
    def get_similarity(img):
        img_feats = strategy.encode_image(model, img, **kwargs)
        return strategy.calculate_similarity(img_feats, text_features, **kwargs)

    similarity = get_similarity(image)
    similarity__eyes = get_similarity(image__mask_eyes)
    similarity__nose = get_similarity(image__mask_nose)
    similarity__mouth = get_similarity(image__mask_mouth)
    similarity__eyebrows = get_similarity(image__mask_eyebrows)

    diff_eyes = (similarity - similarity__eyes).tolist()
    diff_nose = (similarity - similarity__nose).tolist()
    diff_mouth = (similarity - similarity__mouth).tolist()
    diff_eyebrows = (similarity - similarity__eyebrows).tolist()

    return diff_eyes, diff_nose, diff_mouth, diff_eyebrows

def process_exp1(strategy: VLMStrategy, model, labels, caption_concept_list, captions_mllm, selected_obj_atts, image, **kwargs):
    """Calculates single-text, objective-corrected, and generated-caption similarity scores."""
    image_features = strategy.encode_image(model, image, **kwargs)
    
    # similarity_orig
    text_features_orig = strategy.encode_text(model, caption_concept_list, **kwargs)
    similarity_orig = strategy.calculate_similarity(image_features, text_features_orig, **kwargs).tolist()
    
    # similarity_objective_corrected
    similarity_objective_corrected = []
    for a, sub_att in enumerate(labels):
        obj_att_list = [f'a photo of a face with {obj_att} attribute' for obj_att in list(selected_obj_atts[sub_att].keys())]
        text_features_obj = strategy.encode_text(model, obj_att_list, **kwargs)
        sims = strategy.calculate_similarity(image_features, text_features_obj, **kwargs)
        max_sim = torch.max(sims).item()
        min_sim = torch.min(sims).item()
        similarity_objective_corrected.append(dp.normalized_item(similarity_orig[a], max_sim, min_sim))
    
    # similarity_range
    similarity_range = []
    for sub_att in labels:
        mllm_caps = captions_mllm[sub_att]
        text_features_range = strategy.encode_text(model, mllm_caps, **kwargs)
        sim = strategy.calculate_similarity(image_features, text_features_range, **kwargs).tolist()
        max_sim = max(sim)
        min_sim = min(sim)
        similarity_range.append((min_sim, max_sim))
    
    return similarity_orig, similarity_objective_corrected, similarity_range
