import os
import torch
import argparse
from PIL import Image
import json

from lib import data_processing as dp
from lib.config import VLM_CONFIGS, DATASET_CONFIGS
from lib.strategies import VLMStrategy, STRATEGY_MAP

import load_models
from sklearn.metrics import mean_absolute_error

import warnings
warnings.filterwarnings("ignore")
import logging.config
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})

device = (torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu"))

def calculate_similarities(strategy: VLMStrategy, model, image, caption_concept_list, captions_mllm, labels, **kwargs):
    """Calculates similarities between an image and various text captions using a specified VLM strategy."""
    
    image_features = strategy.encode_image(model, image, **kwargs)
    
    text_features_orig = strategy.encode_text(model, caption_concept_list, **kwargs)
    similarity_orig = strategy.calculate_similarity(image_features, text_features_orig, **kwargs).tolist()

    similarity_range = []
    for sub_att in labels:
        mllm_caps = captions_mllm.get(sub_att)
        if mllm_caps:
            text_features_range = strategy.encode_text(model, mllm_caps, **kwargs)
            sims = strategy.calculate_similarity(image_features, text_features_range, **kwargs).tolist()
            similarity_range.append((min(sims), max(sims)))
        else:
            similarity_range.append((0.0, 0.0))

    return similarity_orig, similarity_range

def calculate_evaluation_metrics(
    mask_regions,
    labels,
    aggregated_data,
    vlm_orig_scores,
    vlm_range_scores
):
    """Calculates final evaluation metrics from aggregated data."""
    metrics = {
        'pct_region_identify_correct': dp.calc_acc(aggregated_data['region_identify_correct']),
        'acc_NEG_REQ': {
            reg: dp.calc_acc(aggregated_data['mask_regions_detection_NEG_REQ'][reg])
            for reg in mask_regions
        },
        'acc_POS_REQ': {
            masked_region: {
                checked_region: dp.calc_acc(scores)
                for checked_region, scores in checked_outcomes.items()
            }
            for masked_region, checked_outcomes in aggregated_data['mask_regions_detection_POS_REQ'].items()
        }
    }

    # Initialize dictionaries for attribute-specific metrics
    metrics['mllm_mae'] = {}
    metrics['mllm_in_range_acc'] = {}
    metrics['match_pct'] = {}
    metrics['in_range_acc'] = {}

    for attr in labels:
        metrics['mllm_mae'][attr] = mean_absolute_error(aggregated_data['gt_scores'][attr], aggregated_data['mllm_single_scores'][attr])
        metrics['mllm_in_range_acc'][attr] = dp.check_in_range(aggregated_data['mllm_range_scores'][attr], aggregated_data['gt_scores'][attr])
        metrics['match_pct'][attr] = dp.pairwise_compare(vlm_orig_scores[attr], aggregated_data['gt_scores'][attr])
        metrics['in_range_acc'][attr] = dp.pairwise_compare(vlm_range_scores[attr], aggregated_data['gt_scores'][attr], check_range=True)

    return metrics

def save_results(results, folder_path, mllm_choice, vlm_model):
    """Saves evaluation results to a JSON file."""
    filename = f'{mllm_choice}__{vlm_model}__final_results.json'
    with open(os.path.join(folder_path, filename), "w") as f:
        json.dump(results, f)

def load_and_process_observations(observations_folder_path, mllm_choice, dataset_config, labels, mask_regions, image_src_base):
    """
    Loads and aggregates all pre-computed MLLM observation data from disk.
    """
    # Initialization
    results = {
        'image_masking_fail': 0,
        'region_identify_correct': [],
        'mask_regions_detection_NEG_REQ': {key: [] for key in mask_regions},
        'mask_regions_detection_POS_REQ': {m_key: {c_key: [] for c_key in mask_regions if c_key != m_key} for m_key in mask_regions},
        'gt_scores': {key: [] for key in labels},
        'mllm_single_scores': {key: [] for key in labels},
        'mllm_range_scores': {key: [] for key in labels},
        'captions_mllm_diversity_score': {key: [] for key in labels},
        'image_data_for_vlm': []  # This will store {'image_path': str, 'captions': dict}
    }

    # Pre-load OMI corrected scores if necessary
    omi_corrected_map = {}
    if dataset_config['path'] == 'OMI':
        omi_corrected_path = f'{image_src_base}/original_scores/omi__norm_0_1__ranks.json'
        with open(omi_corrected_path, 'r') as file:
            omi_corrected_list = json.load(file)
        omi_corrected_map = {item['image_name']: item for item in omi_corrected_list}

    files = [entry for entry in os.listdir(observations_folder_path) if os.path.isdir(os.path.join(observations_folder_path, entry))]

    for file_name in files:
        obs_path = os.path.join(observations_folder_path, file_name)
        if not os.listdir(obs_path):
            results['image_masking_fail'] += 1
            continue

        with open(os.path.join(obs_path, f'{mllm_choice}__save_mllm_obs.json'), 'r') as file:
            save_mllm_obs = json.load(file)

        # Aggregate region detection results
        neg_req_results = save_mllm_obs.get('mask_regions_detection_NEG_REQ', {})
        for region, outcome in neg_req_results.items():
            results['mask_regions_detection_NEG_REQ'][region].append(outcome)

        pos_req_results = save_mllm_obs.get('mask_regions_detection_POS_REQ', {})
        for checked_region, outcomes in pos_req_results.items():
            corresponding_masked_regions = [m for m in mask_regions if m != checked_region]
            for masked_region, outcome in zip(corresponding_masked_regions, outcomes):
                results['mask_regions_detection_POS_REQ'][masked_region][checked_region].append(outcome)

        if save_mllm_obs.get('region_identify_correct') == 1:
            results['region_identify_correct'].append(1)

            # Get ground truth and MLLM scores
            omi_gt_data = omi_corrected_map.get(int(file_name))
            for lab in labels:
                results['gt_scores'][lab].append(omi_gt_data[lab][0] if omi_gt_data and lab in omi_gt_data else (save_mllm_obs.get('gt_scores', {}).get(lab, [0])[0]))
                results['mllm_single_scores'][lab].extend(save_mllm_obs.get('mllm_single_scores', {}).get(lab, []))
                results['mllm_range_scores'][lab].extend(save_mllm_obs.get('mllm_range_scores', {}).get(lab, []))

            # Load filtered captions for VLM evaluation
            with open(os.path.join(obs_path, f'{mllm_choice}__mllm_filtered.json'), 'r') as file:
                mllm_filtered = json.load(file)
            
            results['image_data_for_vlm'].append({
                'image_path': os.path.join(image_src_base, dataset_config['path'], f'{file_name}.jpg'),
                'captions': {lab: mllm_filtered.get(lab, {}).get('captions_mllm', []) for lab in labels}
            })
            for lab in labels:
                results['captions_mllm_diversity_score'][lab].append(mllm_filtered.get(lab, {}).get('captions_mllm_diversity_score', 0))
        else:
            results['region_identify_correct'].append(0)
    
    return results

def run_vlm_evaluation(strategy, vlm_models, image_data_for_vlm, labels, caption_concept_list, vlm_model_name):
    """Runs the VLM similarity calculations for all valid images."""
    vlm_orig_scores = {key: [] for key in labels}
    vlm_range_scores = {key: [] for key in labels}

    with torch.no_grad():
        for item in image_data_for_vlm:
            image = Image.open(item['image_path']).convert("RGB")
            
            similarity_orig, similarity_range = calculate_similarities(
                strategy=strategy,
                image=image,
                caption_concept_list=caption_concept_list,
                captions_mllm=item['captions'],
                labels=labels,
                vlm_model_name=vlm_model_name,
                **vlm_models
            )
            
            for sim_orig, sim_range, attr in zip(similarity_orig, similarity_range, labels):
                vlm_orig_scores[attr].append(sim_orig)
                vlm_range_scores[attr].append(sim_range)
    return vlm_orig_scores, vlm_range_scores

def main(args):
    src = args.data_dir

    dataset_config = DATASET_CONFIGS[args.dataset]
    labels = dataset_config['labels']
    
    mask_regions = ['eyes', 'nose', 'mouth', 'eyebrows']
        
    observations_folder_path = f'{args.output_dir}/{args.mllm}/{args.dataset}'
    os.makedirs(observations_folder_path, exist_ok=True)
        
    print(f"Loading VLM for evaluation: {args.vlm_model}...")
    vlm_models = load_models.load_models_for_eval(args.vlm_model, args.checkpoints_dir)
    print("VLM loaded.")

    vlm_type = VLM_CONFIGS[args.vlm_model]['type']
    strategy = STRATEGY_MAP[vlm_type]
    
    vlm_models['model'].eval()

    # 1. Load and process all pre-computed MLLM observations
    print("Loading and processing MLLM observations...")
    aggregated_data = load_and_process_observations(
        observations_folder_path, args.mllm, dataset_config, labels, mask_regions, src
    )
    print(f"Processed observations. Found {len(aggregated_data['image_data_for_vlm'])} valid samples for VLM evaluation.")

    # 2. Run VLM evaluation on the processed data
    print("Running VLM evaluation...")
    caption_concept_list = [f'a photo of a {item} face' for item in labels]
    vlm_orig_scores, vlm_range_scores = run_vlm_evaluation(
        strategy, vlm_models, aggregated_data['image_data_for_vlm'], labels, caption_concept_list, args.vlm_model
    )
    print("VLM evaluation complete.")

    # 3. Calculate final metrics and save results
    print("Calculating final metrics and saving results...")
    final_metrics = calculate_evaluation_metrics(
        mask_regions=mask_regions,
        labels=labels,
        aggregated_data=aggregated_data,
        vlm_orig_scores=vlm_orig_scores,
        vlm_range_scores=vlm_range_scores
    )
    save_results(final_metrics, observations_folder_path, args.mllm, args.vlm_model)
    print("Done.")
                             
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run VLM evaluation on pre-computed MLLM observations.")
    parser.add_argument('--data_dir', type=str, default='subjective_attribute_datasets', help='Path to dataset directory')
    parser.add_argument('--mllm', type=str, default='llava_qwen2_si',
                        choices=['internvl2', 'internvl3', 'minicpm', 'llava_qwen2_si'],
                        help='The MLLM whose observations are being evaluated.')
    parser.add_argument('--dataset', type=str, default='omi_subj_attr',
                        choices=['cfd_subj_attr', 'omi_subj_attr', 'uaf_subj_attr'],
                        help='The subjective attribute dataset that was processed.')
    parser.add_argument('--vlm_model', type=str, default='farl_clip_vit_b16',
                        choices=['clip_vit_l14', 'clip_vit_b16', 'meru_vit_b16', 'hycoclip_vit_b16', 'metaclip', 'farl_clip_vit_b16'],
                        help='The VLM to use for evaluation.')
    parser.add_argument('--output_dir', type=str, default='demo_results',
                        help='The root directory where MLLM observations are stored.')
    parser.add_argument('--checkpoints_dir', type=str, default='pretrained_models',
                        help='The directory where large, downloaded model checkpoints are stored.')

    args = parser.parse_args()
    main(args)