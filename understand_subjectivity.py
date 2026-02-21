import os
import random
import statistics
import torch
import re
import json
import argparse
import logging.config

from transformers import pipeline
from sentence_transformers import SentenceTransformer
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from lib import image_caption
from lib import image_utils
from lib import data_processing as dp
from lib.config import DATASET_CONFIGS
import load_models

import warnings
warnings.filterwarnings("ignore")

logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})

class SubjectivityAnalyzer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mask_regions = ['eyes', 'nose', 'mouth', 'eyebrows']
        self.localization_check_count = len(self.mask_regions)
        
        self.setup_labels()
        self.setup_models()
        self.setup_directories()

    def setup_labels(self):
        if self.args.dataset in DATASET_CONFIGS:
            self.labels = DATASET_CONFIGS[self.args.dataset]['labels']
        else:
            raise ValueError(f"Unknown dataset: {self.args.dataset}")

    def setup_models(self):
        print(f"Loading MLLM: {self.args.mllm}...")
        self.mllm_model, self.mllm_tokenizer, self.mllm_gen_config, self.mllm_processor = \
            load_models.load_models_for_attribute_understanding(self.args.mllm)

        print("Loading auxiliary models...")
        # Optimize sentiment analysis by running on GPU if available
        sentiment_device = 0 if torch.cuda.is_available() else -1
        self.sentiment_analysis = pipeline(
            "sentiment-analysis",
            model='distilbert/distilbert-base-uncased-finetuned-sst-2-english',
            device=sentiment_device,
            truncation=True
        )
        
        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)

        base_options = python.BaseOptions(model_asset_path='lib/face_landmarker_v2_with_blendshapes.task')
        options = vision.FaceLandmarkerOptions(
            base_options=base_options, 
            output_face_blendshapes=True, 
            output_facial_transformation_matrixes=True, 
            num_faces=1
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def setup_directories(self):
        self.observations_folder_path = f'{self.args.output_dir}/{self.args.mllm}/{self.args.dataset}'
        os.makedirs(self.observations_folder_path, exist_ok=True)

    def get_mllm_response(self, image_input, prompt, history=None):
        if self.args.mllm in ['internvl2', 'internvl3']:
            response, new_history = self.mllm_model.chat(
                self.mllm_tokenizer, 
                image_input, 
                prompt, 
                self.mllm_gen_config, 
                history=history, 
                return_history=True
            )
            return response, new_history
            
        elif self.args.mllm == 'minicpm':
            if history is None:
                msgs = [{'role': 'user', 'content': [image_input, prompt]}]
            else:
                msgs = history
                msgs.append({'role': 'user', 'content': [prompt]})

            response = self.mllm_model.chat(msgs=msgs, tokenizer=self.mllm_tokenizer)
            msgs.append({'role': 'assistant', 'content': [response]})
            return response, msgs
            
        elif self.args.mllm == 'llava_qwen2_si':
            if history is None:
                conversation = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image"}]}]
            else:
                conversation = history
                conversation.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
            
            prompt_str = self.mllm_processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = self.mllm_processor(images=image_input, text=prompt_str, return_tensors='pt').to(self.device, torch.float16)
            output = self.mllm_model.generate(**inputs, max_new_tokens=200, do_sample=False)
            full_response = self.mllm_processor.decode(output[0][2:], skip_special_tokens=True)
            
            response = full_response.split('assistant')[-1].strip()
            
            conversation.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
            return response, conversation

        return "", None

    def localize_regions(self, mllm_input):
        mask_regions_detection_NEG_REQ = {key: [] for key in self.mask_regions}
        mask_regions_detection_POS_REQ = {key: [] for key in self.mask_regions}
        region_identify = 0
        
        # mllm_input[0] is original, mllm_input[1:] are masked images in order of mask_regions
        for m_region, region_mask_input in zip(self.mask_regions, mllm_input[1:]):
            n_times_dict = {key: [] for key in self.mask_regions}
            
            for _ in range(self.args.localization_iters):
                for reg in self.mask_regions:
                    if self.args.mllm in ['internvl2', 'internvl3']:
                        prompt = f'<image>\nIs the {reg} attribute visible in this photo of a face?'
                    else:
                        prompt = f'Is the {reg} attribute visible in this photo of a face?'
                    
                    mask_region_check, _ = self.get_mllm_response(region_mask_input, prompt, history=None)
                    # -------------------------------------------------

                    mask_sentiment = self.sentiment_analysis(mask_region_check)[0]
                    
                    is_negative = mask_sentiment['label'] == 'NEGATIVE' and mask_sentiment['score'] > 0.9
                    
                    if reg == m_region:
                        n_times_dict[reg].append(1 if is_negative else 0)
                    else:
                        n_times_dict[reg].append(0 if is_negative else 1)

            # Aggregate results for this masked region
            for reg in self.mask_regions:
                if n_times_dict[reg].count(1) >= 2:
                    if reg == m_region:
                        mask_regions_detection_NEG_REQ[reg].append(1)
                        if self.args.mllm == 'minicpm':
                            if reg != 'eyebrows':
                                region_identify += 1
                        else:
                            region_identify += 1
                    else:
                        mask_regions_detection_POS_REQ[reg].append(1)
                else:
                    if reg == m_region:
                        mask_regions_detection_NEG_REQ[reg].append(0)
                    else:
                        mask_regions_detection_POS_REQ[reg].append(0)

        return region_identify, mask_regions_detection_NEG_REQ, mask_regions_detection_POS_REQ

    def analyze_attributes(self, mllm_input, gt_data_batch):
        gt_scores = {key: [] for key in self.labels}
        mllm_single_scores = {key: [] for key in self.labels}
        mllm_range_scores = {key: [] for key in self.labels}
        mllm_filtered = {key: {} for key in self.labels}

        for idx, sub_att in enumerate(self.labels):
            score_list = []
            captions = []
            
            for _ in range(self.args.description_iters):
                # --- CONVERSATIONAL STATEMENTS (DO NOT CHANGE) ---
                # Turn 1
                if self.args.mllm in ['internvl2', 'internvl3']:
                    prompt = f'<image>\nDescribe the {sub_att} attribute based on this photo of a face in less than 10 words.'
                else:
                    prompt = f'Describe the {sub_att} attribute based on this photo of a face.'
                response1, history = self.get_mllm_response(mllm_input[0], prompt, history=None)
                
                # Turn 2
                prompt = 'Based on your analysis, only provide a floating point score between 0 and 1.'
                response2, history = self.get_mllm_response(mllm_input[0], prompt, history=history)
                score_list += re.findall(r"[-+]?\d*\.?\d+", response2)
                
                # Turn 3
                if self.args.mllm in ['internvl2', 'internvl3']:
                    prompt = f'Generate 5 captions for this photo of a face that highlight the {sub_att} attribute based on your score.'
                else:
                    prompt = f'Generate 5 captions for this photo of a face that highlight the {sub_att} attribute based on your score. Put the captions in quotations.'
                response3, history = self.get_mllm_response(mllm_input[0], prompt, history=history)
                captions += dp.clean_sentence(response3)
                # -------------------------------------------------

            # Process scores and captions
            scores = [float(i) for i in score_list]
            if not scores: scores = [0.0] # Handle empty scores safety
            max_score = max(scores)
            min_score = min(scores)
            
            # Calculate diversity and filter captions
            if captions:
                captions_emb = self.sentence_model.encode(captions)
                similarities = self.sentence_model.similarity(captions_emb, captions_emb)
                
                _, top_indices = torch.topk(similarities, k=1, dim=1, largest=False)
                top_indices = torch.unique(top_indices).tolist()
                
                if len(top_indices) < 5:
                    if len(captions) > 5:
                        selected_captions = random.sample(captions, 5)
                    else:
                        selected_captions = captions
                else:
                    top_indices = random.sample(top_indices, 5)
                    selected_captions = [captions[i] for i in top_indices]
                
                diversity_score = dp.calc_semantic_diversity(selected_captions, self.sentence_model)
            else:
                selected_captions = []
                diversity_score = 0.0

            mllm_filtered[sub_att].update({
                'captions_mllm': selected_captions,
                'captions_mllm_diversity_score': diversity_score
            })
            
            mllm_single_scores[sub_att].append(statistics.median(scores))
            mllm_range_scores[sub_att].append((min_score, max_score))
            
            # Extract Ground Truth score from batch data
            # data[0] is the attributes dict batch. list(data[0].values()) gives tensors for each attribute.
            gt_scores[sub_att].append(list(gt_data_batch.values())[idx].item())

        return gt_scores, mllm_single_scores, mllm_range_scores, mllm_filtered

    def run(self):
        # Load dataset
        _, data_loader = image_caption.get_test_loader(
            self.args.data_dir, '1', batch_size=1, workers=0, 
            shuffle=False, split='test', 
            load_subj_attr_data=self.args.dataset, load_image=False
        )
        
        count = 0
        with torch.no_grad():
            for _, data in enumerate(data_loader):
                # Unpack data based on load_image=False structure
                # data[0]: attributes dict, data[1]: image_name tuple, data[2]: image_path tuple
                gt_data_batch = data[0]
                image_name = data[1][0]
                image_path = data[2][0]
                
                image_observations = f'{self.observations_folder_path}/{image_name}'
                
                if os.path.exists(image_observations):
                    continue

                os.makedirs(image_observations, exist_ok=True)
                
                # Load and mask image
                mllm_input, load_success = image_utils.load_and_mask_image(image_path, self.detector, mllm_choice=self.args.mllm)
                if not load_success:
                    continue
                
                # Step 1: Localization
                region_identify, neg_req, pos_req = self.localize_regions(mllm_input)
                
                region_identify_correct = 0
                gt_scores = {key: [] for key in self.labels}
                mllm_single_scores = {key: [] for key in self.labels}
                mllm_range_scores = {key: [] for key in self.labels}
                
                if region_identify == self.localization_check_count:
                    region_identify_correct = 1
                    
                    # Step 2: Attribute Analysis
                    gt_scores, mllm_single_scores, mllm_range_scores, mllm_filtered = self.analyze_attributes(mllm_input, gt_data_batch)
                    
                    # Save filtered captions
                    with open(f"{image_observations}/{self.args.mllm}__mllm_filtered.json", "w") as f:
                        json.dump(mllm_filtered, f)
                
                # Save observations
                save_mllm_obs = {
                    'region_identify_correct': region_identify_correct,
                    'mask_regions_detection_NEG_REQ': neg_req,
                    'mask_regions_detection_POS_REQ': pos_req,
                    'gt_scores': gt_scores,
                    'mllm_single_scores': mllm_single_scores,
                    'mllm_range_scores': mllm_range_scores
                }
                with open(f"{image_observations}/{self.args.mllm}__save_mllm_obs.json", "w") as f:
                    json.dump(save_mllm_obs, f)
                
                print(f'Item {count+1} processed: {image_name}')
                count += 1
                if self.args.num_images != -1 and count == self.args.num_images:
                    break

def main(args):
    analyzer = SubjectivityAnalyzer(args)
    analyzer.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Understand Subjectivity")
    parser.add_argument('--data_dir', type=str, default='subjective_attribute_datasets', help='Path to dataset directory')
    parser.add_argument('--localization_iters', type=int, default=3, help='Number of iterations for localization')
    parser.add_argument('--description_iters', type=int, default=5, help='Number of iterations for description')
    parser.add_argument('--num_images', type=int, default=10, help='Number of images to process')
    parser.add_argument('--dataset', type=str, default='omi_subj_attr', choices=['cfd_subj_attr', 'uaf_subj_attr', 'omi_subj_attr'], help='Dataset to use')
    parser.add_argument('--mllm', type=str, default='llava_qwen2_si', choices=['internvl2', 'internvl3', 'minicpm', 'llava_qwen2_si'], help='MLLM model to use')
    parser.add_argument('--output_dir', type=str, default='demo_results', help='Output directory')
    
    args = parser.parse_args()
    main(args)