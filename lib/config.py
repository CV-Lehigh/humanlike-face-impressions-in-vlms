import omegaconf
import typing
import collections
import torch

# --- MLLM Configurations ---
MLLM_CONFIGS = {
    'internvl2': {
        'path': 'OpenGVLab/InternVL2-8B',
        'model_class': 'AutoModel',
        'params': {'attn_implementation': 'flash_attention_2'},
        'gen_config': {'max_new_tokens': 1024, 'do_sample': True},
        'preprocess': 'dynamic',
        'prompts': {
            'q1_template': '<image>\nDescribe the {sub_att} attribute based on this photo of a face in less than 10 words.',
            'q3_template': 'Generate 5 captions for this photo of a face that highlight the {sub_att} attribute based on your score.'
        }
    },
    'internvl3': {
        'path': 'OpenGVLab/InternVL3-14B',
        'model_class': 'AutoModel',
        'params': {'attn_implementation': 'flash_attention_2'},
        'gen_config': {'max_new_tokens': 1024, 'do_sample': True},
        'prompts': {
            'q1_template': '<image>\nDescribe the {sub_att} attribute based on this photo of a face in less than 10 words.',
            'q3_template': 'Generate 5 captions for this photo of a face that highlight the {sub_att} attribute based on your score.'
        }
    },
    'minicpm': {
        'path': 'openbmb/MiniCPM-o-2_6',
        'model_class': 'AutoModel',
        'params': {'attn_implementation': 'flash_attention_2'},
        'gen_config': None,
        'prompts': {
            'q1_template': 'Describe the {sub_att} attribute based on this photo of a face.',
            'q3_template': 'Generate 5 captions for this photo of a face that highlight the {sub_att} attribute based on your score. Put the captions in quotations.'
        },
        'quirks': { # Model-specific behavioral flags
            'ignore_eyebrows_localization': True
        },
        'device_placement': 'cuda' # Requires explicit .cuda() call after loading
    },
    'llava_qwen2_si': {
        'path': 'llava-hf/llava-onevision-qwen2-7b-si-hf',
        'model_class': 'LlavaOnevisionForConditionalGeneration',
        'params': {
            'torch_dtype': torch.float16  # This model specifically requires float16, not the default bfloat16.
        },
        'gen_config': None,
        'prompts': {
            'q1_template': '<image>\nDescribe the {sub_att} attribute based on this photo of a face in less than 10 words.',
            'q3_template': 'Generate 5 captions for this photo of a face that highlight the {sub_att} attribute based on your score.'
        },
        'device_placement': 0 # Requires explicit .to(0) call after loading
    }
}

# --- VLM (for evaluation) Configurations ---
VLM_CONFIGS = {
    'farl_clip_vit_b16': {
        'type': 'open_clip',
        'model_name': 'ViT-B-16',
        'pretrained': '',
        'checkpoint': 'FaRL-Base-Patch16-LAIONFace20M-ep16.pth' # Relative to the checkpoints_dir provided at runtime
    },
    'clip_vit_l14': {
        'type': 'huggingface_clip',
        'model_id': 'openai/clip-vit-large-patch14'
    },
    'clip_vit_b16': {
        'type': 'huggingface_clip',
        'model_id': 'openai/clip-vit-base-patch16'
    },
    'meru_vit_b16': {
        'type': 'hycoclip',
        'model_name_base': 'meru'
    },
    'hycoclip_vit_b16': {
        'type': 'hycoclip',
        'model_name_base': 'hycoclip'
    },
    'metaclip': {
        'type': 'open_clip',
        'model_name': 'ViT-B-32-quickgelu',
        'pretrained': 'metaclip_400m',
        'checkpoint': None
    }
}

# --- Dataset Configurations ---
DATASET_CONFIGS = {
    'cfd_subj_attr': {
        'labels': ['attractive', 'feminine', 'masculine', 'happy', 'sad', 'trustworthy', 'dominance'],
        'path': 'CFD',
        'scores_file': 'cfd__norm_0_1__ranks.json'
    },
    'omi_subj_attr': {
        'labels': ['attractive', 'happy', 'trustworthy', 'dominance'],
        'path': 'OMI',
        'scores_file': 'omi__norm_0_1__ranks.json'
    },
    'uaf_subj_attr': {
        'labels': ['attractive', 'happy', 'trustworthy'],
        'path': 'UAF',
        'scores_file': 'uaf__norm_0_1.json'
    }
}

# Globals required for loading HyboCLIP/MeruCLIP checkpoints safely
HYCOCLIP_SAFE_GLOBALS = [
    omegaconf.listconfig.ListConfig,
    omegaconf.dictconfig.DictConfig,
    omegaconf.base.ContainerMetadata,
    omegaconf.base.Metadata,
    omegaconf.nodes.AnyNode,
    typing.Any,
    list,
    collections.defaultdict,
    dict,
    int,
]