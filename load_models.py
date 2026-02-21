import os
import torch
import torchvision.transforms as transforms

from transformers import (
    Qwen2VLForConditionalGeneration, LlavaOnevisionForConditionalGeneration, AutoTokenizer, 
    AutoProcessor, AutoModel, pipeline, AutoModelForSequenceClassification
)
from sentence_transformers import SentenceTransformer
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from MetaCLIP.src.open_clip.factory import create_model_and_transforms

from lib.config import MLLM_CONFIGS, VLM_CONFIGS, HYCOCLIP_SAFE_GLOBALS

import warnings
warnings.filterwarnings("ignore")
import logging.config
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': True,
})

device = (torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu"))

import contextlib
import functools

@contextlib.contextmanager
def unsafe_torch_load_context():
    """
    Context manager to temporarily patch torch.load to disable weights_only check.
    
    This will execute any pickled Python code inside checkpoint files loaded
    within this context. Only use with trusted checkpoints.
    """
    original_torch_load = torch.load

    @functools.wraps(original_torch_load)
    def unsafe_torch_load_wrapper(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_torch_load(*args, **kwargs)

    try:
        torch.load = unsafe_torch_load_wrapper
        yield
    finally:
        torch.load = original_torch_load

def load_analysis_tools():
    """Loads and initializes all necessary models for the analysis pipeline."""
    print("Initializing analysis tools...")
    sentiment_model_name = 'distilbert/distilbert-base-uncased-finetuned-sst-2-english'
    sentiment_analysis_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
    sentiment_analysis_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name, model_max_len=512)
    sentiment_analysis = pipeline("sentiment-analysis", model=sentiment_analysis_model, tokenizer=sentiment_analysis_tokenizer)

    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Construct path relative to this file's location to make it robust
    current_dir = os.path.dirname(os.path.abspath(__file__))
    face_landmarker_path = os.path.join(current_dir, 'lib', 'face_landmarker_v2_with_blendshapes.task')
    if not os.path.exists(face_landmarker_path):
        raise FileNotFoundError(f"Face landmarker model not found at {face_landmarker_path}. Please ensure the path is correct.")
    base_options = python.BaseOptions(model_asset_path=face_landmarker_path)
    options = vision.FaceLandmarkerOptions(base_options=base_options, output_face_blendshapes=True, output_facial_transformation_matrixes=True, num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)
    print("Analysis tools initialized.")
    return sentiment_analysis, sentence_model, detector

def load_models_for_attribute_understanding(mllm_choice):
    """Loads a multi-modal large language model (MLLM) based on the provided choice."""
    if mllm_choice not in MLLM_CONFIGS:
        raise ValueError(f"Unsupported MLLM choice: {mllm_choice}")

    config = MLLM_CONFIGS[mllm_choice]
    model_class_name = config['model_class']
    model_class = globals()[model_class_name]
    
    # Common parameters for most HF models
    load_params = {
        'torch_dtype': torch.bfloat16,
        'low_cpu_mem_usage': True,
        'trust_remote_code': True,
        # 'load_in_8bit': True  # Disabled due to a common GLIBC incompatibility with bitsandbytes.
        # The pre-compiled bitsandbytes library may require a newer version of GLIBC
        # than is available on some systems, causing a crash. Disabling 8-bit loading
        # avoids this issue but will increase VRAM usage. If you have a compatible
        # environment, you can re-enable this for lower memory consumption.
    }

    load_params.update(config.get('params', {}))

    model = model_class.from_pretrained(config['path'], **load_params).eval()
    if 'device_placement' in config:
        model = model.to(config['device_placement'])

    tokenizer = AutoTokenizer.from_pretrained(config['path'], trust_remote_code=True, use_fast=False)
    processor = AutoProcessor.from_pretrained(config['path']) if 'llava' in mllm_choice else None
    gen_config = config.get('gen_config')

    return model, tokenizer, gen_config, processor

def load_models_for_eval(pretrained_model, checkpoints_dir):
    """Loads a vision-language model (VLM) for evaluation based on the provided model name."""
    if pretrained_model not in VLM_CONFIGS:
        raise ValueError(f"Unsupported pretrained_model for evaluation: {pretrained_model}")

    config = VLM_CONFIGS[pretrained_model]
    model_type = config['type']

    # Placeholders for return values
    tokenizer = None
    processor = None
    image_transform = None
    _curv = None
    preprocess = None

    if model_type == 'open_clip':
        with unsafe_torch_load_context():
            model, _, preprocess = create_model_and_transforms(config['model_name'], pretrained=config['pretrained'])
        if config.get('checkpoint'):
            checkpoint_path = os.path.join(checkpoints_dir, config['checkpoint'])
            # The checkpoint might have been saved with weights_only=False, so we need the context here too.
            with unsafe_torch_load_context():
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        model = model.to(device)
        
        # Wrap preprocess to ensure tensor is on the correct device
        _preprocess = preprocess
        def preprocess(image):
            return _preprocess(image).to(device)

    elif model_type == 'huggingface_clip':
        model = AutoModel.from_pretrained(config['model_id'])
        tokenizer = AutoTokenizer.from_pretrained(config['model_id'])
        processor = AutoProcessor.from_pretrained(config['model_id'])

    elif model_type == 'hycoclip':
        # The hycoclip library has a dependency on a `distributed` utils file
        # that may be missing. We provide a mock implementation for single-process
        # execution if the module is not found.
        import sys
        import types
        
        # Add the outer hycoclip directory to sys.path so the inner hycoclip package can be imported directly
        current_dir = os.path.dirname(os.path.abspath(__file__))
        hycoclip_path = os.path.join(current_dir, 'hycoclip')
        if hycoclip_path not in sys.path:
            sys.path.insert(0, hycoclip_path)
            
        try:
            import hycoclip.utils.distributed as dist
        except ModuleNotFoundError:
            class DummyDistributed:
                def is_main_process(self): return True
                def get_rank(self): return 0
                def get_world_size(self): return 1
                def barrier(self): pass
            
            if 'hycoclip.utils' not in sys.modules:
                sys.modules['hycoclip.utils'] = types.ModuleType('hycoclip.utils')
            sys.modules['hycoclip.utils.distributed'] = DummyDistributed()

        from hycoclip.config import LazyConfig, LazyFactory
        from hycoclip.utils.checkpointing import CheckpointManager
        from hycoclip.tokenizer import Tokenizer

        model_name_base = config['model_name_base']
        config_path = f'hycoclip/configs/train_{model_name_base}_vit_b.py'
        checkpoint_path = f'hycoclip/checkpoints/{model_name_base}_vit_b.pth'
        _C_TRAIN = LazyConfig.load(config_path)
        model = LazyFactory.build_model(_C_TRAIN, device).eval()
        with torch.serialization.safe_globals(HYCOCLIP_SAFE_GLOBALS):
            CheckpointManager(model=model).load(checkpoint_path)
        
        _tokenizer_instance = Tokenizer()
        def tokenizer(text):
            tokens = _tokenizer_instance(text)
            if isinstance(tokens, list):
                if tokens and isinstance(tokens[0], torch.Tensor):
                    tokens = torch.stack(tokens)
                else:
                    tokens = torch.tensor(tokens, dtype=torch.long)
            if tokens.dim() == 1:
                tokens = tokens.unsqueeze(0)
            
            # Pad to context length 77
            if tokens.shape[1] < 77:
                padding = torch.zeros((tokens.shape[0], 77 - tokens.shape[1]), dtype=tokens.dtype)
                tokens = torch.cat([tokens, padding], dim=1)
            elif tokens.shape[1] > 77:
                tokens = tokens[:, :77]
                
            return tokens.to(device)
            
        _image_transform = transforms.Compose([transforms.Resize((224, 224), transforms.InterpolationMode.BICUBIC), transforms.ToTensor()])
        def image_transform(image):
            return _image_transform(image).to(device)
            
        _curv = model.curv.exp()

    # Return a consistent dictionary of variables
    return {
        "model": model,
        "tokenizer": tokenizer,
        "processor": processor,
        "image_transform": image_transform,
        "_curv": _curv,
        "preprocess": preprocess,
    }