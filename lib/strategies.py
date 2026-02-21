import torch
from hycoclip.hycoclip import lorentz as L
from MetaCLIP.src.open_clip.tokenizer import tokenize
from . import data_processing as dp

device = (torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu"))

class VLMStrategy:
    """Base class for a Vision-Language Model evaluation strategy."""
    def encode_image(self, model, image, **kwargs):
        raise NotImplementedError
    def encode_text(self, model, texts, **kwargs):
        raise NotImplementedError
    def calculate_similarity(self, image_features, text_features, **kwargs):
        raise NotImplementedError

class HuggingFaceClipStrategy(VLMStrategy):
    """Strategy for standard HuggingFace CLIP models."""
    def encode_image(self, model, image, **kwargs):
        processor = kwargs['processor']
        return dp.clip_image_preprocess(image, processor, model)
    def encode_text(self, model, texts, **kwargs):
        tokenizer = kwargs['tokenizer']
        inputs = tokenizer(texts, padding=True, return_tensors="pt")
        text_features = model.get_text_features(**inputs)
        return text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    def calculate_similarity(self, image_features, text_features, **kwargs):
        return torch.nn.functional.cosine_similarity(text_features, image_features)

class HycoClipStrategy(VLMStrategy):
    """Strategy for HycoCLIP/MeruCLIP models with hyperbolic geometry."""
    def encode_image(self, model, image, **kwargs):
        image_transform = kwargs['image_transform']
        return model.encode_image(image_transform(image).unsqueeze(0).to(model.device), project=True)
    def encode_text(self, model, texts, **kwargs):
        tokenizer = kwargs['tokenizer']
        # HycoCLIP tokenizer processes one text at a time
        text_tokens = [tokenizer(text) for text in texts]
        return model.encode_text(torch.cat(text_tokens), project=True)
    def calculate_similarity(self, image_features, text_features, **kwargs):
        _curv = kwargs['_curv']
        return L.pairwise_inner(text_features, image_features, _curv).squeeze()

class OpenClipStrategy(VLMStrategy):
    """Strategy for OpenCLIP-based models like MetaCLIP and FaRL."""
    def encode_image(self, model, image, **kwargs):
        preprocess = kwargs['preprocess']
        # MetaCLIP uses a special helper, FaRL uses the standard preprocess
        if 'metaclip' in kwargs.get('vlm_model_name', ''):
            return dp.metaclip_image_preprocess(image, preprocess, model)
        else:
            return model.encode_image(preprocess(image).unsqueeze(0).to(device))
    def encode_text(self, model, texts, **kwargs):
        text = tokenize(texts).to(device)
        text_features = model.encode_text(text)
        return text_features / text_features.norm(dim=-1, keepdim=True)
    def calculate_similarity(self, image_features, text_features, **kwargs):
        return torch.nn.functional.cosine_similarity(text_features, image_features)

STRATEGY_MAP = {
    'huggingface_clip': HuggingFaceClipStrategy(),
    'hycoclip': HycoClipStrategy(),
    'open_clip': OpenClipStrategy(),
}
