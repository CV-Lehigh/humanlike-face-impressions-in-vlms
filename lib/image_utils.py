import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from . import data_processing as dp

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    """Builds a standard image transformation pipeline."""
    return transforms.Compose([
        transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        transforms.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Finds the best matching aspect ratio from a set of targets."""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """Preprocesses an image by splitting it into patches of the closest aspect ratio."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def intervl2_image_preprocess(image, input_size=448, max_num=12):
    """Prepares an image for the InternVL2 model."""
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    return torch.stack(pixel_values).to(torch.bfloat16).cuda()

def load_and_mask_image(image_file, detector, mllm_choice=None):
    """Loads an image and creates masked versions for different facial regions."""
    from .config import MLLM_CONFIGS  # Local import to avoid potential circular dependencies
    try:
        image = Image.open(image_file).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_file}")
        return None, False

    masked_images = {}
    for region in ['eyes', 'nose', 'mouth', 'eyebrows']:
        masked_img = dp.apply_eye_mask(image, detector, [region])
        if masked_img is None:
            print(f"Warning: Failed to apply mask for region '{region}' on image {image_file}.")
            return None, False
        masked_images[region] = masked_img

    images_to_process = {'original': image, **masked_images}
    processed_images = {}
    
    mllm_config = MLLM_CONFIGS.get(mllm_choice, {})
    use_dynamic_preprocess = mllm_config.get('preprocess') == 'dynamic'

    if use_dynamic_preprocess:
        for name, img in images_to_process.items():
            processed_images[name] = intervl2_image_preprocess(img)
    else:
        processed_images = images_to_process

    return (processed_images['original'], processed_images['eyes'], processed_images['nose'], 
            processed_images['mouth'], processed_images['eyebrows']), True