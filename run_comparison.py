import os
import sys

# necessary to access the intermediate attention layers of ViT architecture
os.environ["TIMM_FUSED_ATTN"] = '0'
# add folder that contains script for running transformer explainability to sys path
sys.path.append(os.path.join(os.path.dirname(__file__), 'transformer_explainability'))

import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import timm
from PIL import Image
import numpy as np
from lime import lime_image
from matplotlib import pyplot as plt
from transformer_explainability.vit_explaination_generator import LRP 
import vit_rollout
from transformer_explainability.vit_lrp import vit_base_patch16_224 as vit_LRP
import cv2
from tqdm import tqdm
from captum.attr import IntegratedGradients, KernelShap, GradientShap
from skimage.segmentation import slic, quickshift
from scipy import ndimage
import pickle
import seaborn as sns
import itertools
import random
import json

# custom Dataset class to load Imagenette Dataset
class ImagenetteDataset(Dataset):
    def __init__(self, transform, class_path_to_label, subset_size=None):
        self.root_path = os.path.join('data', 'imagenette2')
        class_paths = os.listdir(os.path.join(self.root_path, 'train'))
        self.data = []
        for class_path in class_paths:
            # these are the standard training images of ImageNet1k
            for split in ['train', 'val']:
                for img_path in os.listdir(os.path.join(self.root_path, split, class_path)):
                    self.data.append([os.path.join(self.root_path, split, class_path, img_path), class_path])
        self.transform = transform 
        self.class_path_to_label = class_path_to_label

        # option to only work on a subset
        if subset_size is not None and subset_size < len(self.data):
            self.data = random.sample(self.data, subset_size)

    def __len__(self):
        return len(self.data)    
    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = Image.open(img_path)
        img_tensor = self.transform(img)
        class_id = torch.tensor(int(self.class_path_to_label[class_name]))
        
        return img_tensor, class_id

# normalization between 0 and 1 of all values of an array
def norm_array(array: np.ndarray) -> np.ndarray:
    return (array - np.min(array)) / (np.max(array) - np.min(array))

# threshold all values of an array to either be 0 or 1 (create binary mask)
def threshold_array(array: np.ndarray) -> np.ndarray:
    _, array = cv2.threshold((array*255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    array[array == 255] = 1.0
    # it would also be possible to choose a percentile
    # array = np.where(array > np.percentile(array, percentile), 1.0, 0)
    return array

# takes an attribution map as input and outputs a binary mask
def create_binary_mask(attr_map: np.ndarray) -> np.ndarray:
    
    # we are only interested in parts that contribute positively to a given class
    attr_map = np.maximum(attr_map, 0)

    if np.ndim(attr_map) > 3:
        raise ValueError("Array has more than three axes.")
    elif np.ndim(attr_map) == 3:
        # summarize the values among the color channel axis 
        attr_map = np.mean(attr_map, axis=-1)

    # normalize values to be between 0 and 1
    attr_map = norm_array(attr_map)

    # apply a dynamic threshold to get a binary mask
    binary_mask = threshold_array(attr_map)

    return binary_mask

# in case of binary mask on pixel level, this function applies some smoothing
# techniques to create unified saliency regions
def postprocess_pixel_mask(mask, percentile=80):

    # connect closely located pixels and fill holes
    mask_closed = ndimage.grey_closing(mask, structure=np.ones((6, 6)))
    mask_smoothed = ndimage.binary_fill_holes(mask_closed)

    # search for connected components within the binary mask and label them
    connected_components, num_comp = ndimage.label(mask_smoothed, structure=np.ones((3,3)))
    # for each component, count the amount of pixels (0 is the background and thus excluded)
    counts = np.bincount(connected_components.ravel())[1:]
    # only keep regions which area is above a certain percentile 
    # (because we work with indices, we need to add 1)
    regions_above_threshold = np.where(counts > np.percentile(counts, percentile))[0] + 1  # 
    
    return np.isin(connected_components, regions_above_threshold) * 1.0

def get_attr_map_lime(model, device, lime_explainer, image, label):
    
    # Define the prediction function for the model
    def predict(img):
        img = img.transpose((0, -1, -2, -3)) # Change the image from HWC to CHW format
        img = torch.from_numpy(img).to(device)
        with torch.no_grad():
            outputs = model(img)
        return outputs.cpu().numpy()
    
    # Get the explanation for the image w.r.t. the actual class 
    # (i.e. not necessarily the predicted class)
    explanation = lime_explainer.explain_instance(
        image.squeeze(0).cpu().numpy().transpose(1,2,0), 
        predict,
        labels=label.cpu().numpy(),
        top_labels=None, 
        hide_color=0, 
        num_samples=1000,
        # progress_bar=False, # should be available
        )

    # The method provided by the libary returns the top-n features which is not really comparable
    # to other methods and yields in the problem of choosing n which is why we apply thresholding on
    # the values returned by the method
    # _, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)

    # This is just a quick way to get the raw explaination values of lime w.r.t. the actual class of shape (224,244) 
    attr_map = np.vectorize(dict(explanation.local_exp[label.cpu().numpy()[0]]).get)(explanation.segments)

    return create_binary_mask(attr_map)

def get_attr_map_ig(ig_explainer, image, label):

    # get the attribution map for the image
    attr_map = ig_explainer.attribute(image, target=label)
    attr_map = attr_map.squeeze().cpu().detach().numpy().transpose(1,2,0)

    return postprocess_pixel_mask(create_binary_mask(attr_map))

# Transformer Interpretability Beyond Attention Visualization
# https://github.com/hila-chefer/Transformer-Explainability
def get_attr_map_beyond_attention(attribution_generator, image, label): 

    # get attribution map
    attr_map = attribution_generator.generate_LRP(image, method="transformer_attribution", index=label).detach()
    # interpolate attribution map to size (244,244)
    # attr_map = torch.nn.functional.interpolate(attr_map.reshape(1, 1, 14, 14), scale_factor=16, mode='bilinear').reshape(224, 224).data.cpu().numpy()
    attr_map = cv2.resize(attr_map.reshape(14,14).cpu().numpy(), (224, 224))
    return create_binary_mask(attr_map)

# Attention Rollout
# https://github.com/jacobgil/vit-explain
def get_attr_map_attn_rollout(attn_rollout, image):

    attr_map = attn_rollout(image)

    attr_map = cv2.resize(attr_map, (224, 224))

    return create_binary_mask(attr_map)

# KernelSHAP
def get_attr_map_kernel_shap(kernel_shap_explainer, image, label, device):

    # segment the images using the quickshift algorithm (same as in LIME for comparibility)
    segments_slic = torch.from_numpy(
        quickshift(image.squeeze(0).cpu().numpy().transpose(1,2,0), kernel_size=4, max_dist=200, ratio=0.2)
    ).to(device)
    # slic(image.cpu().numpy().transpose(1,2,0), n_segments=100, compactness=10, sigma=0, start_label=0)

    attr_map = kernel_shap_explainer.attribute(
        inputs = image, 
        target=label, 
        feature_mask=segments_slic, 
        n_samples=1000, 
        show_progress=False
    )

    # convert tensor to numpy array and change format from CHW to HWC
    attr_map = attr_map.squeeze(0).cpu().numpy().transpose(1,2,0)

    return create_binary_mask(attr_map)

# GradientSHAP
def get_attr_map_gradient_shap(gradient_shap_explainer, image, label, device):

    attr_map = gradient_shap_explainer.attribute(
        inputs = image, 
        baselines=torch.zeros_like(image).to(device),
        target=label, 
        n_samples=50
    )

    # convert tensor to numpy array and change format from CHW to HWC
    attr_map = attr_map.squeeze(0).cpu().numpy().transpose(1,2,0)

    return postprocess_pixel_mask(create_binary_mask(attr_map))

def get_attr_maps(args, model, device, dataloader):
    
    # initialize dict to store the attribution maps for each method
    attr_maps = {method: [] for method in args.methods}

    # initialize the explainer modules
    lime_explainer = lime_image.LimeImageExplainer()
    ig_explainer = IntegratedGradients(model)

    model_LRP = vit_LRP(pretrained=True).to(device)
    model_LRP.eval()
    attribution_generator = LRP(model_LRP)

    attn_rollout = vit_rollout.VITAttentionRollout(model, attention_layer_name="attn_drop", head_fusion="max", discard_ratio=0.9)

    kernel_shap_explainer = KernelShap(model)
    gradient_shap_explainer = GradientShap(model)

    # iterate through the images
    for image, label in tqdm(dataloader, desc="Iterate through dataset:"):
        
        label, image = label.to(device), image.to(device)

        ## LIME ##
        attr_maps["LIME"].append(
            get_attr_map_lime(model, device, lime_explainer, image, label)
        )

        ## IntegratedGradients ##
        attr_maps["IntegratedGradients"].append(
            get_attr_map_ig(ig_explainer, image, label)
        )

        ## Transformer Explainability beyond Attention ##
        attr_maps["BeyondAttention"].append(
            get_attr_map_beyond_attention(attribution_generator, image, label)
        )

        ## Attention Rollout ##
        attr_maps["AttentionRollout"].append(
            get_attr_map_attn_rollout(attn_rollout, image)
        )

        ## KernelSHAP ##
        attr_maps["KernelSHAP"].append(
            get_attr_map_kernel_shap(kernel_shap_explainer, image, label, device)
        )

        ## GradientSHAP ##
        attr_maps["GradientSHAP"].append(
            get_attr_map_gradient_shap(gradient_shap_explainer, image, label, device)
        )

    # create a numpy array for each explanaition method
    for key, value in attr_maps.items():
        attr_maps[key] = np.stack(value, axis=0)

    if args.save_attr_maps:
        with open('attr_maps.pickle', 'wb') as f:
            pickle.dump(attr_maps, f)

    return attr_maps

def plot_test_images(args, dataloader, attr_maps, idx2label):
    
    image, label = next(iter(dataloader))

    fig, axes = plt.subplots(1, len(args.methods), figsize=(len(args.methods) * 16, 16))

    for idx, method in enumerate(args.methods):
        axes[idx].imshow(image[0].numpy().transpose(1,2,0))
        axes[idx].imshow(attr_maps[method][0], cmap='jet', alpha=0.7)
        axes[idx].set_title(f"Method: {method}", fontsize=25)
        axes[idx].axis('off')

    fig.suptitle(f"Class Label: {idx2label[label.numpy()[0]]}", fontsize=30)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("sample_attr_maps.png")

def calculate_metrics(args, attr_maps):
    def calculate_iou(masks1, masks2):
        intersection = np.logical_and(masks1, masks2)
        union = np.logical_or(masks1, masks2)
        iou_score = np.sum(intersection, axis=(1,2)) / np.sum(union, axis=(1,2))
        return np.mean(iou_score, axis=0)
    def calculate_dice_coefficient(masks1, masks2):
        intersection = np.logical_and(masks1, masks2)
        dice_coefficient = (2 * np.sum(intersection, axis=(1,2))) / (np.sum(masks1, axis=(1,2)) + np.sum(masks2, axis=(1,2)))
        return np.mean(dice_coefficient, axis=0)
    def calculate_pixel_accuracy(masks1, masks2):
        num_pixels = masks1.shape[-1] * masks1.shape[-2]
        pixel_accuracy = np.sum(masks1 == masks2, axis=(1,2)) / num_pixels
        return np.mean(pixel_accuracy, axis=0)

    dict_metrics = {
        "IoU": {},
        "DiceCoefficient": {},
        "PixelAccuracy": {},
    }

    # iterate through all possible combination of explainability method and calculate the metrics
    for pair in itertools.combinations(args.methods, r=2):
        dict_metrics["IoU"][(pair[0], pair[1])] = calculate_iou(attr_maps[pair[0]], attr_maps[pair[1]])
        dict_metrics["DiceCoefficient"][(pair[0], pair[1])] = calculate_dice_coefficient(attr_maps[pair[0]], attr_maps[pair[1]])
        dict_metrics["PixelAccuracy"][(pair[0], pair[1])] = calculate_pixel_accuracy(attr_maps[pair[0]], attr_maps[pair[1]])

    return dict_metrics 

def plot_metrics_heatmaps(args, dict_metrics):

    fig, axes = plt.subplots(1, len(dict_metrics), figsize=(len(dict_metrics) * 16, 16))

    # Iterate over each metric in the dict_metrics
    for idx, (metric_name, metric_data) in enumerate(dict_metrics.items()):
        # Convert the metric_data dictionary to a matrix
        matrix = [
            [
                metric_data.get(
                    (method1, method2), 
                    metric_data.get((method2, method1), 1.0)
                ) for method2 in args.methods
            ]
            for method1 in args.methods
        ]

        # Create heatmap using seaborn
        sns.heatmap(matrix, annot=True, cmap='YlGnBu', ax=axes[idx], annot_kws={"fontsize": 20})
        axes[idx].set_xticklabels(args.methods, rotation=45, ha='right', fontsize=20)
        axes[idx].set_yticklabels(args.methods, rotation=0, fontsize=20)
        axes[idx].set_title(metric_name, fontsize=25)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("metrics_heatmaps.png")

if __name__ == "__main__":

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Working on device: {DEVICE}")
    BATCH_SIZE = 1 # most explainability methods do not support batches

    # Define the argument parser
    parser = argparse.ArgumentParser(description='Explainability Method Comparing Script')
    parser.add_argument(
        '--methods', 
        nargs='+', 
        default=['LIME', 'IntegratedGradients', 'BeyondAttention', 'AttentionRollout', 'KernelSHAP', 'GradientSHAP'], 
        choices=['LIME', 'IntegratedGradients', 'BeyondAttention', 'AttentionRollout', 'KernelSHAP', 'GradientSHAP'],
        help='Specify the explainability method to compare'
    )
    parser.add_argument('--idx_sample_img', type=int, default=0, help='Index of a sample image to plot for each method')
    parser.add_argument('--save_attr_maps', action='store_true', help='Whether to save the computed attribution maps for each method')
    parser.add_argument('--subset_size', type=int, default=None, help='Index of a sample image to plot for each method')

    args = parser.parse_args()

    # load pre-trained ViT model using the timm libary
    model = timm.create_model('vit_base_patch16_224.orig_in21k_ft_in1k', pretrained=True).to(DEVICE)
    model.eval()

    # string representation of ImageNet1k class labels
    with open(os.path.join('data', 'imagenet-class-labels.json'), "r") as json_file:
        class_idx = json.load(json_file)
    idx2label = [class_idx[str(k)] for k in range(len(class_idx))]

    # class label mapping from imagefolder to class label in ImageNet1k
    class_path_to_label = {
        "n01440764": "0",
        "n02102040": "217",
        "n02979186": "482",
        "n03000684": "491",
        "n03028079": "497",
        "n03394916": "566", 
        "n03417042": "569",
        "n03425413": "571",
        "n03445777": "574",
        "n03888257": "701"
    }

    # Define the transformation to apply to the images
    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),         
        transforms.Lambda(lambda x: x.expand(3, -1, -1) if x.shape[0] == 1 else x),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    dataset = ImagenetteDataset(transform, class_path_to_label, args.subset_size)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    if os.path.isfile('./attr_maps.pickle'):
        with open('attr_maps.pickle', 'rb') as f:
            attr_maps = pickle.load(f)
        print(f"Loaded attribution maps from cache")
    else: 
        print("File 'attr_maps.pickle' does not exist in the current directory. Computing...")
        attr_maps = get_attr_maps(args, model, DEVICE, dataloader)

    plot_test_images(args, dataloader, attr_maps, idx2label)

    dict_metrics = calculate_metrics(args, attr_maps)

    plot_metrics_heatmaps(args, dict_metrics)