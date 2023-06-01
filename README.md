# Vision Transformer (ViT) Explainability Comparison

This projects aims to provide a comparision of state-of-the-art (SOTA) explainability method applied to the ViT architecture. It builds upon existing work (https://arxiv.org/pdf/2202.01602.pdf, https://arxiv.org/pdf/2105.03287.pdf) that compares explainability method, but other than the papers mentioned before, it focuses on the area computer vision. 

## Methodology

The repository uses the ViT model as underlying architecture to conduct the experiments and loads pre-trained weights using the `timm` libary. The following explainability methods are comapred:

- LIME
- KernelSHAP
- GradientSHAP
- IntegratedGradients
- Attention Rollout
- Transformer Interpretability Beyond Attention Visualization (https://github.com/hila-chefer/Transformer-Explainability)

For each method (and each sample), a binary image salience map is computed using a unifed scheme:

1. In order to extract only the regions which _positively_ contribute to the target class, the raw attribution scores are clipped to [0, ∞)
2. Average the attribution scores over the three color channels (only applicable to some methods, e.g., IntegratedGradients)
3. Normalize values to be within [0, 1]
4. Create a binary mask using binary and otso thresholding

Some method work based on superpixels (e.g., LIME), and other method work on low-scale attention maps which are upscaled (e.g., AttentionRollout). For these method the areas of attribution are mostly unified and homogenous whereas for other methods which work based on pixel contribution (e.g., Integrated Gradient), the pixel-grained areas are scattered and scarse. In order to provide a better comparison, post-processing is applied to these binary masks:

5. Using a 6 x 6 kernel, greyscale-closing and binary hole filling is applied on the binary masks
6. Further post-processing detects all unfied regions of the binary masks and only keeps the components with an area above the 80th percentile

## Usage

### Installation

In order for the scripts to work, install the packages specified in the `requirements.txt` file. 

### Reconstructing results

Simply execute the script `run_comparison.py`. You can pass the following arguments depending on your needs:

- `--methods <...>`: Select the methods you want to run the comparison on (default: all)
- `--idx_sample_img <idx> `: The script will create saliency maps for one example image; with this argument you can pass the desired index (default: 0)
- `--save_attr_maps`: Whether to save the computed attribution map as .pickle-file.