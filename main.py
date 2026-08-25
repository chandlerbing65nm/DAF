# Standard
import os, time, argparse, copy, json

# Third-party
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Local
from adapt import get_method
from utils import segmentation_datasets
from utils.metrics import intersect_and_union, process_metrics, total_area_to_metrics
from utils.misc import set_global_seeds, save_configuration, aggregate_pred_patches
from datetime import datetime

_original_print = print

def print(*args, **kwargs):
    """Module-level print that prepends a [YYYY-MM-DD HH:MM:SS] timestamp."""
    ts = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    _original_print(ts, *args, **kwargs)


def str2bool(value):
    if isinstance(value, bool):
        return value

    value = value.lower()
    if value == 'true':
        return True
    if value == 'false':
        return False
    raise argparse.ArgumentTypeError("Expected 'True' or 'False'")


def validate_token_merge_args(args):
    if not getattr(args, 'token_merge', False):
        return

    supported_methods = {'tent', 'mlmp', 'method'}
    if args.method not in supported_methods:
        raise ValueError(
            f"--token_merge is only supported for methods {sorted(supported_methods)}; got {args.method}."
        )

    supported_ovss_types = {'clip', 'sclip', 'naclip'}
    if args.ovss_type not in supported_ovss_types:
        raise ValueError(
            f"--token_merge is only supported for ovss types {sorted(supported_ovss_types)}; got {args.ovss_type}."
        )

    if args.merge_type != 'algm':
        raise ValueError(f"Unsupported --merge_type {args.merge_type}. Only 'algm' is implemented.")

    if not str(args.ovss_backbone).startswith('ViT'):
        raise ValueError("--token_merge currently supports only ViT image encoders.")

    if len(args.algm_layers) == 0:
        raise ValueError("--algm_layers must contain at least one layer index.")
    if any(layer < 1 for layer in args.algm_layers):
        raise ValueError("--algm_layers must contain positive 1-based layer indices.")

    if len(args.algm_window_size) != 2:
        raise ValueError("--algm_window_size expects exactly two integers.")
    if any(side <= 0 for side in args.algm_window_size):
        raise ValueError("--algm_window_size values must be positive integers.")


"""TODO List:
- end of main_segmentation is necessary?


- datasets
    we can have download datasets script? not sure
    but we can have a dataset.md
- repo:
    - add a section (supported methods=> list them and add reference to each of them)
    - we can talk about how to perform all methods (including No Adapt)
    - in acknowledgements, we can say athat we modified the original CLIP code to "ovss/clip/model.py" to be able to perform segmentation 
"""



def argparser():
    parser = argparse.ArgumentParser(
        description="Test-Time Adaptation of Vision-Language Models for Open-Vocabulary Semantic Segmentation"
    )
    
    # ----------------------------------------
    # I/O Directories
    # ----------------------------------------
    parser.add_argument(
        '--save_dir',
        type=str,
        default='save/',
        help='Directory to save model weights and results'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='.data/',
        help='Root directory for datasets'
    )
    parser.add_argument(
        '--prompt_dir',
        type=str,
        default='',
        help='Path to the YAML file containing prompt templates'
    )
    
    # ----------------------------------------
    # Dataset Settings
    # ----------------------------------------
    parser.add_argument(
        '--dataset',
        type=str,
        default='COCOStuffDataset',
        choices=(
            'COCOStuffDataset', 'COCOStuff10kDataset', 'COCOObjectDataset', 'CityscapesDataset', 'CityscapesFoggyDataset', 'BDD100kDataset', 'ADE20kDataset', 'LoveDADataset', 'ACDCDataset',
            'CarlaDataset',
            'DarkZurichDataset',
            'DrivingDataset',
            'PascalVOC20Dataset', 'PascalVOC21Dataset',
            'PascalContext59Dataset', 'PascalContext60Dataset', 'SUIM6Dataset', 'SUIM5Dataset',
            'DUTUSEG5Dataset', 'DUTUSEG4Dataset'
        ),
        help='Which dataset to load'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=0,
        help='Number of data-loading workers'
    )
    parser.add_argument(
        '--init_resize',
        nargs='+',
        type=int,
        default=None,
        help=(
            'Resize images before patch extraction. '
            'Order doesn’t matter (e.g., (560,448) same as (448,560)). '
            'If None, use original size (batch_size must be 1).'
        )
    )
    parser.add_argument(
        '--patch_size',
        nargs='+',
        type=int,
        default=None,
        help='Size of each image patch after resize (model input size)'
    )
    parser.add_argument(
        '--patch_stride',
        type=int,
        default=None,
        help='Stride for extracting patches'
    )
    parser.add_argument(
        '--corruptions_list',
        nargs='+',
        type=str,
        default=None,
        help='List of corruptions to apply for robustness (e.g., gaussian, motion_blur)'
    )
    
    # ----------------------------------------
    # Model Settings
    # ----------------------------------------
    parser.add_argument(
        '--ovss_type',
        type=str,
        default='ncalip',
        help='Open-Vocabulary Semantic Segmentation type (e.g., nacalip, clip, clip, etc.)'
    )
    parser.add_argument(
        '--ovss_backbone',
        type=str,
        default='ViT-B/32',
        help='CLIP vision backbone (e.g., ViT-B/32, ViT-L/14)'
    )
    parser.add_argument(
        '--token_merge',
        type=str2bool,
        default=False,
        help='Enable ViT image-token merging in the OVSS image encoder (True/False)'
    )
    parser.add_argument(
        '--merge_type',
        type=str,
        default='algm',
        help="Token merge variant to use. Only 'algm' is currently implemented."
    )
    parser.add_argument(
        '--algm_layers',
        nargs='+',
        type=int,
        default=[1, 7],
        help='1-based transformer layer indices where ALGM token merging is applied'
    )
    parser.add_argument(
        '--algm_threshold',
        type=float,
        default=0.8,
        help='Threshold used by the first ALGM local merge stage'
    )
    parser.add_argument(
        '--algm_window_size',
        nargs=2,
        type=int,
        default=[2, 2],
        help='Window size for the first ALGM local merge stage'
    )
    parser.add_argument(
        '--class_extensions',
        action='store_true',
        help='Enable dataset-specific class extensions if available'
    )
    
    # ----------------------------------------
    # Adaptation / Training Settings
    # ----------------------------------------
    parser.add_argument(
        '--adapt',
        action='store_true',
        help='Enable test-time adaptation'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='tent',
        help='Adaptation method name (e.g., mlmp watt, tent)'
    )
    parser.add_argument(
        '--reset_mode',
        type=str,
        default='episodic',
        choices=('episodic', 'normal', 'continual'),
        help='Reset behavior for TTA'
    )
    parser.add_argument(
        '--lifelong',
        type=str,
        default='None',
        choices=('None', 'shuffle_domain_pround', 'shuffle_domain_pbatch', 'recurring_domain_pround'),
        help='Lifelong domain scheduling mode'
    )
    parser.add_argument(
        '--lifelong_rnds',
        type=int,
        default=3,
        help='Number of lifelong rounds'
    )
    parser.add_argument(
        '--domain_gen',
        type=str2bool,
        default=False,
        help='If True, adapt on all but the last domain_gen_num domains and directly evaluate the last domains with adapted weights'
    )
    parser.add_argument(
        '--domain_gen_num',
        type=int,
        default=5,
        help='Number of last domains to hold out from adaptation for domain generalization evaluation'
    )
    parser.add_argument(
        '--batch_size', '--batch-size',
        type=int,
        default=1,
        dest='batch_size',
        help='Batch size for adaptation'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate for adaptation optimizer'
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default='adam',
        choices=('adam', 'adamw', 'sgd'),
        help='Optimizer for adaptation'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=1,
        help='Number of adaptation iterations per batch'
    )
    parser.add_argument(
        '--trials',
        type=int,
        default=1,
        help='Number of experimental repetitions'
    )
    
    # ----------------------------------------
    # Debug / Misc
    # ----------------------------------------
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--plot_loss',
        action='store_true',
        help='Plot the loss curve (averaged over batches and seeds)'
    )
    parser.add_argument(
        '--runtime_calculation',
        action='store_true',
        help='Calculate the runtime of adaptation and evaluation'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    parser.add_argument(
        '--save_demo',
        type=lambda x: x.lower() == 'true',
        default=False,
        help='Save K random prediction overlays per corruption domain (True/False)'
    )
    parser.add_argument(
        '--save_k',
        type=int,
        default=5,
        help='Number of random demo images to save per corruption domain when --save_demo is True'
    )

    return parser

def add_method_specific_args(parser, method):
    '''
    Add method-specific arguments to the parser
    '''
    if method == 'mlmp':
        parser.add_argument(
            '--vision_outputs',
            nargs='+',
            type=int,
            default=(-1,),
            help='Indices of vision layers to extract outputs from'
        )
        parser.add_argument(
            '--prompt_integration',
             type=str, default='loss', 
             help='If we have different prompt templates, how to integrate them (loss-level or text-level). MLMP uses loss-level integration by default.'
             )
        parser.add_argument(
            '--alpha_cls', 
            type=float, 
            default=1.0, 
            help='Weight for the classification loss in MLMP'
            )
        parser.add_argument(
            '--loss_ent',
            type=lambda x: x.lower() == 'true',
            default=True,
            help='Enable entropy minimization loss (True/False)'
        )
        parser.add_argument(
            '--lamb_ent',
            type=float,
            default=1.0,
            help='Lambda multiplier for entropy minimization loss'
        )
        parser.add_argument(
            '--loss_div',
            type=lambda x: x.lower() == 'true',
            default=False,
            help='Enable class-wise diversity loss to prevent model collapse (True/False)'
        )
        parser.add_argument(
            '--lamb_div',
            type=float,
            default=1.0,
            help='Lambda multiplier for diversity loss'
        )
        parser.add_argument(
            '--loss_cmac',
            type=lambda x: x.lower() == 'true',
            default=False,
            help='Enable Cross-Modal Anchor Consistency loss (True/False)'
        )
        parser.add_argument(
            '--lamb_cmac',
            type=float,
            default=1.0,
            help='Lambda multiplier for CMAC loss'
        )
        parser.add_argument(
            '--loss_src_cons',
            type=lambda x: x.lower() == 'true',
            default=False,
            help='Enable source-logit KL consistency loss (True/False)'
        )
        parser.add_argument(
            '--lamb_src_cons',
            type=float,
            default=1.0,
            help='Lambda multiplier for source consistency loss'
        )
        parser.add_argument(
            '--loss_feat_cons',
            type=lambda x: x.lower() == 'true',
            default=False,
            help='Enable source feature consistency loss (True/False)'
        )
        parser.add_argument(
            '--lamb_feat_cons',
            type=float,
            default=1.0,
            help='Lambda multiplier for feature consistency loss'
        )
        parser.add_argument(
            '--feat_cons_type',
            type=str,
            default='cosine',
            choices=['cosine', 'l2'],
            help='Distance type for feature consistency loss (cosine or l2)'
        )
        parser.add_argument(
            '--module_safs',
            type=lambda x: x.lower() == 'true',
            default=False,
            help='Enable Source-Anchored Feature Salience sample filtering (True/False)'
        )
        parser.add_argument(
            '--alpha_safs',
            type=float,
            default=0.5,
            help='Margin parameter for SAFS adaptive threshold'
        )
        parser.add_argument(
            '--diag_safs',
            type=lambda x: x.lower() == 'true',
            default=False,
            help='Enable SAFS diagnostic justification analysis (True/False)'
        )
        parser.add_argument(
            '--diag_cmac',
            type=lambda x: x.lower() == 'true',
            default=False,
            help='Enable CMAC diagnostic justification analysis (True/False)'
        )
        parser.add_argument(
            '--diag_div',
            type=lambda x: x.lower() == 'true',
            default=False,
            help='Enable DIV diagnostic justification analysis (True/False)'
        )

    elif method == 'tent':
        parser.add_argument(
            '--loss_ent',
            type=lambda x: x.lower() == 'true',
            default=True,
            help='Enable entropy minimization loss (True/False)'
        )
        parser.add_argument(
            '--lamb_ent',
            type=float,
            default=1.0,
            help='Lambda multiplier for entropy minimization loss'
        )
        parser.add_argument(
            '--loss_div',
            type=lambda x: x.lower() == 'true',
            default=False,
            help='Enable class-wise diversity loss to prevent model collapse (True/False)'
        )
        parser.add_argument(
            '--lamb_div',
            type=float,
            default=1.0,
            help='Lambda multiplier for diversity loss'
        )
        parser.add_argument(
            '--loss_cmac',
            type=lambda x: x.lower() == 'true',
            default=False,
            help='Enable Cross-Modal Anchor Consistency loss (True/False)'
        )
        parser.add_argument(
            '--lamb_cmac',
            type=float,
            default=1.0,
            help='Lambda multiplier for CMAC loss'
        )
        parser.add_argument(
            '--loss_src_cons',
            type=lambda x: x.lower() == 'true',
            default=False,
            help='Enable source-logit KL consistency loss (True/False)'
        )
        parser.add_argument(
            '--lamb_src_cons',
            type=float,
            default=1.0,
            help='Lambda multiplier for source consistency loss'
        )
        parser.add_argument(
            '--loss_feat_cons',
            type=lambda x: x.lower() == 'true',
            default=False,
            help='Enable source feature consistency loss (True/False)'
        )
        parser.add_argument(
            '--lamb_feat_cons',
            type=float,
            default=1.0,
            help='Lambda multiplier for feature consistency loss'
        )
        parser.add_argument(
            '--feat_cons_type',
            type=str,
            default='cosine',
            choices=['cosine', 'l2'],
            help='Distance type for feature consistency loss (cosine or l2)'
        )
        parser.add_argument(
            '--module_safs',
            type=lambda x: x.lower() == 'true',
            default=False,
            help='Enable Source-Anchored Feature Salience sample filtering (True/False)'
        )
        parser.add_argument(
            '--alpha_safs',
            type=float,
            default=0.5,
            help='Margin parameter for SAFS adaptive threshold'
        )
        parser.add_argument(
            '--diag_safs',
            type=lambda x: x.lower() == 'true',
            default=False,
            help='Enable SAFS diagnostic justification analysis (True/False)'
        )
        parser.add_argument(
            '--diag_cmac',
            type=lambda x: x.lower() == 'true',
            default=False,
            help='Enable CMAC diagnostic justification analysis (True/False)'
        )
        parser.add_argument(
            '--diag_div',
            type=lambda x: x.lower() == 'true',
            default=False,
            help='Enable DIV diagnostic justification analysis (True/False)'
        )

    elif method == 'cotta':
        parser.add_argument(
            '--cotta_mt',
            type=float,
            default=0.999,
            help='EMA teacher momentum for model update'
        )
        parser.add_argument(
            '--cotta_rst',
            type=float,
            default=0.01,
            help='Stochastic restore probability for adapted parameters'
        )
        parser.add_argument(
            '--cotta_ap',
            type=float,
            default=0.92,
            help='Anchor confidence threshold for augmentation-averaged prediction'
        )
        parser.add_argument(
            '--cotta_n_augmentations',
            type=int,
            default=32,
            help='Number of augmentations for augmentation-averaged prediction'
        )
        parser.add_argument(
            '--loss_ent',
            type=lambda x: x.lower() == 'true',
            default=True,
            help='Enable entropy minimization loss (True/False)'
        )
        parser.add_argument(
            '--lamb_ent',
            type=float,
            default=1.0,
            help='Lambda multiplier for entropy minimization loss'
        )
        parser.add_argument(
            '--loss_div',
            type=lambda x: x.lower() == 'true',
            default=False,
            help='Enable class-wise diversity loss to prevent model collapse (True/False)'
        )
        parser.add_argument(
            '--lamb_div',
            type=float,
            default=1.0,
            help='Lambda multiplier for diversity loss'
        )
        parser.add_argument(
            '--loss_cmac',
            type=lambda x: x.lower() == 'true',
            default=False,
            help='Enable Cross-Modal Anchor Consistency loss (True/False)'
        )
        parser.add_argument(
            '--lamb_cmac',
            type=float,
            default=1.0,
            help='Lambda multiplier for CMAC loss'
        )
        parser.add_argument(
            '--module_safs',
            type=lambda x: x.lower() == 'true',
            default=False,
            help='Enable SAFS sample filtering module (True/False)'
        )
        parser.add_argument(
            '--alpha_safs',
            type=float,
            default=0.5,
            help='Alpha parameter for SAFS adaptive threshold'
        )
        parser.add_argument(
            '--diag_safs',
            type=lambda x: x.lower() == 'true',
            default=False,
            help='Enable SAFS diagnostic justification analysis (True/False)'
        )
        parser.add_argument(
            '--diag_cmac',
            type=lambda x: x.lower() == 'true',
            default=False,
            help='Enable CMAC diagnostic justification analysis (True/False)'
        )
        parser.add_argument(
            '--diag_div',
            type=lambda x: x.lower() == 'true',
            default=False,
            help='Enable DIV diagnostic justification analysis (True/False)'
        )

    elif method == 'watt':
        parser.add_argument(
            '--watt_l', 
            default=2, 
            type=int, 
            help='Number of adaptation iterations for each text embedding before weight averaging'
            )
        parser.add_argument('--watt_m', 
            default=5, 
            type=int, 
            help='Number of repetitions of the adaptation and weight averaging process'
            )

    elif method == 'clipartt':
        parser.add_argument(
            '--clipartt_k', 
            default=3, 
            type=int, 
            help='Number of classes taken to build the area pseudo label'
            )

    elif method == 'method':
        parser.add_argument(
            '--train_imag_norm',
            type=lambda x: x.lower() == 'true',
            default=True,
            help='Train LayerNorm layers in the visual encoder (True/False)'
        )
        parser.add_argument(
            '--last_imag_k_norm',
            type=int,
            default=0,
            help='Train only the last K visual transformer blocks LN layers. 0 = all LN layers.'
        )
        parser.add_argument(
            '--train_imag_attn',
            type=lambda x: x.lower() == 'true',
            default=False,
            help='Train attention layers in the visual encoder (True/False)'
        )
        parser.add_argument(
            '--last_imag_k_attn',
            type=int,
            default=0,
            help='Train only the last K visual transformer blocks attn layers. 0 = all attn layers.'
        )
        parser.add_argument(
            '--train_text_norm',
            type=lambda x: x.lower() == 'true',
            default=False,
            help='Train LayerNorm layers in the text encoder (True/False)'
        )
        parser.add_argument(
            '--last_text_k_norm',
            type=int,
            default=0,
            help='Train only the last K text transformer blocks LN layers. 0 = all LN layers.'
        )
        parser.add_argument(
            '--loss_ent',
            type=lambda x: x.lower() == 'true',
            default=True,
            help='Enable entropy minimization loss (True/False)'
        )
        parser.add_argument(
            '--lamb_ent',
            type=float,
            default=1.0,
            help='Lambda multiplier for entropy minimization loss'
        )
        parser.add_argument(
            '--loss_div',
            type=lambda x: x.lower() == 'true',
            default=False,
            help='Enable class-wise diversity loss to prevent model collapse (True/False)'
        )
        parser.add_argument(
            '--lamb_div',
            type=float,
            default=1.0,
            help='Lambda multiplier for diversity loss'
        )
        parser.add_argument(
            '--loss_aug_cons',
            type=lambda x: x.lower() == 'true',
            default=False,
            help='Enable pixel-wise augmentation consistency loss (True/False)'
        )
        parser.add_argument(
            '--lamb_aug_cons',
            type=float,
            default=1.0,
            help='Lambda multiplier for augmentation consistency loss'
        )
        parser.add_argument(
            '--loss_src_cons',
            type=lambda x: x.lower() == 'true',
            default=False,
            help='Enable source model consistency loss (True/False)'
        )
        parser.add_argument(
            '--lamb_src_cons',
            type=float,
            default=1.0,
            help='Lambda multiplier for source model consistency loss'
        )
        parser.add_argument(
            '--loss_cmac',
            type=lambda x: x.lower() == 'true',
            default=False,
            help='Enable Cross-Modal Anchor Consistency loss (True/False). Confidence-weighted KL divergence exploiting CLIP dual-encoder geometry.'
        )
        parser.add_argument(
            '--lamb_cmac',
            type=float,
            default=1.0,
            help='Lambda multiplier for CMAC loss'
        )
        parser.add_argument(
            '--loss_pba',
            type=lambda x: x.lower() == 'true',
            default=False,
            help='Enable Prototypical Barycenter Alignment loss (True/False). Geometry-aware replacement for entropy minimization exploiting CLIP dual-encoder hyperspherical geometry.'
        )
        parser.add_argument(
            '--lamb_pba',
            type=float,
            default=1.0,
            help='Lambda multiplier for Prototypical Barycenter Alignment loss'
        )
        parser.add_argument(
            '--updownsample',
            type=float,
            default=1.0,
            help='Control prediction upsampling vs GT downsampling ratio (0.0=native token res, 1.0=full patch res, 0.5=halfway)'
        )
        parser.add_argument(
            '--prompt_average',
            type=lambda x: x.lower() == 'true',
            default=False,
            help='Enable averaging of prompt encodings during adaptation and evaluation (True/False). Only effective when --prompt_dir is set.'
        )
        parser.add_argument(
            '--cons_type',
            type=str,
            default='sym_kl',
            choices=['sym_kl', 'for_kl', 'rev_kl'],
            help='Consistency loss type: symmetric KL (sym_kl), forward KL (for_kl), or reverse KL (rev_kl). Applied to aug_cons and src_cons losses only.'
        )
        parser.add_argument(
            '--module_safs',
            type=lambda x: x.lower() == 'true',
            default=False,
            help='Enable Source-Anchored Feature Salience sample filtering (True/False). Filters uninformative samples by comparing angular feature shift between adapted and frozen source models in CLIP hyperspherical space.'
        )
        parser.add_argument(
            '--alpha_safs',
            type=float,
            default=0.5,
            help='Margin parameter for SAFS adaptive threshold (tau = mu - alpha * sigma). Higher = more lenient (keep more samples), lower = stricter filtering.'
        )
        parser.add_argument(
            '--diag_safs',
            type=lambda x: x.lower() == 'true',
            default=False,
            help='Enable SAFS diagnostic justification analysis (True/False). Collects per-sample feature shift, entropy, prediction change, and prediction agreement stats during adaptation without filtering. Run with --module_safs False.'
        )
        parser.add_argument(
            '--diag_cmac',
            type=lambda x: x.lower() == 'true',
            default=False,
            help='Enable CMAC diagnostic justification analysis (True/False). Collects per-step prototype drift metrics (assigned sim, unassigned sim, class distribution) during adaptation. Run with --loss_cmac False --loss_div False --loss_ent True.'
        )
        parser.add_argument(
            '--diag_div',
            type=lambda x: x.lower() == 'true',
            default=False,
            help='Enable DIV diagnostic justification analysis (True/False). Collects per-step class prediction collapse metrics (class fractions, HHI, active class count) during adaptation. Run with --loss_div False --loss_cmac False --module_safs False --loss_ent True.'
        )

    
    return parser


CLIP_MEAN_DEMO = [122.7709, 116.7460, 104.0937]
CLIP_STD_DEMO = [68.5005, 66.6322, 70.3232]


def get_demo_indices(dataset_len, k, seed):
    """Return sorted list of K deterministic random indices."""
    k = min(k, dataset_len)
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(dataset_len, size=k, replace=False).tolist())


def get_demo_method_name(args):
    """Return the method subfolder name for demo save path."""
    if not args.adapt:
        return 'source'
    if args.method == 'tent' and getattr(args, 'loss_div', False) and getattr(args, 'loss_cmac', False) and getattr(args, 'module_safs', False):
        return 'daf_t'
    if args.method == 'mlmp' and getattr(args, 'loss_div', False) and getattr(args, 'loss_cmac', False) and getattr(args, 'module_safs', False):
        return 'daf_m'
    return args.method


def colorize_mask(mask, palette):
    """Convert a class index mask (H, W) to an RGB image using the dataset palette."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx in range(len(palette)):
        rgb[mask == cls_idx] = palette[cls_idx]
    return rgb


def save_demo_overlay(img_tensor, mask, palette, save_path, alpha=0.5):
    """Save an alpha-blended overlay of a colorized mask on the de-normalized input image.

    Args:
        img_tensor: [C, H, W] normalized tensor (CLIP mean/std, RGB order)
        mask: [H, W] numpy array or tensor of class indices
        palette: list of [R, G, B] per class
        save_path: output file path
        alpha: overlay transparency (0=all image, 1=all mask)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # De-normalize: img = img * std + mean  (back to 0-255 RGB)
    mean = torch.tensor(CLIP_MEAN_DEMO).view(3, 1, 1)
    std = torch.tensor(CLIP_STD_DEMO).view(3, 1, 1)
    img_denorm = img_tensor.detach().cpu() * std + mean
    img_np = img_denorm.clamp(0, 255).numpy().astype(np.uint8)  # [3, H, W]
    img_np = img_np.transpose(1, 2, 0)  # [H, W, 3]

    # Colorize mask
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask_rgb = colorize_mask(mask, palette)  # [H, W, 3]

    # Alpha blend
    overlay = ((1 - alpha) * img_np + alpha * mask_rgb).astype(np.uint8)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(img_np.shape[1] / 100, img_np.shape[0] / 100), dpi=100)
    ax.imshow(overlay)
    ax.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def save_demo_input(img_tensor, save_path):
    """Save the de-normalized input image."""
    mean = torch.tensor(CLIP_MEAN_DEMO).view(3, 1, 1)
    std = torch.tensor(CLIP_STD_DEMO).view(3, 1, 1)
    img_denorm = img_tensor.detach().cpu() * std + mean
    img_np = img_denorm.clamp(0, 255).numpy().astype(np.uint8)
    img_np = img_np.transpose(1, 2, 0)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    from PIL import Image
    Image.fromarray(img_np).save(save_path)


def main(args):

    validate_token_merge_args(args)

    # Save the configuration settings
    save_configuration(args)

    # Start the timer
    start_time = time.time()

    # Set the device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create the save directory if it doesn't exist
    all_results_path = os.path.join(args.save_dir, "results.txt")
    os.makedirs(os.path.dirname(all_results_path), exist_ok=True)

    if args.domain_gen and args.lifelong != 'None':
        raise ValueError("--domain_gen only works when --lifelong is None")

    if args.domain_gen:
        run_domain_gen(args, device, start_time, all_results_path)
        return

    if args.lifelong != 'None':
        run_lifelong(args, device, start_time, all_results_path)
        return

    # create necessary variables
    all_results = dict()
    headers = "mIoU, mDice, mAcc"
    adapt_time_all_corr = []
    eval_time_all_corr = []
    continual_methods = None
    domain_summary = []
    demo_indices = None

    for c_idx, corruption in enumerate(args.corruptions_list):

        data_loader, org_classes = segmentation_datasets.prepare_data(args.dataset, args.data_dir, args.init_resize,
                                                                  args.patch_size, args.patch_stride, corruption=corruption, 
                                                                  batch_size=args.batch_size, num_workers=args.workers,
                                                                  shuffle=not getattr(args, 'save_demo', False))

        if getattr(args, 'save_demo', False) and demo_indices is None:
            demo_indices = set(get_demo_indices(len(data_loader.dataset), args.save_k, args.seed))
            print(f"+++ Demo: saving {len(demo_indices)} images per corruption")

        # Check if the extensions of classes should be used
        if args.class_extensions and data_loader.dataset.class_extensions is not None:
            ext_classes = data_loader.dataset.class_extensions
            args.classes = ext_classes
            print(f"\n+++ Using class extensions")
            print(f"+++ The number of classes [no extension]: {len(org_classes)}")
            print(f"+++ The number of classes after extension:  {len(ext_classes)}")

        else:
            args.classes = org_classes
            print(f"\n+++ The number of classes [no extension]: {len(org_classes)}")

        num_org_classes = len(org_classes)
        ignore_index = data_loader.dataset.ignore_index # the index of the ignore label in the segmentation map

        if args.reset_mode == 'episodic':
            adapt_method = get_method(args, device)
        elif args.reset_mode == 'continual' and continual_methods is None:
            continual_methods = [get_method(args, device) for _ in range(args.trials)]

        # Results path
        c_results_path = os.path.join(args.save_dir, f"{c_idx:02}_{corruption}", "results.txt")
        os.makedirs(os.path.dirname(c_results_path), exist_ok=True)

        miou_seeds = []
        dice_seeds = []
        acc_seeds = []
        loss_seed_report = []
        safs_stats_per_corruption = []
        per_class_iou_seeds = []
        pct_metrics_seeds = []
        acdc_cond_miou_seeds = {}
        acdc_cond_dice_seeds = {}
        acdc_cond_acc_seeds = {}

        for t in range(args.trials):
            if args.reset_mode == 'normal':
                adapt_method = get_method(args, device)
            elif args.reset_mode == 'continual':
                adapt_method = continual_methods[t]

            safs_len_before = len(adapt_method.safs_stats) if hasattr(adapt_method, 'safs_stats') else 0
            results = []
            sample_conditions = []
            loss_batch_report = []
            global_sample_idx = 0
            for batch_idx, data in tqdm(enumerate(data_loader), total=len(data_loader)):

                if args.debug and batch_idx == 10: 
                    break

                inputs = data['img_patches'] 
                labels = data['gt_patches']  
                original_gts = data['gt'] 

                patch_grid_shape = data['meta']['patch_grid_shape'] 
                image_shapes = data['meta']['img_shape']
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                if args.reset_mode == 'episodic':
                    adapt_method.reset()
                
                # perform adaptation
                if args.adapt:
                    if getattr(args, 'diag_safs', False) or getattr(args, 'diag_div', False):
                        diag_labels = labels
                        diag_ignore = ignore_index
                        loss_iter_report = adapt_method.adapt(inputs, diag_labels=diag_labels, diag_ignore_index=diag_ignore)
                    else:
                        loss_iter_report = adapt_method.adapt(inputs)
                    loss_batch_report.append(loss_iter_report)

                # perform evaluation 
                with torch.no_grad():
                    patch_preds = adapt_method.evaluate(inputs)

                # compute eval scale for updownsample support
                eval_size = getattr(adapt_method, 'eval_size', args.patch_size[0])
                eval_scale = eval_size / args.patch_size[0]

                # aggregate the predictions to construct the final segmentation map for each image in the batch
                if args.init_resize:
                    if eval_scale < 1.0:
                        scaled_patch_size = (round(args.patch_size[0] * eval_scale), round(args.patch_size[1] * eval_scale))
                        scaled_patch_stride = round(args.patch_stride * eval_scale)
                        scaled_img_shapes = [(round(h * eval_scale), round(w * eval_scale)) for h, w in image_shapes]
                        reconstructed_preds = aggregate_pred_patches(patch_preds, patch_grid_shape, scaled_img_shapes, scaled_patch_size, scaled_patch_stride)
                    else:
                        reconstructed_preds = aggregate_pred_patches(patch_preds, patch_grid_shape, image_shapes, args.patch_size, args.patch_stride)
                else:
                    reconstructed_preds = patch_preds

                
                # calculate the metrics for each image in the batch (since the images may have different sizes)
                for idx, (pd, gt) in enumerate(zip(reconstructed_preds, original_gts)):

                    # get the predictions
                    pd = pd.softmax(dim=0) # [num_org_classes, H, W]

                    # fix the extensions indices
                    if args.class_extensions and data_loader.dataset.class_extensions is not None:
                        ext_to_real_cls_indx = torch.Tensor(data_loader.dataset.extentions_to_real_class_idx).to(torch.int64).to(device)
                        num_cls, num_queries = max(ext_to_real_cls_indx) + 1, len(ext_to_real_cls_indx)
                        ext_to_real_cls_indx = torch.nn.functional.one_hot(ext_to_real_cls_indx)
                        ext_to_real_cls_indx = ext_to_real_cls_indx.T.view(num_cls, num_queries, 1, 1)
                        pd = pd.unsqueeze(0)
                        pd = (pd * ext_to_real_cls_indx).max(1)[0]


                    pd = pd.argmax(dim=0)  # [H, W]
                    pd = pd.to(gt.device)  

                    # get the ground truth
                    gt = gt[0]             # [H, W]
                    if eval_scale < 1.0:
                        target_h, target_w = scaled_img_shapes[idx]
                        gt = torch.nn.functional.interpolate(
                            gt.unsqueeze(0).unsqueeze(0).float(), size=(target_h, target_w), mode='nearest'
                        ).squeeze(0).squeeze(0).long()
                    # metric calculation
                    results.append(intersect_and_union(pd, gt, num_org_classes, ignore_index))

                    if args.dataset == "DrivingDataset" and corruption == "acdc":
                        img_path = data['meta']['img_path'][idx]
                        parts = img_path.split('/')
                        condition = next((parts[i+1] for i, p in enumerate(parts) if p == 'rgb_anon' and i+1 < len(parts)), None)
                        sample_conditions.append(condition)

                    # save demo overlay
                    if getattr(args, 'save_demo', False) and global_sample_idx in demo_indices:
                        demo_dir = os.path.join('diagnostics', 'x_demo', args.dataset, corruption, get_demo_method_name(args))
                        img_tensor = data['img'][idx]
                        save_demo_overlay(img_tensor, pd, data_loader.dataset.metainfo['palette'],
                                          os.path.join(demo_dir, f"pred_{global_sample_idx:04d}.png"))
                        if not args.adapt:
                            save_demo_overlay(img_tensor, gt, data_loader.dataset.metainfo['palette'],
                                              os.path.join(demo_dir, f"gt_{global_sample_idx:04d}.png"))
                            save_demo_input(img_tensor,
                                            os.path.join(demo_dir, f"input_{global_sample_idx:04d}.png"))
                    global_sample_idx += 1
               
            
            # Convert the batch report to a numpy array for easier averaging
            loss_batch_report = np.array(loss_batch_report)

            # Average loss over batches for each iteration
            avg_loss_per_iter = np.mean(loss_batch_report, axis=0)  # Shape: [10] (for 10 iterations)
            loss_seed_report.append(avg_loss_per_iter)

            
            metrics = process_metrics(results, org_classes)
            miou_seeds.append(metrics['mIoU'])
            dice_seeds.append(metrics['mDice'])
            acc_seeds.append(metrics['mAcc'])
            per_class_iou_seeds.append(compute_per_class_iou(results))
            pct_metrics_seeds.append(compute_metrics_at_pct(results))
            print(f"Results for corruption: {corruption}, trial: {t}, mIoU:  {metrics['mIoU']}, mDice:  {metrics['mDice']}, mAcc: {metrics['mAcc']}")

            if args.dataset == "DrivingDataset" and corruption == "acdc" and sample_conditions:
                for cond in ('fog', 'night', 'rain', 'snow'):
                    cond_indices = [i for i, c in enumerate(sample_conditions) if c == cond]
                    if cond_indices:
                        cond_metrics = summarize_results([results[i] for i in cond_indices])
                        acdc_cond_miou_seeds.setdefault(cond, []).append(cond_metrics['mIoU'])
                        acdc_cond_dice_seeds.setdefault(cond, []).append(cond_metrics['mDice'])
                        acdc_cond_acc_seeds.setdefault(cond, []).append(cond_metrics['mAcc'])
                        print(f"  ACDC real_{cond} (trial {t}): mIoU {cond_metrics['mIoU']:.2f}, mDice {cond_metrics['mDice']:.2f}, mAcc {cond_metrics['mAcc']:.2f}")


            # Saving the weights if self.weights_track list is not empty
            if adapt_method.model.weights_track:
                weights_path = os.path.join(args.save_dir, "weights")
                
                weights = adapt_method.model.weights_track
                weights = np.hstack(weights)
                os.makedirs(weights_path, exist_ok=True)
                
                # save to a file
                np.save(os.path.join(weights_path, f"{corruption}_s{t}.npy"), np.array(weights))

                # plot and save the mean and std of weights across the layers
                weights_mean = np.mean(weights, axis=1)
                weights_std = np.std(weights, axis=1)
                plt.figure()
                plt.errorbar(range(len(weights_mean)), weights_mean, yerr=weights_std, fmt='o')
                plt.xlabel('Layer')
                plt.ylabel('Weight')
                plt.title(f'Mean and Std of Weights for {corruption}')
                plt.savefig(os.path.join(weights_path, f"{corruption}_s{t}.png"))
                plt.close()

                # reset the weights_track list
                adapt_method.model.weights_track = []

        
        if hasattr(adapt_method, 'safs_stats'):
            safs_len_after = len(adapt_method.safs_stats)
            safs_stats_per_corruption.extend(adapt_method.safs_stats[safs_len_before:safs_len_after])
        safs_filtered, safs_unfiltered = aggregate_safs_stats(safs_stats_per_corruption)
        if safs_filtered is not None:
            print(f"SAFS: filtered={safs_filtered}, unfiltered={safs_unfiltered}")

        if getattr(args, 'diag_safs', False) and hasattr(adapt_method, 'diag_safs_stats'):
            miou_mean_for_diag = np.array(miou_seeds).mean() if miou_seeds else None
            write_diag_safs_logs(args, corruption, miou_mean_for_diag, adapt_method.diag_safs_stats)

        if getattr(args, 'diag_cmac', False) and hasattr(adapt_method, 'diag_cmac_stats') and adapt_method.diag_cmac_stats:
            write_diag_cmac_logs(args, corruption, adapt_method.diag_cmac_stats)
            adapt_method.diag_cmac_step_offset += len(adapt_method.diag_cmac_stats)
            adapt_method.diag_cmac_stats = []

        if getattr(args, 'diag_div', False) and hasattr(adapt_method, 'diag_div_stats') and adapt_method.diag_div_stats:
            write_diag_div_logs(args, corruption, adapt_method.diag_div_stats)
            adapt_method.diag_div_step_offset += len(adapt_method.diag_div_stats)
            adapt_method.diag_div_stats = []

        miou_mean, miou_std = np.array(miou_seeds).mean(), np.array(miou_seeds).std()
        dice_mean, dice_std = np.array(dice_seeds).mean(), np.array(dice_seeds).std()
        acc_mean, acc_std = np.array(acc_seeds).mean(), np.array(acc_seeds).std()

        print(f"mIoU:  {miou_mean:.2f},{miou_std:.2f}")
        print(f"mDice: {dice_mean:.2f},{dice_std:.2f}")
        print(f"mAcc:  {acc_mean:.2f},{acc_std:.2f}")

        if args.dataset == "DrivingDataset" and corruption == "acdc" and acdc_cond_miou_seeds:
            print("  --- ACDC per-condition ---")
            for cond in ('fog', 'night', 'rain', 'snow'):
                if cond in acdc_cond_miou_seeds:
                    cond_miou = np.mean(acdc_cond_miou_seeds[cond])
                    cond_dice = np.mean(acdc_cond_dice_seeds[cond])
                    cond_acc = np.mean(acdc_cond_acc_seeds[cond])
                    print(f"  real_{cond}: mIoU {cond_miou:.2f}, mDice {cond_dice:.2f}, mAcc {cond_acc:.2f}")

        c_results_print = f"{miou_mean:.2f} +/- {miou_std:.2f}, {dice_mean:.2f} +/- {dice_std:.2f}, {acc_mean:.2f} +/- {acc_std:.2f}"
        with open(c_results_path, 'w') as f:        
            f.write(headers + "\n")
            f.write(c_results_print)
            if args.dataset == "DrivingDataset" and corruption == "acdc" and acdc_cond_miou_seeds:
                f.write("\nACDC per-condition:\n")
                for cond in ('fog', 'night', 'rain', 'snow'):
                    if cond in acdc_cond_miou_seeds:
                        cond_miou = np.mean(acdc_cond_miou_seeds[cond])
                        cond_dice = np.mean(acdc_cond_dice_seeds[cond])
                        cond_acc = np.mean(acdc_cond_acc_seeds[cond])
                        f.write(f"real_{cond}, {cond_miou:.2f}, {cond_dice:.2f}, {cond_acc:.2f}\n")    

        loss_mean, loss_inc, loss_dec = compute_loss_stats(loss_seed_report)

        avg_per_class_iou = np.mean(per_class_iou_seeds, axis=0) if per_class_iou_seeds else None
        avg_pct_metrics = {}
        if pct_metrics_seeds:
            for pct in pct_metrics_seeds[0]:
                avg_pct_metrics[pct] = {
                    'mIoU': np.mean([s[pct]['mIoU'] for s in pct_metrics_seeds]),
                    'mDice': np.mean([s[pct]['mDice'] for s in pct_metrics_seeds]),
                    'mAcc': np.mean([s[pct]['mAcc'] for s in pct_metrics_seeds]),
                }

        all_results[corruption] = c_results_print
        domain_summary.append({
            'corruption': corruption,
            'mIoU_mean': miou_mean,
            'mIoU_std': miou_std,
            'mDice_mean': dice_mean,
            'mDice_std': dice_std,
            'mAcc_mean': acc_mean,
            'mAcc_std': acc_std,
            'loss_mean': loss_mean,
            'loss_increase': loss_inc,
            'loss_decrease': loss_dec,
            'safs_filtered': safs_filtered,
            'safs_unfiltered': safs_unfiltered,
            'per_class_iou': avg_per_class_iou,
            'pct_metrics': avg_pct_metrics,
        })

        # Convert the seed report to a numpy array and average over trials (seeds)
        loss_seed_report = np.array(loss_seed_report)
        avg_loss_over_seeds = np.mean(loss_seed_report, axis=0)  # Shape: [10] (averaged over seeds)

        if args.plot_loss and args.adapt:
            # Plot the averaged loss for this corruption
            plt.figure()
            plt.plot(range(1, len(avg_loss_over_seeds)+1), avg_loss_over_seeds)
            plt.xlabel('Iteration')
            plt.ylabel('Average Loss')
            plt.title(f'Average Loss per Iteration for {corruption}')
            
            # Save the plot in the specified directory
            save_path = os.path.join(args.save_dir, f'loss_{corruption}.png')
            plt.savefig(save_path)
            plt.close()

        # if the runtime calculation is enabled, we will have access to adapt_method.adapt_times and adapt_method.eval_times (each one contains a list of times)
        if args.runtime_calculation:
            if args.adapt:
                mean_adapt_time = np.mean(adapt_method.adapt_times[20:])
                std_adapt_time = np.std(adapt_method.adapt_times[20:])
            else:
                mean_adapt_time = 0
                std_adapt_time = 0
            
            mean_eval_time = np.mean(adapt_method.eval_times[20:])
            std_eval_time = np.std(adapt_method.eval_times[20:])

            mean_total_time = mean_adapt_time + mean_eval_time

            run_time_txt = f"{corruption}, {mean_adapt_time:0.3f} +/- {std_adapt_time:0.3f}, {mean_eval_time:0.3f} +/- {std_eval_time:0.3f}, {mean_total_time:0.3f}"
            print(run_time_txt)
            
            runtime_save_dir = os.path.join(args.save_dir, "runtime.txt")
            with open(runtime_save_dir, 'a+') as f:
                f.write(run_time_txt + "\n")

            adapt_time_all_corr.append(mean_adapt_time)
            eval_time_all_corr.append(mean_eval_time)

    total_duration = time.time() - start_time
    mean_duration_per_seed = total_duration / args.trials
    gpu_info = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    print("\n===== Per-domain Summary =====")
    for domain_metrics in domain_summary:
        loss_str = ""
        if domain_metrics.get('loss_mean') is not None:
            loss_str = (
                f", Loss {domain_metrics['loss_mean']:.4f}"
                f", Loss+ {domain_metrics['loss_increase']:.4f}"
                f", Loss- {domain_metrics['loss_decrease']:.4f}"
            )
        safs_str = ""
        if domain_metrics.get('safs_filtered') is not None:
            safs_str = (
                f", SAFS filtered {domain_metrics['safs_filtered']}"
                f", unfiltered {domain_metrics['safs_unfiltered']}"
            )
        print(
            f"{domain_metrics['corruption']}: "
            f"mIoU {domain_metrics['mIoU_mean']:.2f} +/- {domain_metrics['mIoU_std']:.2f}, "
            f"mDice {domain_metrics['mDice_mean']:.2f} +/- {domain_metrics['mDice_std']:.2f}, "
            f"mAcc {domain_metrics['mAcc_mean']:.2f} +/- {domain_metrics['mAcc_std']:.2f}"
            f"{loss_str}{safs_str}"
        )

    overall_miou_mean = np.mean([domain_metrics['mIoU_mean'] for domain_metrics in domain_summary])
    overall_mdice_mean = np.mean([domain_metrics['mDice_mean'] for domain_metrics in domain_summary])
    overall_macc_mean = np.mean([domain_metrics['mAcc_mean'] for domain_metrics in domain_summary])

    print("===== Overall Mean Summary =====")
    print(
        f"Overall mean across domains: "
        f"mIoU {overall_miou_mean:.2f}, "
        f"mDice {overall_mdice_mean:.2f}, "
        f"mAcc {overall_macc_mean:.2f}"
    )

    # Overall mean per class across domains
    per_class_iou_domains = [
        dm['per_class_iou'] for dm in domain_summary
        if dm.get('per_class_iou') is not None
    ]
    if per_class_iou_domains:
        overall_per_class_iou = np.mean(per_class_iou_domains, axis=0)
        print("Overall mean per class across domains:")
        for cls_name, cls_iou in zip(org_classes, overall_per_class_iou):
            print(f"  {cls_name}: {cls_iou:.2f}")

    # Overall mean per dataset % across domains
    pct_domains = [
        dm['pct_metrics'] for dm in domain_summary
        if dm.get('pct_metrics')
    ]
    if pct_domains:
        print("Overall mean per dataset % across domains:")
        for pct in sorted(pct_domains[0].keys()):
            pct_miou = np.mean([dm[pct]['mIoU'] for dm in pct_domains])
            pct_mdice = np.mean([dm[pct]['mDice'] for dm in pct_domains])
            pct_macc = np.mean([dm[pct]['mAcc'] for dm in pct_domains])
            print(f"  {int(pct * 100)}%: mIoU {pct_miou:.2f}, mDice {pct_mdice:.2f}, mAcc {pct_macc:.2f}")

    with open(all_results_path, 'w') as f:
        f.write(headers + "\n")
        for corruption, results in all_results.items():
            f.write(f"{corruption}, {results}\n")
        f.write(f"\nGPU: {gpu_info}\n")
        f.write(f"Total Duration (s): {total_duration:.2f}\n")
        f.write(f"Mean Duration per Seed (s): {mean_duration_per_seed:.2f}\n")
        if per_class_iou_domains:
            f.write("\nOverall mean per class across domains:\n")
            for cls_name, cls_iou in zip(org_classes, overall_per_class_iou):
                f.write(f"  {cls_name}: {cls_iou:.2f}\n")
        if pct_domains:
            f.write("\nOverall mean per dataset % across domains:\n")
            for pct in sorted(pct_domains[0].keys()):
                pct_miou = np.mean([dm[pct]['mIoU'] for dm in pct_domains])
                pct_mdice = np.mean([dm[pct]['mDice'] for dm in pct_domains])
                pct_macc = np.mean([dm[pct]['mAcc'] for dm in pct_domains])
                f.write(f"  {int(pct * 100)}%: mIoU {pct_miou:.2f}, mDice {pct_mdice:.2f}, mAcc {pct_macc:.2f}\n")


def run_domain_gen(args, device, start_time, all_results_path):
    headers = "mIoU, mDice, mAcc"
    all_results = dict()
    domain_summary = []
    adapt_time_all_corr = []
    eval_time_all_corr = []

    holdout_count = min(args.domain_gen_num, len(args.corruptions_list))
    adapt_corruptions = set(args.corruptions_list[:-holdout_count]) if holdout_count > 0 else set(args.corruptions_list)
    eval_corruptions = list(args.corruptions_list[-holdout_count:]) if holdout_count > 0 else []

    domain_infos = []
    for c_idx, corruption in enumerate(args.corruptions_list):
        domain_infos.append(prepare_domain_info(args, device, corruption, c_idx))

    args.classes = domain_infos[0]['classes']

    demo_info = None
    if getattr(args, 'save_demo', False):
        demo_indices = set(get_demo_indices(len(domain_infos[0]['data_loader'].dataset), args.save_k, args.seed))
        demo_info = {
            'indices': demo_indices,
            'palette': domain_infos[0]['data_loader'].dataset.metainfo['palette'],
            'global_sample_idx': 0,
        }
        print(f"+++ Demo: saving {len(demo_indices)} images per corruption")

    continual_methods = None
    if args.reset_mode == 'continual':
        continual_methods = [get_method(args, device) for _ in range(args.trials)]

    for t in range(args.trials):
        if args.reset_mode == 'continual':
            adapt_method = continual_methods[t]
        else:
            adapt_method = get_method(args, device)

        for domain_idx, domain_info in enumerate(domain_infos):

            corruption = domain_info['corruption']
            should_adapt_domain = corruption in adapt_corruptions

            if args.reset_mode == 'normal' and should_adapt_domain:
                adapt_method.reset()

            results = []
            loss_batch_report = []
            weights_batch_report = []
            adapt_len_before = len(adapt_method.adapt_times) if args.runtime_calculation and args.adapt else 0
            eval_len_before = len(adapt_method.eval_times) if args.runtime_calculation else 0
            safs_len_before = len(adapt_method.safs_stats) if hasattr(adapt_method, 'safs_stats') else 0
            if demo_info is not None:
                demo_info['global_sample_idx'] = 0

            for batch_idx, data in tqdm(enumerate(domain_info['data_loader']), total=len(domain_info['data_loader'])):
                if args.debug and batch_idx == 10:
                    break

                if args.reset_mode == 'episodic' and should_adapt_domain:
                    adapt_method.reset()

                batch_results, loss_iter_report, _, _, weights = process_single_batch(
                    args,
                    device,
                    adapt_method,
                    data,
                    domain_info,
                    demo_info=demo_info,
                ) if should_adapt_domain and args.adapt else process_single_batch_no_adapt(
                    args,
                    device,
                    adapt_method,
                    data,
                    domain_info,
                    demo_info=demo_info,
                )

                results.extend(batch_results)
                if loss_iter_report is not None:
                    loss_batch_report.append(loss_iter_report)

                if weights:
                    weights_batch_report.extend(weights)

            if hasattr(adapt_method, 'safs_stats'):
                safs_len_after = len(adapt_method.safs_stats)
                domain_info['safs_stats_list'].extend(adapt_method.safs_stats[safs_len_before:safs_len_after])

            metrics = process_metrics(results, domain_info['org_classes'])
            domain_info['miou_seeds'].append(metrics['mIoU'])
            domain_info['dice_seeds'].append(metrics['mDice'])
            domain_info['acc_seeds'].append(metrics['mAcc'])
            domain_info['per_class_iou_seeds'].append(compute_per_class_iou(results))
            domain_info['pct_metrics_seeds'].append(compute_metrics_at_pct(results))
            print(f"Results for corruption: {corruption}, trial: {t}, mIoU:  {metrics['mIoU']}, mDice:  {metrics['mDice']}, mAcc: {metrics['mAcc']}")

            if loss_batch_report:
                loss_batch_report = np.array(loss_batch_report)
                avg_loss_per_iter = np.mean(loss_batch_report, axis=0)
                domain_info['loss_seed_report'].append(avg_loss_per_iter)

            if weights_batch_report:
                weights_path = os.path.join(args.save_dir, "weights")

                weights = weights_batch_report
                weights = np.hstack(weights)
                os.makedirs(weights_path, exist_ok=True)

                np.save(os.path.join(weights_path, f"{corruption}_s{t}.npy"), np.array(weights))

                weights_mean = np.mean(weights, axis=1)
                weights_std = np.std(weights, axis=1)
                plt.figure()
                plt.errorbar(range(len(weights_mean)), weights_mean, yerr=weights_std, fmt='o')
                plt.xlabel('Layer')
                plt.ylabel('Weight')
                plt.title(f'Mean and Std of Weights for {corruption}')
                plt.savefig(os.path.join(weights_path, f"{corruption}_s{t}.png"))
                plt.close()

            if args.runtime_calculation:
                if args.adapt:
                    adapt_times = adapt_method.adapt_times[adapt_len_before:]
                    mean_adapt_time = np.mean(adapt_times[20:]) if len(adapt_times) > 20 else (np.mean(adapt_times) if len(adapt_times) > 0 else 0)
                    std_adapt_time = np.std(adapt_times[20:]) if len(adapt_times) > 20 else (np.std(adapt_times) if len(adapt_times) > 0 else 0)
                else:
                    mean_adapt_time = 0
                    std_adapt_time = 0

                eval_times = adapt_method.eval_times[eval_len_before:]
                mean_eval_time = np.mean(eval_times[20:]) if len(eval_times) > 20 else (np.mean(eval_times) if len(eval_times) > 0 else 0)
                std_eval_time = np.std(eval_times[20:]) if len(eval_times) > 20 else (np.std(eval_times) if len(eval_times) > 0 else 0)

                mean_total_time = mean_adapt_time + mean_eval_time

                run_time_txt = f"{corruption}, {mean_adapt_time:0.3f} +/- {std_adapt_time:0.3f}, {mean_eval_time:0.3f} +/- {std_eval_time:0.3f}, {mean_total_time:0.3f}"
                print(run_time_txt)

                runtime_save_dir = os.path.join(args.save_dir, "runtime.txt")
                with open(runtime_save_dir, 'a+') as f:
                    f.write(run_time_txt + "\n")

                adapt_time_all_corr.append(mean_adapt_time)
                eval_time_all_corr.append(mean_eval_time)

    for domain_info in domain_infos:
        corruption = domain_info['corruption']
        miou_mean = np.array(domain_info['miou_seeds']).mean()
        miou_std = np.array(domain_info['miou_seeds']).std()
        dice_mean = np.array(domain_info['dice_seeds']).mean()
        dice_std = np.array(domain_info['dice_seeds']).std()
        acc_mean = np.array(domain_info['acc_seeds']).mean()
        acc_std = np.array(domain_info['acc_seeds']).std()

        safs_filtered, safs_unfiltered = aggregate_safs_stats(domain_info['safs_stats_list'])
        print(f"mIoU:  {miou_mean:.2f},{miou_std:.2f}")
        print(f"mDice: {dice_mean:.2f},{dice_std:.2f}")
        print(f"mAcc:  {acc_mean:.2f},{acc_std:.2f}")
        if safs_filtered is not None:
            print(f"SAFS: filtered={safs_filtered}, unfiltered={safs_unfiltered}")

        if getattr(args, 'diag_safs', False) and hasattr(adapt_method, 'diag_safs_stats') and adapt_method.diag_safs_stats:
            write_diag_safs_logs(args, corruption, miou_mean, adapt_method.diag_safs_stats)

        if getattr(args, 'diag_cmac', False) and hasattr(adapt_method, 'diag_cmac_stats') and adapt_method.diag_cmac_stats:
            write_diag_cmac_logs(args, corruption, adapt_method.diag_cmac_stats)
            adapt_method.diag_cmac_step_offset += len(adapt_method.diag_cmac_stats)
            adapt_method.diag_cmac_stats = []

        if getattr(args, 'diag_div', False) and hasattr(adapt_method, 'diag_div_stats') and adapt_method.diag_div_stats:
            write_diag_div_logs(args, corruption, adapt_method.diag_div_stats)
            adapt_method.diag_div_step_offset += len(adapt_method.diag_div_stats)
            adapt_method.diag_div_stats = []

        c_results_print = f"{miou_mean:.2f} +/- {miou_std:.2f}, {dice_mean:.2f} +/- {dice_std:.2f}, {acc_mean:.2f} +/- {acc_std:.2f}"
        with open(domain_info['c_results_path'], 'w') as f:
            f.write(headers + "\n")
            f.write(c_results_print)

        loss_mean, loss_inc, loss_dec = compute_loss_stats(domain_info['loss_seed_report'])

        avg_per_class_iou = np.mean(domain_info['per_class_iou_seeds'], axis=0) if domain_info['per_class_iou_seeds'] else None
        avg_pct_metrics = {}
        if domain_info['pct_metrics_seeds']:
            for pct in domain_info['pct_metrics_seeds'][0]:
                avg_pct_metrics[pct] = {
                    'mIoU': np.mean([s[pct]['mIoU'] for s in domain_info['pct_metrics_seeds']]),
                    'mDice': np.mean([s[pct]['mDice'] for s in domain_info['pct_metrics_seeds']]),
                    'mAcc': np.mean([s[pct]['mAcc'] for s in domain_info['pct_metrics_seeds']]),
                }

        all_results[corruption] = c_results_print
        domain_summary.append({
            'corruption': corruption,
            'mIoU_mean': miou_mean,
            'mIoU_std': miou_std,
            'mDice_mean': dice_mean,
            'mDice_std': dice_std,
            'mAcc_mean': acc_mean,
            'mAcc_std': acc_std,
            'loss_mean': loss_mean,
            'loss_increase': loss_inc,
            'loss_decrease': loss_dec,
            'safs_filtered': safs_filtered,
            'safs_unfiltered': safs_unfiltered,
            'per_class_iou': avg_per_class_iou,
            'pct_metrics': avg_pct_metrics,
        })

        if args.plot_loss and args.adapt and domain_info['loss_seed_report']:
            loss_seed_report = np.array(domain_info['loss_seed_report'])
            avg_loss_over_seeds = np.mean(loss_seed_report, axis=0)
            plt.figure()
            plt.plot(range(1, len(avg_loss_over_seeds) + 1), avg_loss_over_seeds)
            plt.xlabel('Iteration')
            plt.ylabel('Average Loss')
            plt.title(f'Average Loss per Iteration for {corruption}')
            save_path = os.path.join(args.save_dir, f'loss_{corruption}.png')
            plt.savefig(save_path)
            plt.close()

    total_duration = time.time() - start_time
    mean_duration_per_seed = total_duration / args.trials
    gpu_info = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    print("\n===== Per-domain Summary =====")
    for domain_metrics in domain_summary:
        loss_str = ""
        if domain_metrics.get('loss_mean') is not None:
            loss_str = (
                f", Loss {domain_metrics['loss_mean']:.4f}"
                f", Loss+ {domain_metrics['loss_increase']:.4f}"
                f", Loss- {domain_metrics['loss_decrease']:.4f}"
            )
        safs_str = ""
        if domain_metrics.get('safs_filtered') is not None:
            safs_str = (
                f", SAFS filtered {domain_metrics['safs_filtered']}"
                f", unfiltered {domain_metrics['safs_unfiltered']}"
            )
        print(
            f"{domain_metrics['corruption']}: "
            f"mIoU {domain_metrics['mIoU_mean']:.2f} +/- {domain_metrics['mIoU_std']:.2f}, "
            f"mDice {domain_metrics['mDice_mean']:.2f} +/- {domain_metrics['mDice_std']:.2f}, "
            f"mAcc {domain_metrics['mAcc_mean']:.2f} +/- {domain_metrics['mAcc_std']:.2f}"
            f"{loss_str}{safs_str}"
        )

    if eval_corruptions:
        summary_domains = [domain_metrics for domain_metrics in domain_summary if domain_metrics['corruption'] in eval_corruptions]
    else:
        summary_domains = domain_summary

    overall_miou_mean = np.mean([domain_metrics['mIoU_mean'] for domain_metrics in summary_domains])
    overall_mdice_mean = np.mean([domain_metrics['mDice_mean'] for domain_metrics in summary_domains])
    overall_macc_mean = np.mean([domain_metrics['mAcc_mean'] for domain_metrics in summary_domains])

    print("===== Overall Mean Summary =====")
    print(
        f"Overall mean across evaluation domains: "
        f"mIoU {overall_miou_mean:.2f}, "
        f"mDice {overall_mdice_mean:.2f}, "
        f"mAcc {overall_macc_mean:.2f}"
    )

    # Overall mean per class across domains
    per_class_iou_domains = [
        dm['per_class_iou'] for dm in summary_domains
        if dm.get('per_class_iou') is not None
    ]
    if per_class_iou_domains:
        overall_per_class_iou = np.mean(per_class_iou_domains, axis=0)
        print("Overall mean per class across domains:")
        for cls_name, cls_iou in zip(domain_infos[0]['org_classes'], overall_per_class_iou):
            print(f"  {cls_name}: {cls_iou:.2f}")

    # Overall mean per dataset % across domains
    pct_domains = [
        dm['pct_metrics'] for dm in summary_domains
        if dm.get('pct_metrics')
    ]
    if pct_domains:
        print("Overall mean per dataset % across domains:")
        for pct in sorted(pct_domains[0].keys()):
            pct_miou = np.mean([dm[pct]['mIoU'] for dm in pct_domains])
            pct_mdice = np.mean([dm[pct]['mDice'] for dm in pct_domains])
            pct_macc = np.mean([dm[pct]['mAcc'] for dm in pct_domains])
            print(f"  {int(pct * 100)}%: mIoU {pct_miou:.2f}, mDice {pct_mdice:.2f}, mAcc {pct_macc:.2f}")

    with open(all_results_path, 'w') as f:
        f.write(headers + "\n")
        for corruption, results in all_results.items():
            f.write(f"{corruption}, {results}\n")
        f.write(f"\nGPU: {gpu_info}\n")
        f.write(f"Total Duration (s): {total_duration:.2f}\n")
        f.write(f"Mean Duration per Seed (s): {mean_duration_per_seed:.2f}\n")
        if per_class_iou_domains:
            f.write("\nOverall mean per class across domains:\n")
            for cls_name, cls_iou in zip(domain_infos[0]['org_classes'], overall_per_class_iou):
                f.write(f"  {cls_name}: {cls_iou:.2f}\n")
        if pct_domains:
            f.write("\nOverall mean per dataset % across domains:\n")
            for pct in sorted(pct_domains[0].keys()):
                pct_miou = np.mean([dm[pct]['mIoU'] for dm in pct_domains])
                pct_mdice = np.mean([dm[pct]['mDice'] for dm in pct_domains])
                pct_macc = np.mean([dm[pct]['mAcc'] for dm in pct_domains])
                f.write(f"  {int(pct * 100)}%: mIoU {pct_miou:.2f}, mDice {pct_mdice:.2f}, mAcc {pct_macc:.2f}\n")


def process_single_batch_no_adapt(args, device, adapt_method, data, domain_info, demo_info=None):
    inputs = data['img_patches']
    labels = data['gt_patches']
    original_gts = data['gt']

    patch_grid_shape = data['meta']['patch_grid_shape']
    image_shapes = data['meta']['img_shape']
    inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

    with torch.no_grad():
        patch_preds = adapt_method.evaluate(inputs)

    eval_size = getattr(adapt_method, 'eval_size', args.patch_size[0])
    eval_scale = eval_size / args.patch_size[0]

    if args.init_resize:
        if eval_scale < 1.0:
            scaled_patch_size = (round(args.patch_size[0] * eval_scale), round(args.patch_size[1] * eval_scale))
            scaled_patch_stride = round(args.patch_stride * eval_scale)
            scaled_img_shapes = [(round(h * eval_scale), round(w * eval_scale)) for h, w in image_shapes]
            reconstructed_preds = aggregate_pred_patches(
                patch_preds, patch_grid_shape, scaled_img_shapes, scaled_patch_size, scaled_patch_stride)
        else:
            reconstructed_preds = aggregate_pred_patches(
                patch_preds, patch_grid_shape, image_shapes, args.patch_size, args.patch_stride)
    else:
        reconstructed_preds = patch_preds

    batch_results = []
    for idx, (pd, gt) in enumerate(zip(reconstructed_preds, original_gts)):
        pd = pd.softmax(dim=0)

        if domain_info['ext_to_real_cls_indx'] is not None:
            pd = pd.unsqueeze(0)
            pd = (pd * domain_info['ext_to_real_cls_indx']).max(1)[0]

        pd = pd.argmax(dim=0)
        pd = pd.to(gt.device)
        gt = gt[0]
        if eval_scale < 1.0:
            target_h, target_w = scaled_img_shapes[idx]
            gt = torch.nn.functional.interpolate(
                gt.unsqueeze(0).unsqueeze(0).float(), size=(target_h, target_w), mode='nearest'
            ).squeeze(0).squeeze(0).long()
        batch_results.append(
            intersect_and_union(pd, gt, domain_info['num_org_classes'], domain_info['ignore_index'])
        )

        if demo_info is not None and demo_info.get('global_sample_idx') in demo_info['indices']:
            demo_dir = os.path.join('diagnostics', 'x_demo', args.dataset, domain_info['corruption'], get_demo_method_name(args))
            img_tensor = data['img'][idx]
            save_demo_overlay(img_tensor, pd, demo_info['palette'],
                              os.path.join(demo_dir, f"pred_{demo_info['global_sample_idx']:04d}.png"))
            save_demo_overlay(img_tensor, gt, demo_info['palette'],
                              os.path.join(demo_dir, f"gt_{demo_info['global_sample_idx']:04d}.png"))
            save_demo_input(img_tensor,
                            os.path.join(demo_dir, f"input_{demo_info['global_sample_idx']:04d}.png"))
        if demo_info is not None:
            demo_info['global_sample_idx'] += 1

    return batch_results, None, [], [], []


def prepare_domain_info(args, device, corruption, c_idx):
    data_loader, org_classes = segmentation_datasets.prepare_data(
        args.dataset,
        args.data_dir,
        args.init_resize,
        args.patch_size,
        args.patch_stride,
        corruption=corruption,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=not getattr(args, 'save_demo', False),
    )

    if args.class_extensions and data_loader.dataset.class_extensions is not None:
        classes = data_loader.dataset.class_extensions
        print(f"\n+++ Using class extensions")
        print(f"+++ The number of classes [no extension]: {len(org_classes)}")
        print(f"+++ The number of classes after extension:  {len(classes)}")
        ext_to_real_cls_indx = torch.Tensor(data_loader.dataset.extentions_to_real_class_idx).to(torch.int64).to(device)
        num_cls, num_queries = max(ext_to_real_cls_indx) + 1, len(ext_to_real_cls_indx)
        ext_to_real_cls_indx = torch.nn.functional.one_hot(ext_to_real_cls_indx)
        ext_to_real_cls_indx = ext_to_real_cls_indx.T.view(num_cls, num_queries, 1, 1)
    else:
        classes = org_classes
        ext_to_real_cls_indx = None
        print(f"\n+++ The number of classes [no extension]: {len(org_classes)}")

    c_results_path = os.path.join(args.save_dir, f"{c_idx:02}_{corruption}", "results.txt")
    os.makedirs(os.path.dirname(c_results_path), exist_ok=True)

    return {
        'corruption': corruption,
        'data_loader': data_loader,
        'org_classes': org_classes,
        'classes': classes,
        'num_org_classes': len(org_classes),
        'ignore_index': data_loader.dataset.ignore_index,
        'ext_to_real_cls_indx': ext_to_real_cls_indx,
        'c_results_path': c_results_path,
        'miou_seeds': [],
        'dice_seeds': [],
        'acc_seeds': [],
        'loss_seed_report': [],
        'safs_stats_list': [],
        'per_class_iou_seeds': [],
        'pct_metrics_seeds': [],
    }


def process_single_batch(args, device, adapt_method, data, domain_info, demo_info=None):
    inputs = data['img_patches']
    labels = data['gt_patches']
    original_gts = data['gt']

    patch_grid_shape = data['meta']['patch_grid_shape']
    image_shapes = data['meta']['img_shape']
    inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

    adapt_len_before = len(adapt_method.adapt_times) if args.runtime_calculation and args.adapt else None
    eval_len_before = len(adapt_method.eval_times) if args.runtime_calculation else None

    loss_iter_report = None
    if args.adapt:
        if getattr(args, 'diag_safs', False) or getattr(args, 'diag_div', False):
            diag_labels = labels
            diag_ignore = domain_info['ignore_index']
            loss_iter_report = adapt_method.adapt(inputs, diag_labels=diag_labels, diag_ignore_index=diag_ignore)
        else:
            loss_iter_report = adapt_method.adapt(inputs)

    with torch.no_grad():
        patch_preds = adapt_method.evaluate(inputs)

    eval_size = getattr(adapt_method, 'eval_size', args.patch_size[0])
    eval_scale = eval_size / args.patch_size[0]

    if args.init_resize:
        if eval_scale < 1.0:
            scaled_patch_size = (round(args.patch_size[0] * eval_scale), round(args.patch_size[1] * eval_scale))
            scaled_patch_stride = round(args.patch_stride * eval_scale)
            scaled_img_shapes = [(round(h * eval_scale), round(w * eval_scale)) for h, w in image_shapes]
            reconstructed_preds = aggregate_pred_patches(
                patch_preds, patch_grid_shape, scaled_img_shapes, scaled_patch_size, scaled_patch_stride)
        else:
            reconstructed_preds = aggregate_pred_patches(
                patch_preds, patch_grid_shape, image_shapes, args.patch_size, args.patch_stride)
    else:
        reconstructed_preds = patch_preds

    batch_results = []
    for idx, (pd, gt) in enumerate(zip(reconstructed_preds, original_gts)):
        pd = pd.softmax(dim=0)

        if domain_info['ext_to_real_cls_indx'] is not None:
            pd = pd.unsqueeze(0)
            pd = (pd * domain_info['ext_to_real_cls_indx']).max(1)[0]

        pd = pd.argmax(dim=0)
        pd = pd.to(gt.device)
        gt = gt[0]
        if eval_scale < 1.0:
            target_h, target_w = scaled_img_shapes[idx]
            gt = torch.nn.functional.interpolate(
                gt.unsqueeze(0).unsqueeze(0).float(), size=(target_h, target_w), mode='nearest'
            ).squeeze(0).squeeze(0).long()
        batch_results.append(
            intersect_and_union(pd, gt, domain_info['num_org_classes'], domain_info['ignore_index'])
        )

        if demo_info is not None and demo_info.get('global_sample_idx') in demo_info['indices']:
            demo_dir = os.path.join('diagnostics', 'x_demo', args.dataset, domain_info['corruption'], get_demo_method_name(args))
            img_tensor = data['img'][idx]
            save_demo_overlay(img_tensor, pd, demo_info['palette'],
                              os.path.join(demo_dir, f"pred_{demo_info['global_sample_idx']:04d}.png"))
        if demo_info is not None:
            demo_info['global_sample_idx'] += 1

    adapt_times = []
    eval_times = []
    if args.runtime_calculation and args.adapt:
        adapt_times = adapt_method.adapt_times[adapt_len_before:]
    if args.runtime_calculation:
        eval_times = adapt_method.eval_times[eval_len_before:]

    weights = []
    if adapt_method.model.weights_track:
        weights = list(adapt_method.model.weights_track)
        adapt_method.model.weights_track = []

    return batch_results, loss_iter_report, adapt_times, eval_times, weights


def summarize_results(results):
    results = tuple(zip(*results))
    total_area_intersect = sum(results[0])
    total_area_union = sum(results[1])
    total_area_pred_label = sum(results[2])
    total_area_label = sum(results[3])
    ret_metrics = total_area_to_metrics(
        total_area_intersect,
        total_area_union,
        total_area_pred_label,
        total_area_label,
    )

    return {
        'mIoU': np.round(np.nanmean(ret_metrics['IoU']) * 100, 2),
        'mDice': np.round(np.nanmean(ret_metrics['Dice']) * 100, 2),
        'mAcc': np.round(np.nanmean(ret_metrics['Acc']) * 100, 2),
    }


def compute_per_class_iou(results):
    """Compute per-class IoU from accumulated intersect/union results.

    Args:
        results: List of (area_intersect, area_union, area_pred_label,
                  area_label) tuples from intersect_and_union.

    Returns:
        np.ndarray: Per-class IoU values in 0-100 scale, shape (num_classes,).
    """
    results = tuple(zip(*results))
    total_area_intersect = sum(results[0])
    total_area_union = sum(results[1])
    total_area_pred_label = sum(results[2])
    total_area_label = sum(results[3])
    ret_metrics = total_area_to_metrics(
        total_area_intersect,
        total_area_union,
        total_area_pred_label,
        total_area_label,
    )
    return np.round(np.nan_to_num(ret_metrics['IoU']) * 100, 2)


def compute_metrics_at_pct(results, percentages=(0.1, 0.2, 0.4, 0.8)):
    """Compute summary metrics at various dataset percentage cutoffs.

    Args:
        results: List of (area_intersect, area_union, area_pred_label,
                  area_label) tuples from intersect_and_union.
        percentages: Tuple of fractions (0-1) of the dataset to evaluate.

    Returns:
        dict: {pct_fraction: {'mIoU': float, 'mDice': float, 'mAcc': float}}
    """
    pct_metrics = {}
    n = len(results)
    for pct in percentages:
        cutoff = max(1, int(n * pct))
        slice_results = results[:cutoff]
        pct_metrics[pct] = summarize_results(slice_results)
    return pct_metrics


def aggregate_safs_stats(stats_list):
    """Aggregate SAFS filtering stats across batches.

    Returns (total_filtered, total_unfiltered) or (None, None) if empty.
    """
    if not stats_list:
        return None, None
    total_filtered = sum(s['filtered'] for s in stats_list)
    total_unfiltered = sum(s['kept'] for s in stats_list)
    return total_filtered, total_unfiltered


def write_diag_safs_logs(args, corruption, miou, diag_stats):
    """Write SAFS diagnostic stats to JSON log file.

    Saves per-sample diagnostic metrics (feature_shift, entropy, pred_change,
    pred_agreement) along with the corruption's mIoU to
    diagnostics/safs/logs/{dataset}/{corruption}.json
    """
    if not diag_stats:
        return
    log_dir = os.path.join('diagnostics', 'safs', 'logs', args.dataset)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{corruption}.json")
    log_data = {
        'dataset': args.dataset,
        'corruption': corruption,
        'mIoU': miou,
        'num_samples': len(diag_stats),
        'samples': diag_stats,
    }
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)
    print(f"SAFS diagnostics written to {log_path}")


def write_diag_cmac_logs(args, corruption, diag_stats):
    """Write CMAC diagnostic stats to JSON log file.

    Saves per-step prototype drift metrics (assigned_sim, unassigned_sim,
    class_fractions) to diagnostics/cmac/logs/{dataset}/{corruption}.json
    """
    if not diag_stats:
        return
    log_dir = os.path.join('diagnostics', 'cmac', 'logs', args.dataset)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{corruption}.json")
    log_data = {
        'dataset': args.dataset,
        'corruption': corruption,
        'num_steps': len(diag_stats),
        'steps': diag_stats,
    }
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)
    print(f"CMAC diagnostics written to {log_path}")


def write_diag_div_logs(args, corruption, diag_stats):
    """Write DIV diagnostic stats to JSON log file.

    Saves per-step class prediction collapse metrics (class_fractions, hhi,
    num_active_classes, max_class_share, absent_class_mass, absent_class_fpr)
    to diagnostics/div/logs/{dataset}_div_{on|off}/{corruption}.json
    """
    if not diag_stats:
        return
    div_suffix = 'div_on' if getattr(args, 'loss_div', False) else 'div_off'
    log_dir = os.path.join('diagnostics', 'div', 'logs', f"{args.dataset}_{div_suffix}")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{corruption}.json")
    log_data = {
        'dataset': args.dataset,
        'corruption': corruption,
        'loss_div': getattr(args, 'loss_div', False),
        'num_steps': len(diag_stats),
        'steps': diag_stats,
    }
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)
    print(f"DIV diagnostics written to {log_path}")

    absent_mass_vals = [s['absent_class_mass'] for s in diag_stats if s.get('absent_class_mass') is not None]
    absent_fpr_vals = [s['absent_class_fpr'] for s in diag_stats if s.get('absent_class_fpr') is not None]
    num_present_vals = [s['num_present_classes'] for s in diag_stats if s.get('num_present_classes') is not None]
    num_absent_vals = [s['num_absent_classes'] for s in diag_stats if s.get('num_absent_classes') is not None]
    print(f"  [DIV diag] loss_div={getattr(args, 'loss_div', False)}, corruption={corruption}")
    if absent_mass_vals:
        print(f"    absent_class_mass: mean={np.mean(absent_mass_vals):.6f}, std={np.std(absent_mass_vals):.6f}")
    if absent_fpr_vals:
        print(f"    absent_class_fpr:  mean={np.mean(absent_fpr_vals):.6f}, std={np.std(absent_fpr_vals):.6f}")
    if num_present_vals:
        print(f"    num_present_classes: mean={np.mean(num_present_vals):.1f}")
    if num_absent_vals:
        print(f"    num_absent_classes:  mean={np.mean(num_absent_vals):.1f}")


def compute_loss_stats(loss_seed_report):
    if loss_seed_report is None or len(loss_seed_report) == 0:
        return None, None, None

    loss_arr = np.array(loss_seed_report, dtype=float)
    if loss_arr.size == 0 or np.all(np.isnan(loss_arr)):
        return None, None, None

    avg_loss_per_iter = np.nanmean(loss_arr, axis=0)
    avg_loss = float(np.nanmean(avg_loss_per_iter))

    if len(avg_loss_per_iter) > 1:
        deltas = np.diff(avg_loss_per_iter)
        positive_deltas = deltas[deltas > 0]
        negative_deltas = deltas[deltas < 0]
        avg_increase = float(np.mean(positive_deltas)) if len(positive_deltas) > 0 else 0.0
        avg_decrease = float(np.mean(negative_deltas)) if len(negative_deltas) > 0 else 0.0
    else:
        avg_increase = 0.0
        avg_decrease = 0.0

    return avg_loss, avg_increase, avg_decrease


def run_lifelong(args, device, start_time, all_results_path):
    headers = "mIoU, mDice, mAcc"
    all_results = dict()
    domain_summary = []
    adapt_time_all_corr = []
    eval_time_all_corr = []
    round_summary = [
        {
            'mIoU': [],
            'mDice': [],
            'mAcc': [],
        }
        for _ in range(args.lifelong_rnds)
    ]

    domain_infos = []
    for c_idx, corruption in enumerate(args.corruptions_list):
        domain_infos.append(prepare_domain_info(args, device, corruption, c_idx))

    args.classes = domain_infos[0]['classes']
    domain_map = {domain_info['corruption']: domain_info for domain_info in domain_infos}

    demo_info = None
    if getattr(args, 'save_demo', False):
        demo_indices = set(get_demo_indices(len(domain_infos[0]['data_loader'].dataset), args.save_k, args.seed))
        demo_info = {
            'indices': demo_indices,
            'palette': domain_infos[0]['data_loader'].dataset.metainfo['palette'],
            'global_sample_idx': 0,
        }
        print(f"+++ Demo: saving {len(demo_indices)} images per corruption")

    continual_methods = None
    if args.reset_mode == 'continual':
        continual_methods = [get_method(args, device) for _ in range(args.trials)]

    for t in range(args.trials):
        if args.reset_mode == 'continual':
            adapt_method = continual_methods[t]
        else:
            adapt_method = get_method(args, device)

        trial_results = {domain_info['corruption']: [] for domain_info in domain_infos}
        trial_loss_batch_report = {domain_info['corruption']: [] for domain_info in domain_infos}
        trial_adapt_times = {domain_info['corruption']: [] for domain_info in domain_infos}
        trial_eval_times = {domain_info['corruption']: [] for domain_info in domain_infos}
        trial_weights = {domain_info['corruption']: [] for domain_info in domain_infos}

        for round_idx in range(args.lifelong_rnds):
            round_results = {domain_info['corruption']: [] for domain_info in domain_infos}
            print(f"\n===== Lifelong Round {round_idx + 1}/{args.lifelong_rnds} | Trial {t} =====")

            if args.lifelong in ('shuffle_domain_pround', 'recurring_domain_pround'):
                if args.lifelong == 'shuffle_domain_pround':
                    round_rng = np.random.default_rng(args.seed + round_idx)
                    corruption_order = list(round_rng.permutation(args.corruptions_list))
                else:
                    corruption_order = list(args.corruptions_list)
                print(f"Round {round_idx + 1} domain order: {' -> '.join(corruption_order)}")

                for domain_order_idx, corruption in enumerate(corruption_order):

                    domain_info = domain_map[corruption]

                    if args.reset_mode == 'normal':
                        adapt_method.reset()

                    safs_len_before = len(adapt_method.safs_stats) if hasattr(adapt_method, 'safs_stats') else 0
                    if demo_info is not None:
                        demo_info['global_sample_idx'] = 0
                    for batch_idx, data in tqdm(enumerate(domain_info['data_loader']), total=len(domain_info['data_loader'])):
                        if args.debug and batch_idx == 10:
                            break

                        if args.reset_mode == 'episodic':
                            adapt_method.reset()

                        batch_results, loss_iter_report, adapt_times, eval_times, weights = process_single_batch(
                            args,
                            device,
                            adapt_method,
                            data,
                            domain_info,
                            demo_info=demo_info,
                        )

                        trial_results[corruption].extend(batch_results)
                        round_results[corruption].extend(batch_results)
                        if loss_iter_report is not None:
                            trial_loss_batch_report[corruption].append(loss_iter_report)
                        trial_adapt_times[corruption].extend(adapt_times)
                        trial_eval_times[corruption].extend(eval_times)
                        trial_weights[corruption].extend(weights)

                    if hasattr(adapt_method, 'safs_stats'):
                        safs_len_after = len(adapt_method.safs_stats)
                        domain_info['safs_stats_list'].extend(adapt_method.safs_stats[safs_len_before:safs_len_after])

            elif args.lifelong == 'shuffle_domain_pbatch':
                iterators = {domain_info['corruption']: iter(domain_info['data_loader']) for domain_info in domain_infos}
                active_corruptions = [domain_info['corruption'] for domain_info in domain_infos]
                demo_offsets = {domain_info['corruption']: 0 for domain_info in domain_infos}
                cycle_idx = 0
                debug_counts = {domain_info['corruption']: 0 for domain_info in domain_infos}

                while active_corruptions:
                    cycle_rng = np.random.default_rng(args.seed + round_idx * 100000 + cycle_idx)
                    cycle_order = list(cycle_rng.permutation(active_corruptions))
                    print(f"Round {round_idx + 1} cycle {cycle_idx + 1} order: {' -> '.join(cycle_order)}")
                    next_active = []

                    for corruption in cycle_order:
                        if args.debug and debug_counts[corruption] == 10:
                            continue

                        try:
                            data = next(iterators[corruption])
                        except StopIteration:
                            continue

                        debug_counts[corruption] += 1
                        domain_info = domain_map[corruption]

                        if args.reset_mode in ('episodic', 'normal'):
                            adapt_method.reset()

                        safs_len_before = len(adapt_method.safs_stats) if hasattr(adapt_method, 'safs_stats') else 0
                        if demo_info is not None:
                            demo_info['global_sample_idx'] = demo_offsets[corruption]
                        batch_results, loss_iter_report, adapt_times, eval_times, weights = process_single_batch(
                            args,
                            device,
                            adapt_method,
                            data,
                            domain_info,
                            demo_info=demo_info,
                        )
                        if demo_info is not None:
                            demo_offsets[corruption] = demo_info['global_sample_idx']
                        if hasattr(adapt_method, 'safs_stats'):
                            safs_len_after = len(adapt_method.safs_stats)
                            domain_info['safs_stats_list'].extend(adapt_method.safs_stats[safs_len_before:safs_len_after])

                        trial_results[corruption].extend(batch_results)
                        round_results[corruption].extend(batch_results)
                        if loss_iter_report is not None:
                            trial_loss_batch_report[corruption].append(loss_iter_report)
                        trial_adapt_times[corruption].extend(adapt_times)
                        trial_eval_times[corruption].extend(eval_times)
                        trial_weights[corruption].extend(weights)
                        next_active.append(corruption)

                    active_corruptions = next_active
                    cycle_idx += 1

            round_domain_metrics = []
            for domain_info in domain_infos:
                corruption = domain_info['corruption']
                metrics = summarize_results(round_results[corruption])
                round_domain_metrics.append(metrics)

            round_miou = np.mean([metrics['mIoU'] for metrics in round_domain_metrics])
            round_mdice = np.mean([metrics['mDice'] for metrics in round_domain_metrics])
            round_macc = np.mean([metrics['mAcc'] for metrics in round_domain_metrics])
            round_summary[round_idx]['mIoU'].append(round_miou)
            round_summary[round_idx]['mDice'].append(round_mdice)
            round_summary[round_idx]['mAcc'].append(round_macc)

            print(
                f"Round {round_idx + 1} final metrics: "
                f"mIoU {round_miou:.2f}, "
                f"mDice {round_mdice:.2f}, "
                f"mAcc {round_macc:.2f}"
            )

        for domain_info in domain_infos:
            corruption = domain_info['corruption']
            metrics = process_metrics(trial_results[corruption], domain_info['org_classes'])
            domain_info['miou_seeds'].append(metrics['mIoU'])
            domain_info['dice_seeds'].append(metrics['mDice'])
            domain_info['acc_seeds'].append(metrics['mAcc'])
            domain_info['per_class_iou_seeds'].append(compute_per_class_iou(trial_results[corruption]))
            domain_info['pct_metrics_seeds'].append(compute_metrics_at_pct(trial_results[corruption]))
            print(f"Results for corruption: {corruption}, trial: {t}, mIoU:  {metrics['mIoU']}, mDice:  {metrics['mDice']}, mAcc: {metrics['mAcc']}")

            if trial_loss_batch_report[corruption]:
                loss_batch_report = np.array(trial_loss_batch_report[corruption])
                avg_loss_per_iter = np.mean(loss_batch_report, axis=0)
                domain_info['loss_seed_report'].append(avg_loss_per_iter)

            if trial_weights[corruption]:
                weights_path = os.path.join(args.save_dir, "weights")
                weights = np.hstack(trial_weights[corruption])
                os.makedirs(weights_path, exist_ok=True)
                np.save(os.path.join(weights_path, f"{corruption}_s{t}.npy"), np.array(weights))

                weights_mean = np.mean(weights, axis=1)
                weights_std = np.std(weights, axis=1)
                plt.figure()
                plt.errorbar(range(len(weights_mean)), weights_mean, yerr=weights_std, fmt='o')
                plt.xlabel('Layer')
                plt.ylabel('Weight')
                plt.title(f'Mean and Std of Weights for {corruption}')
                plt.savefig(os.path.join(weights_path, f"{corruption}_s{t}.png"))
                plt.close()

            if args.runtime_calculation:
                if args.adapt:
                    adapt_times = trial_adapt_times[corruption][20:] if len(trial_adapt_times[corruption]) > 20 else trial_adapt_times[corruption]
                    mean_adapt_time = np.mean(adapt_times) if adapt_times else 0
                    std_adapt_time = np.std(adapt_times) if adapt_times else 0
                else:
                    mean_adapt_time = 0
                    std_adapt_time = 0

                eval_times = trial_eval_times[corruption][20:] if len(trial_eval_times[corruption]) > 20 else trial_eval_times[corruption]
                mean_eval_time = np.mean(eval_times) if eval_times else 0
                std_eval_time = np.std(eval_times) if eval_times else 0
                mean_total_time = mean_adapt_time + mean_eval_time

                run_time_txt = f"{corruption}, {mean_adapt_time:0.3f} +/- {std_adapt_time:0.3f}, {mean_eval_time:0.3f} +/- {std_eval_time:0.3f}, {mean_total_time:0.3f}"
                print(run_time_txt)

                runtime_save_dir = os.path.join(args.save_dir, "runtime.txt")
                with open(runtime_save_dir, 'a+') as f:
                    f.write(run_time_txt + "\n")

                adapt_time_all_corr.append(mean_adapt_time)
                eval_time_all_corr.append(mean_eval_time)

    for domain_info in domain_infos:
        corruption = domain_info['corruption']
        miou_mean = np.array(domain_info['miou_seeds']).mean()
        miou_std = np.array(domain_info['miou_seeds']).std()
        dice_mean = np.array(domain_info['dice_seeds']).mean()
        dice_std = np.array(domain_info['dice_seeds']).std()
        acc_mean = np.array(domain_info['acc_seeds']).mean()
        acc_std = np.array(domain_info['acc_seeds']).std()

        safs_filtered, safs_unfiltered = aggregate_safs_stats(domain_info['safs_stats_list'])
        print(f"mIoU:  {miou_mean:.2f},{miou_std:.2f}")
        print(f"mDice: {dice_mean:.2f},{dice_std:.2f}")
        print(f"mAcc:  {acc_mean:.2f},{acc_std:.2f}")
        if safs_filtered is not None:
            print(f"SAFS: filtered={safs_filtered}, unfiltered={safs_unfiltered}")

        if getattr(args, 'diag_safs', False) and hasattr(adapt_method, 'diag_safs_stats') and adapt_method.diag_safs_stats:
            write_diag_safs_logs(args, corruption, miou_mean, adapt_method.diag_safs_stats)

        if getattr(args, 'diag_cmac', False) and hasattr(adapt_method, 'diag_cmac_stats') and adapt_method.diag_cmac_stats:
            write_diag_cmac_logs(args, corruption, adapt_method.diag_cmac_stats)
            adapt_method.diag_cmac_step_offset += len(adapt_method.diag_cmac_stats)
            adapt_method.diag_cmac_stats = []

        if getattr(args, 'diag_div', False) and hasattr(adapt_method, 'diag_div_stats') and adapt_method.diag_div_stats:
            write_diag_div_logs(args, corruption, adapt_method.diag_div_stats)
            adapt_method.diag_div_step_offset += len(adapt_method.diag_div_stats)
            adapt_method.diag_div_stats = []

        c_results_print = f"{miou_mean:.2f} +/- {miou_std:.2f}, {dice_mean:.2f} +/- {dice_std:.2f}, {acc_mean:.2f} +/- {acc_std:.2f}"
        with open(domain_info['c_results_path'], 'w') as f:
            f.write(headers + "\n")
            f.write(c_results_print)

        loss_mean, loss_inc, loss_dec = compute_loss_stats(domain_info['loss_seed_report'])

        avg_per_class_iou = np.mean(domain_info['per_class_iou_seeds'], axis=0) if domain_info['per_class_iou_seeds'] else None
        avg_pct_metrics = {}
        if domain_info['pct_metrics_seeds']:
            for pct in domain_info['pct_metrics_seeds'][0]:
                avg_pct_metrics[pct] = {
                    'mIoU': np.mean([s[pct]['mIoU'] for s in domain_info['pct_metrics_seeds']]),
                    'mDice': np.mean([s[pct]['mDice'] for s in domain_info['pct_metrics_seeds']]),
                    'mAcc': np.mean([s[pct]['mAcc'] for s in domain_info['pct_metrics_seeds']]),
                }

        all_results[corruption] = c_results_print
        domain_summary.append({
            'corruption': corruption,
            'mIoU_mean': miou_mean,
            'mIoU_std': miou_std,
            'mDice_mean': dice_mean,
            'mDice_std': dice_std,
            'mAcc_mean': acc_mean,
            'mAcc_std': acc_std,
            'loss_mean': loss_mean,
            'loss_increase': loss_inc,
            'loss_decrease': loss_dec,
            'safs_filtered': safs_filtered,
            'safs_unfiltered': safs_unfiltered,
            'per_class_iou': avg_per_class_iou,
            'pct_metrics': avg_pct_metrics,
        })

        if args.plot_loss and args.adapt and domain_info['loss_seed_report']:
            loss_seed_report = np.array(domain_info['loss_seed_report'])
            avg_loss_over_seeds = np.mean(loss_seed_report, axis=0)
            plt.figure()
            plt.plot(range(1, len(avg_loss_over_seeds) + 1), avg_loss_over_seeds)
            plt.xlabel('Iteration')
            plt.ylabel('Average Loss')
            plt.title(f'Average Loss per Iteration for {corruption}')
            save_path = os.path.join(args.save_dir, f'loss_{corruption}.png')
            plt.savefig(save_path)
            plt.close()

    total_duration = time.time() - start_time
    mean_duration_per_seed = total_duration / args.trials
    gpu_info = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    print("\n===== Per-domain Summary =====")
    for domain_metrics in domain_summary:
        loss_str = ""
        if domain_metrics.get('loss_mean') is not None:
            loss_str = (
                f", Loss {domain_metrics['loss_mean']:.4f}"
                f", Loss+ {domain_metrics['loss_increase']:.4f}"
                f", Loss- {domain_metrics['loss_decrease']:.4f}"
            )
        safs_str = ""
        if domain_metrics.get('safs_filtered') is not None:
            safs_str = (
                f", SAFS filtered {domain_metrics['safs_filtered']}"
                f", unfiltered {domain_metrics['safs_unfiltered']}"
            )
        print(
            f"{domain_metrics['corruption']}: "
            f"mIoU {domain_metrics['mIoU_mean']:.2f} +/- {domain_metrics['mIoU_std']:.2f}, "
            f"mDice {domain_metrics['mDice_mean']:.2f} +/- {domain_metrics['mDice_std']:.2f}, "
            f"mAcc {domain_metrics['mAcc_mean']:.2f} +/- {domain_metrics['mAcc_std']:.2f}"
            f"{loss_str}{safs_str}"
        )

    print("===== Per-round Summary =====")
    for round_idx, metrics in enumerate(round_summary):
        round_miou_mean = np.mean(metrics['mIoU'])
        round_miou_std = np.std(metrics['mIoU'])
        round_mdice_mean = np.mean(metrics['mDice'])
        round_mdice_std = np.std(metrics['mDice'])
        round_macc_mean = np.mean(metrics['mAcc'])
        round_macc_std = np.std(metrics['mAcc'])
        print(
            f"Round {round_idx + 1}: "
            f"mIoU {round_miou_mean:.2f} +/- {round_miou_std:.2f}, "
            f"mDice {round_mdice_mean:.2f} +/- {round_mdice_std:.2f}, "
            f"mAcc {round_macc_mean:.2f} +/- {round_macc_std:.2f}"
        )

    overall_miou_mean = np.mean([domain_metrics['mIoU_mean'] for domain_metrics in domain_summary])
    overall_mdice_mean = np.mean([domain_metrics['mDice_mean'] for domain_metrics in domain_summary])
    overall_macc_mean = np.mean([domain_metrics['mAcc_mean'] for domain_metrics in domain_summary])

    print("===== Overall Mean Summary =====")
    print(
        f"Overall mean across domains: "
        f"mIoU {overall_miou_mean:.2f}, "
        f"mDice {overall_mdice_mean:.2f}, "
        f"mAcc {overall_macc_mean:.2f}"
    )

    # Overall mean per class across domains
    per_class_iou_domains = [
        dm['per_class_iou'] for dm in domain_summary
        if dm.get('per_class_iou') is not None
    ]
    if per_class_iou_domains:
        overall_per_class_iou = np.mean(per_class_iou_domains, axis=0)
        print("Overall mean per class across domains:")
        for cls_name, cls_iou in zip(domain_infos[0]['org_classes'], overall_per_class_iou):
            print(f"  {cls_name}: {cls_iou:.2f}")

    # Overall mean per dataset % across domains
    pct_domains = [
        dm['pct_metrics'] for dm in domain_summary
        if dm.get('pct_metrics')
    ]
    if pct_domains:
        print("Overall mean per dataset % across domains:")
        for pct in sorted(pct_domains[0].keys()):
            pct_miou = np.mean([dm[pct]['mIoU'] for dm in pct_domains])
            pct_mdice = np.mean([dm[pct]['mDice'] for dm in pct_domains])
            pct_macc = np.mean([dm[pct]['mAcc'] for dm in pct_domains])
            print(f"  {int(pct * 100)}%: mIoU {pct_miou:.2f}, mDice {pct_mdice:.2f}, mAcc {pct_macc:.2f}")

    with open(all_results_path, 'w') as f:
        f.write(headers + "\n")
        for corruption, results in all_results.items():
            f.write(f"{corruption}, {results}\n")
        f.write(f"\nGPU: {gpu_info}\n")
        f.write(f"Total Duration (s): {total_duration:.2f}\n")
        f.write(f"Mean Duration per Seed (s): {mean_duration_per_seed:.2f}\n")
        if per_class_iou_domains:
            f.write("\nOverall mean per class across domains:\n")
            for cls_name, cls_iou in zip(domain_infos[0]['org_classes'], overall_per_class_iou):
                f.write(f"  {cls_name}: {cls_iou:.2f}\n")
        if pct_domains:
            f.write("\nOverall mean per dataset % across domains:\n")
            for pct in sorted(pct_domains[0].keys()):
                pct_miou = np.mean([dm[pct]['mIoU'] for dm in pct_domains])
                pct_mdice = np.mean([dm[pct]['mDice'] for dm in pct_domains])
                pct_macc = np.mean([dm[pct]['mAcc'] for dm in pct_domains])
                f.write(f"  {int(pct * 100)}%: mIoU {pct_miou:.2f}, mDice {pct_mdice:.2f}, mAcc {pct_macc:.2f}\n")




if __name__ == "__main__":
    # Initial argument parsing to get the method
    initial_parser = argparser()
    initial_args, _ = initial_parser.parse_known_args()

    # Create a new parser with method-specific arguments
    parser = argparser()
    parser = add_method_specific_args(parser, initial_args.method)
    args = parser.parse_args()

    # Set the global random seed for reproducibility
    set_global_seeds(args.seed)

    # Run the main function with the parsed arguments
    main(args)
