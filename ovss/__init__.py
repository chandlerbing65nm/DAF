import os
import ovss.clip as clip
from ovss.clip import tokenize as clip_tokenize

CLIP_DOWNLOAD_ROOT = os.environ.get("CLIP_DOWNLOAD_ROOT", "/scratch/project_465002853/clip_cache_doloriel")


def load_ovss(
    ovss_type,
    ovss_backbone,
    device='cpu',
    token_merge=False,
    merge_type='algm',
    algm_layers=(1, 7),
    algm_threshold=0.8,
    algm_window_size=(2, 2),
):
    """
    Load the OVSS model based on the specified type and backbone.

    Args:
        ovss_type: Type of the OVSS model.
        ovss_backbone: Backbone architecture of the OVSS model.
        device: Device to load the model on (e.g., 'cpu' or 'cuda').

    Returns:
        ovss_model: Loaded OVSS model.
    """
    os.makedirs(CLIP_DOWNLOAD_ROOT, exist_ok=True)

    if ovss_type == 'clip':
        arch = "vanilla"
        attn_strategy = "vanilla"
        gaussian_std = 5.0
        ovss_model, _ = clip.load(ovss_backbone, device, download_root=CLIP_DOWNLOAD_ROOT)
        ovss_model.visual.set_params(arch, attn_strategy, gaussian_std)
        tokenize = clip_tokenize

    elif ovss_type == 'sclip':
        arch = "vanilla"
        attn_strategy = "csa"
        gaussian_std = 5.0
        ovss_model, _ = clip.load(ovss_backbone, device, download_root=CLIP_DOWNLOAD_ROOT)
        ovss_model.visual.set_params(arch, attn_strategy, gaussian_std)
        tokenize = clip_tokenize

    elif ovss_type == 'naclip':
        arch = "reduced"
        attn_strategy = "naclip"
        gaussian_std = 5.0
        ovss_model, _ = clip.load(ovss_backbone, device, download_root=CLIP_DOWNLOAD_ROOT)
        ovss_model.visual.set_params(arch, attn_strategy, gaussian_std)
        tokenize = clip_tokenize

    else:
        raise ValueError(f"Unsupported OVSS type: {ovss_type}")

    if token_merge:
        if merge_type != 'algm':
            raise ValueError(f"Unsupported merge_type: {merge_type}")
        if not str(ovss_backbone).startswith('ViT'):
            raise ValueError("ALGM token merging is only supported for ViT OVSS backbones.")

        from ovss.clip.algm import apply_patch as apply_algm_patch

        apply_algm_patch(
            ovss_model.visual,
            selected_layers=algm_layers,
            threshold=algm_threshold,
            window_size=algm_window_size,
            merge_type=merge_type,
        )

    return ovss_model, tokenize
