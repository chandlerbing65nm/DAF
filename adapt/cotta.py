"""
COTTA: Continual Test-Time Adaptation.

Builds upon: https://github.com/qinenergy/cotta
Corresponding paper: https://arxiv.org/abs/2203.13591

OVSS TTA Adaptation Notes:
===========================
The original COTTA method is designed for classification, where the model outputs
1D logits of shape (batch, num_classes). It uses a teacher-student framework with
three key mechanisms:

1. Teacher-student with EMA: A teacher model (EMA of the student) provides soft
   targets for the student to learn from. The EMA momentum (mt=0.999) ensures
   slow, stable teacher updates.

2. Augmentation-averaged prediction: When the anchor (source) model's confidence
   is low (below threshold ap), the teacher is run on multiple augmented versions
   of the input and predictions are averaged. This reduces noise in the teacher
   signal. When anchor confidence is high, the teacher is run directly.

3. Stochastic restore: With probability rst, individual adapted parameters are
   restored to their initial values. This prevents error accumulation during
   continual adaptation across domain shifts.

For open-vocabulary semantic segmentation (OVSS) test-time adaptation, the key
differences are:
1. Logits are per-pixel: shape (#templates, batch, num_classes, H, W).
   We take template 0 and compute the loss over per-pixel logits. The
   cross-entropy between student and teacher is computed per-pixel, then averaged.
2. Only LayerNorm parameters of the visual encoder are adapted (same as TENT).
   The EMA and anchor models are full deep copies of the CLIP model, but only
   visual LayerNorm params have gradients enabled.
3. The anchor model's confidence is computed as the mean max softmax probability
   over per-pixel predictions, averaged across spatial dimensions.
4. TTA transforms operate on torch tensors (not PIL images), since OVSS inputs
   are already normalized tensors. We use color jitter, random affine, gaussian
   blur, gaussian noise, and horizontal flip — all implemented as tensor ops.
5. Evaluation uses the EMA (teacher) model, matching COTTA's design where the
   teacher provides the final predictions.
6. Text embeddings are pre-computed from class names + prompt templates,
   same as TENT.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from .tent import TENT


class GaussianNoise(torch.nn.Module):
    def __init__(self, mean=0., std=1.):
        super().__init__()
        self.std = std
        self.mean = mean

    def forward(self, img):
        noise = torch.randn(img.size()) * self.std + self.mean
        noise = noise.to(img.device)
        return img + noise


class Clip(torch.nn.Module):
    def __init__(self, min_val=0., max_val=1.):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, img):
        return torch.clip(img, self.min_val, self.max_val)


class ColorJitterPro(torch.nn.Module):
    """Randomly change brightness, contrast, saturation, hue, and gamma."""

    def __init__(self, brightness=(0.6, 1.4), contrast=(0.7, 1.3),
                 saturation=(0.5, 1.5), hue=(-0.06, 0.06), gamma=(0.7, 1.3)):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.gamma = gamma

    def forward(self, img):
        fn_idx = torch.randperm(5)
        for fn_id in fn_idx:
            if fn_id == 0:
                lo, hi = self.brightness
                factor = torch.tensor(1.0).uniform_(lo, hi).item()
                img = TF.adjust_brightness(img, factor)
            elif fn_id == 1:
                lo, hi = self.contrast
                factor = torch.tensor(1.0).uniform_(lo, hi).item()
                img = TF.adjust_contrast(img, factor)
            elif fn_id == 2:
                lo, hi = self.saturation
                factor = torch.tensor(1.0).uniform_(lo, hi).item()
                img = TF.adjust_saturation(img, factor)
            elif fn_id == 3:
                lo, hi = self.hue
                factor = torch.tensor(1.0).uniform_(lo, hi).item()
                img = TF.adjust_hue(img, factor)
            elif fn_id == 4:
                lo, hi = self.gamma
                factor = torch.tensor(1.0).uniform_(lo, hi).item()
                img = img.clamp(1e-8, 1.0)
                img = TF.adjust_gamma(img, factor)
        return img


def get_tta_transforms(img_size, gaussian_std=0.005, soft=False, padding_mode='edge'):
    n_pixels = img_size[0] if isinstance(img_size, (list, tuple)) else img_size

    tta_transforms = [
        Clip(0.0, 1.0),
        ColorJitterPro(
            brightness=[0.8, 1.2] if soft else [0.6, 1.4],
            contrast=[0.85, 1.15] if soft else [0.7, 1.3],
            saturation=[0.75, 1.25] if soft else [0.5, 1.5],
            hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
            gamma=[0.85, 1.15] if soft else [0.7, 1.3]
        ),
        transforms.Pad(padding=int(n_pixels / 2), padding_mode=padding_mode),
        transforms.RandomAffine(
            degrees=[-8, 8] if soft else [-15, 15],
            translate=(1/16, 1/16),
            scale=(0.95, 1.05) if soft else (0.9, 1.1),
            shear=None,
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=0
        ),
        transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]),
        transforms.CenterCrop(size=n_pixels),
        transforms.RandomHorizontalFlip(p=0.5),
        GaussianNoise(0, gaussian_std),
        Clip(0.0, 1.0),
    ]
    return transforms.Compose(tta_transforms)


@torch.no_grad()
def ema_update_model(model_to_update, model_to_merge, momentum, device, update_all=False):
    if momentum < 1.0:
        for param_to_update, param_to_merge in zip(model_to_update.parameters(), model_to_merge.parameters()):
            if param_to_update.requires_grad or update_all:
                param_to_update.data = momentum * param_to_update.data + (1 - momentum) * param_to_merge.data.to(device)
    return model_to_update


class COTTA(TENT):
    """
    Continual Test-Time Adaptation for OVSS.

    Inherits all OVSS model loading, text embedding extraction, LayerNorm
    parameter collection, and reset logic from TENT. Adds a teacher-student
    framework with EMA updates, augmentation-averaged predictions, and
    stochastic weight restoration.
    """

    def __init__(self, ovss_type, ovss_backbone, lr, classes, steps=10,
                 prompt_dir=None, runtime_calculation=False, optimizer='adam',
                 reset_mode='episodic',
                 device='cpu',
                 token_merge=False,
                 merge_type='algm',
                 algm_layers=(1, 7),
                 algm_threshold=0.8,
                 algm_window_size=(2, 2),
                 loss_ent=True, lamb_ent=1.0,
                 loss_div=False, lamb_div=1.0,
                 loss_cmac=False, lamb_cmac=1.0,
                 module_safs=False, alpha_safs=0.5,
                 diag_safs=False, diag_cmac=False,
                 diag_div=False,
                 cotta_mt=0.999,
                 cotta_rst=0.01,
                 cotta_ap=0.92,
                 cotta_n_augmentations=32,
                 ):
        super().__init__(
            ovss_type=ovss_type, ovss_backbone=ovss_backbone, lr=lr, classes=classes,
            steps=steps, prompt_dir=prompt_dir, runtime_calculation=runtime_calculation,
            optimizer=optimizer, reset_mode=reset_mode, device=device,
            token_merge=token_merge, merge_type=merge_type,
            algm_layers=algm_layers, algm_threshold=algm_threshold,
            algm_window_size=algm_window_size,
            loss_ent=loss_ent, lamb_ent=lamb_ent,
            loss_div=loss_div, lamb_div=lamb_div,
            loss_cmac=loss_cmac, lamb_cmac=lamb_cmac,
            module_safs=module_safs, alpha_safs=alpha_safs,
            diag_safs=diag_safs, diag_cmac=diag_cmac, diag_div=diag_div,
        )

        self.mt = cotta_mt
        self.rst = cotta_rst
        self.ap = cotta_ap
        self.n_augmentations = cotta_n_augmentations

        # Setup EMA (teacher) model
        self.model_ema = copy.deepcopy(self.model)
        for param in self.model_ema.parameters():
            param.detach_()

        # Setup anchor (source) model
        self.model_anchor = copy.deepcopy(self.model)
        for param in self.model_anchor.parameters():
            param.detach_()

        # Save initial model state for stochastic restore (LN params only)
        self.model_states = copy.deepcopy(self.model.state_dict())

        # TTA transforms — use input image size from init_resize
        # Images are already tensors; transforms operate on tensor format
        img_size = (224, 224)  # default patch size
        self.transform = get_tta_transforms(img_size)

    def perform_adaptation(self, x, diag_labels=None, diag_ignore_index=255):
        """
        Forward pass with COTTA adaptation. Uses teacher-student framework
        with EMA updates, augmentation-averaged predictions, and stochastic
        weight restoration.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, C, H, W).
            diag_labels: Unused (kept for interface compatibility with TENT).
            diag_ignore_index: Unused (kept for interface compatibility).

        Returns:
            List[float]: Recorded loss values for each adaptation iteration.
        """
        loss_report = []

        for _ in range(self.steps):
            self.optimizer.zero_grad()

            # Student forward pass
            logits, _, _ = self.model(x, self.text_x, True, interpolate=False)
            logits_0 = logits[0]  # (batch, num_classes, H, W)

            # Anchor (source) model confidence
            with torch.no_grad():
                anchor_logits, _, _ = self.model_anchor(x, self.text_x, True, interpolate=False)
                anchor_logits_0 = anchor_logits[0]  # (batch, num_classes, H, W)
                # Per-sample max softmax probability, averaged over spatial dims
                anchor_prob = F.softmax(anchor_logits_0, dim=1).max(dim=1)[0]  # (batch, H, W)
                anchor_prob = anchor_prob.mean(dim=(1, 2))  # (batch,)
                anchor_confidence = anchor_prob.mean(0)

            # Teacher (EMA) prediction
            with torch.no_grad():
                if anchor_confidence < self.ap:
                    # Augmentation-averaged prediction
                    ema_outputs = []
                    for _ in range(self.n_augmentations):
                        x_aug = self.transform(x)
                        logits_ema, _, _ = self.model_ema(x_aug, self.text_x, True, interpolate=False)
                        ema_outputs.append(logits_ema[0])  # (batch, num_classes, H, W)
                    outputs_ema = torch.stack(ema_outputs).mean(0)  # (batch, num_classes, H, W)
                else:
                    logits_ema, _, _ = self.model_ema(x, self.text_x, True, interpolate=False)
                    outputs_ema = logits_ema[0]  # (batch, num_classes, H, W)

            # Cross-entropy loss between student and teacher (per-pixel)
            # Using symmetric cross-entropy like the imagenet variant
            loss = -0.5 * (outputs_ema.softmax(1) * logits_0.log_softmax(1)).sum(1) \
                   -0.5 * (logits_0.softmax(1) * outputs_ema.log_softmax(1)).sum(1)
            loss = loss.mean()
            loss_report.append(loss.item())

            # Backward and optimize
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Teacher (EMA) update
            self.model_ema = ema_update_model(
                self.model_ema, self.model, self.mt, self.device, update_all=True
            )

            # Stochastic restore
            with torch.no_grad():
                if self.rst > 0.:
                    for nm, m in self.model.named_modules():
                        for npp, p in m.named_parameters():
                            if npp in ['weight', 'bias'] and p.requires_grad:
                                mask = (torch.rand(p.shape) < self.rst).float().to(self.device)
                                p.data = self.model_states[f"{nm}.{npp}"] * mask + p * (1. - mask)

        return loss_report

    @torch.no_grad()
    def evaluate(self, x):
        """
        Forward pass without adaptation, using the EMA (teacher) model.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, C, H, W).

        Returns:
            torch.Tensor: Per-class logits of shape (batch_size, num_classes, H, W).
        """
        logits, _, _ = self.model_ema(x, self.text_x, True, interpolate=True)
        return logits[0]

    def reset(self):
        """Reset student model, optimizer, EMA model, and anchor model to initial states."""
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("Cannot reset without saved model/optimizer state")
        self.load_model_and_optimizer(self.model, self.optimizer,
                                      self.model_state, self.optimizer_state)
        # Reset EMA and anchor models to initial student state
        self.model_ema = copy.deepcopy(self.model)
        for param in self.model_ema.parameters():
            param.detach_()
        self.model_anchor = copy.deepcopy(self.model)
        for param in self.model_anchor.parameters():
            param.detach_()
        self.model_states = copy.deepcopy(self.model.state_dict())
