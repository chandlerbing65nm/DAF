import time
import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ovss import load_ovss
from optim import get_optimizer
from utils.misc import load_prompts_from_yaml, print_clip_parameters, print_optimizer_parameters

REFERENCE_PROMPT = 'a photo of a {}'


class TENT:
    """
    Test-time adaptation for open-vocabulary semantic segmentation (OVSS) models using TENT.

    Performs iterative optimization of the visual encoder LayerNorm parameters to reduce predictive uncertainty 
    based on the softmax output distribution.

    Inspired by TENT GitHub: https://github.com/DequanWang/tent
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
                 loss_src_cons=False, lamb_src_cons=1.0,
                 loss_feat_cons=False, lamb_feat_cons=1.0, feat_cons_type='cosine',
                 module_safs=False, alpha_safs=0.5,
                 diag_safs=False, diag_cmac=False,
                 diag_div=False,
                 ):
        """
        Initialize the TENT adaptation module.

        Args:
            ovss_type (str): Identifier for the open-vocabulary segmentation model to load.
            ovss_backbone (str): Name of the backbone architecture within the OVSS model.
            lr (float): Learning rate for the LayerNorm optimizer.
            classes (List[str]): List of class names for prompt generation.
            steps (int, optional): Number of adaptation iterations per sample. Defaults to 10.
            prompt_dir (str or None, optional): Path to YAML file with prompt templates. Defaults to None.
            runtime_calculation (bool, optional): Whether to record adaptation/evaluation runtimes. Defaults to False.
            device (str, optional): Compute device, e.g., 'cpu' or 'cuda'. Defaults to 'cpu'.
        """

        self.ovss_type = ovss_type
        self.ovss_backbone = ovss_backbone
        self.lr = lr

        if classes is not None:
            self.classes = classes
        else:
            raise Exception("Classes are required in the init")
        
        self.prompt_dir = prompt_dir
        self.steps = steps
        self.runtime = runtime_calculation
        self.optimizer_name = optimizer
        self.reset_mode = reset_mode
        self.device = device
        self.token_merge = token_merge
        self.merge_type = merge_type
        self.algm_layers = algm_layers
        self.algm_threshold = algm_threshold
        self.algm_window_size = algm_window_size
        self.loss_ent = loss_ent
        self.lamb_ent = lamb_ent
        self.loss_div = loss_div
        self.lamb_div = lamb_div
        self.loss_cmac = loss_cmac
        self.lamb_cmac = lamb_cmac
        self.loss_src_cons = loss_src_cons
        self.lamb_src_cons = lamb_src_cons
        self.loss_feat_cons = loss_feat_cons
        self.lamb_feat_cons = lamb_feat_cons
        self.feat_cons_type = feat_cons_type
        self.module_safs = module_safs
        self.alpha_safs = alpha_safs
        self.safs_stats = []
        self.diag_safs = diag_safs
        self.diag_safs_stats = []
        self.diag_cmac = diag_cmac
        self.diag_cmac_stats = []
        self.diag_cmac_step_offset = 0
        self.diag_div = diag_div
        self.diag_div_stats = []
        self.diag_div_step_offset = 0
        self.source_model = None

        # Load the OVSS model and tokenizer
        self.model, self.tokenize = load_ovss(
            self.ovss_type,
            self.ovss_backbone,
            device=self.device,
            token_merge=self.token_merge,
            merge_type=self.merge_type,
            algm_layers=self.algm_layers,
            algm_threshold=self.algm_threshold,
            algm_window_size=self.algm_window_size,
        )

        if self.prompt_dir:
            # Load the prompt templates
            self.prompt_templates = load_prompts_from_yaml(self.prompt_dir)
            # print the number of prompt templates
            print(f"Number of prompt templates: {len(self.prompt_templates)}")
        else:
            self.prompt_templates = [REFERENCE_PROMPT]

        # Set the gradients for LayerNorm layers only for visual encoder
        self.model.transformer.requires_grad_(False)
        self.model.ln_final.requires_grad_(False)
        self.model.token_embedding.requires_grad_(False)

        self.model.visual = self.set_ln_grads(self.model.visual)

        # Collect the LayerNorm parameters
        params, _ = self.collect_ln_params(self.model.visual)

        # print the parameters
        print_clip_parameters(self.model)

        # Set the optimizer
        self.optimizer = get_optimizer(params, optimizer_name=self.optimizer_name, lr=self.lr)

        # print the parameters passed to the optimizer
        print_optimizer_parameters(self.optimizer, self.model)

        # Save the initial model and optimizer states
        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(self.model, self.optimizer)

        # extracting text features
        with torch.no_grad():
            self.text_x = self.extract_text_embeddings(self.classes, self.prompt_templates, average=False).squeeze() # (class, 512)

        # Create source model for CMAC / SAFS / diagnostics
        if self.loss_cmac or self.module_safs or self.diag_safs or self.diag_cmac:
            self.source_model = copy.deepcopy(self.model)
            self.source_model.requires_grad_(False)
            self.source_model.eval()

        # define variables to store adaptation and evaluation duration
        if self.runtime:
            self.adapt_times = []
            self.eval_times = []

    def adapt(self, x, diag_labels=None, diag_ignore_index=255):
        """
        Forward pass with adaptation.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, C, H, W).

        Returns:
            List[float]: Loss values recorded at each adaptation iteration.
        """

        if self.reset_mode == 'episodic':
            self.reset()
        loss_report = self.perform_adaptation(x, diag_labels=diag_labels, diag_ignore_index=diag_ignore_index)
        return loss_report

    @torch.no_grad() 
    def evaluate(self, x):
        """
        Forward pass without adaptation.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, C, H, W).

        Returns:
            torch.Tensor: Per-class logits of shape (batch_size, num_classes, H, W).

        """

        t1 = time.time()
        logits, _, _ = self.model(x, self.text_x, True, 
                                  interpolate=True) # (#template, batch_size, #classes, H, W)
        logits = logits[0]
        t2 = time.time()
        if self.runtime:
            self.eval_times.append(t2-t1)

        return logits

    def reset(self):
        """
        Resets the model and optimizer to their initial states.
        """
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("Cannot reset without saved model/optimizer state")
        self.load_model_and_optimizer(self.model, self.optimizer,
                                      self.model_state, self.optimizer_state)

    def perform_adaptation(self, x, diag_labels=None, diag_ignore_index=255):
        """
        Forward pass with adaptation for test-time. The model adapts itself during testing by updating on every forward pass.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, C, H, W).
            diag_labels (torch.Tensor, optional): Ground-truth labels for SAFS diagnostics.
            diag_ignore_index (int): Ignore index for mIoU computation in diagnostics.
        
        Returns:
            List[float]: Recorded loss values for each adaptation iteration.
        """

        t1 = time.time()
        loss_report = []
        safs_step_counts = []
        needs_source = self.loss_cmac or self.module_safs or self.diag_safs or self.diag_cmac or self.loss_src_cons or self.loss_feat_cons
        for iter in range(self.steps):
            logits, image_features, _ = self.model(x, self.text_x, True, 
                                      interpolate=False)  # (#template, batch_size, #classes, H, W)

            logits_src = None
            image_features_src = None
            if needs_source:
                with torch.no_grad():
                    logits_src, image_features_src, _ = self.source_model(x, self.text_x, True, interpolate=False)

            keep_mask = None
            if self.module_safs and image_features_src is not None:
                f_adapt_patches = image_features[:, 1:]
                f_src_patches = image_features_src[:, 1:]
                cos_sim = (f_adapt_patches * f_src_patches).sum(dim=-1)
                shift = (1.0 - cos_sim).mean(dim=-1)
                threshold = shift.mean() - self.alpha_safs * shift.std(unbiased=False)
                keep_mask = shift > threshold
                if not keep_mask.any():
                    keep_mask = None
                elif not keep_mask.all().item():
                    logits = logits[:, keep_mask]
                    image_features = image_features[keep_mask]
                    image_features_src = image_features_src[keep_mask]
                    if logits_src is not None:
                        logits_src = logits_src[:, keep_mask]

            loss = torch.tensor(0.0, device=x.device)

            if self.loss_ent:
                entropy_per_pixel = self.softmax_entropy(logits)
                loss = loss + self.lamb_ent * entropy_per_pixel.mean()

            if self.loss_div:
                div_loss = self.diversity_loss(logits)
                loss = loss + self.lamb_div * div_loss

            if self.loss_cmac:
                cmac_loss = self.cmac_loss(image_features, image_features_src, self.text_x)
                loss = loss + cmac_loss

            if self.loss_src_cons and logits_src is not None:
                src_cons_loss = self.consistency_loss(logits, logits_src)
                loss = loss + self.lamb_src_cons * src_cons_loss

            if self.loss_feat_cons and image_features_src is not None:
                feat_cons_loss = self.feature_consistency_loss(image_features, image_features_src)
                loss = loss + self.lamb_feat_cons * feat_cons_loss

            loss_report.append(loss.item())

            total_samples = x.shape[0]
            if self.module_safs and keep_mask is not None:
                kept_samples = keep_mask.sum().item()
            else:
                kept_samples = total_samples
            filtered_samples = total_samples - kept_samples
            safs_step_counts.append((total_samples, kept_samples, filtered_samples))

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.diag_safs and image_features_src is not None:
                with torch.no_grad():
                    f_adapt_patches = image_features[:, 1:]
                    f_src_patches = image_features_src[:, 1:]
                    cos_sim_diag = (f_adapt_patches * f_src_patches).sum(dim=-1)
                    feature_shift = (1.0 - cos_sim_diag).mean(dim=-1)

                    entropy_per_sample = self.softmax_entropy(logits).mean(dim=(0, -2, -1))

                    p_src = F.softmax(logits_src, dim=-3)
                    p_adapt = F.softmax(logits, dim=-3)
                    pred_change = F.kl_div(p_adapt.log(), p_src, reduction='none').sum(dim=-3).mean(dim=(0, -2, -1))

                    src_argmax = logits_src.argmax(dim=-3)
                    adapt_argmax = logits.argmax(dim=-3)
                    pred_agreement = (src_argmax == adapt_argmax).float().mean(dim=(0, -2, -1))

                    miou_change_per_sample = None
                    if diag_labels is not None:
                        pred_adapt_raw = adapt_argmax[0]
                        pred_src_raw = src_argmax[0]
                        if pred_adapt_raw.shape[-2:] != diag_labels.shape[-2:]:
                            pred_adapt_raw = F.interpolate(
                                pred_adapt_raw.unsqueeze(1).float(), size=diag_labels.shape[-2:], mode='nearest'
                            ).squeeze(1).long()
                            pred_src_raw = F.interpolate(
                                pred_src_raw.unsqueeze(1).float(), size=diag_labels.shape[-2:], mode='nearest'
                            ).squeeze(1).long()
                        num_classes = len(self.classes)
                        miou_change_per_sample = []
                        for i in range(pred_adapt_raw.shape[0]):
                            miou_a = self._compute_sample_miou(pred_adapt_raw[i], diag_labels[i], num_classes, diag_ignore_index)
                            miou_s = self._compute_sample_miou(pred_src_raw[i], diag_labels[i], num_classes, diag_ignore_index)
                            miou_change_per_sample.append(miou_a - miou_s)

                    for i in range(feature_shift.shape[0]):
                        entry = {
                            'feature_shift': feature_shift[i].item(),
                            'entropy': entropy_per_sample[i].item(),
                            'pred_change': pred_change[i].item(),
                            'pred_agreement': pred_agreement[i].item(),
                        }
                        if miou_change_per_sample is not None:
                            entry['miou_change'] = miou_change_per_sample[i]
                        self.diag_safs_stats.append(entry)

            if self.diag_cmac and image_features_src is not None:
                with torch.no_grad():
                    if self.text_x.dim() == 3:
                        T = self.text_x.mean(dim=0)
                    else:
                        T = self.text_x
                    num_classes = T.shape[0]

                    f_adapt = image_features[:, 1:]
                    f_src = image_features_src[:, 1:]

                    sim_adapt = f_adapt @ T.t()
                    sim_src = f_src @ T.t()

                    assigned = sim_src.argmax(dim=-1)

                    sim_adapt_assigned = sim_adapt.gather(-1, assigned.unsqueeze(-1)).squeeze(-1)
                    sim_src_assigned = sim_src.gather(-1, assigned.unsqueeze(-1)).squeeze(-1)

                    mask_assigned = F.one_hot(assigned, num_classes=num_classes).bool()
                    sim_adapt_unassigned = sim_adapt.masked_fill(mask_assigned, float('-inf'))
                    closest_unassigned = sim_adapt_unassigned.max(dim=-1).values

                    adapt_assignments = sim_adapt.argmax(dim=-1)
                    class_counts = torch.bincount(adapt_assignments.reshape(-1), minlength=num_classes).float()
                    total_patches = class_counts.sum()
                    class_fractions = (class_counts / total_patches).tolist()

                    step_idx = self.diag_cmac_step_offset + len(self.diag_cmac_stats)
                    self.diag_cmac_stats.append({
                        'step': step_idx,
                        'assigned_sim_mean': sim_adapt_assigned.mean().item(),
                        'assigned_sim_std': sim_adapt_assigned.std().item(),
                        'src_assigned_sim_mean': sim_src_assigned.mean().item(),
                        'unassigned_sim_mean': closest_unassigned.mean().item(),
                        'unassigned_sim_std': closest_unassigned.std().item(),
                        'class_fractions': class_fractions,
                        'num_classes': num_classes,
                    })

            if self.diag_div:
                with torch.no_grad():
                    preds = logits.argmax(dim=-3)
                    num_classes = logits.shape[-3]
                    class_counts = torch.bincount(preds.reshape(-1), minlength=num_classes).float()
                    total = class_counts.sum()
                    class_fractions = (class_counts / total)
                    hhi = (class_fractions ** 2).sum().item()
                    num_active = (class_fractions > 0.01).sum().item()
                    max_share = class_fractions.max().item()

                    absent_class_mass = None
                    absent_class_fpr = None
                    present_classes = None
                    absent_classes = None
                    num_present = None
                    num_absent = None
                    if diag_labels is not None:
                        valid_labels = diag_labels[diag_labels != diag_ignore_index]
                        present_classes = sorted(set(valid_labels.unique().tolist()))
                        absent_classes = [c for c in range(num_classes) if c not in present_classes]
                        num_present = len(present_classes)
                        num_absent = len(absent_classes)
                        if absent_classes:
                            absent_idx = torch.tensor(absent_classes, device=logits.device)
                            p = logits.softmax(dim=-3)
                            absent_mass = p.index_select(-3, absent_idx).sum(dim=-3).mean().item()
                            absent_class_mass = absent_mass
                            absent_pred_mask = torch.zeros_like(preds, dtype=torch.bool)
                            for c in absent_classes:
                                absent_pred_mask = absent_pred_mask | (preds == c)
                            absent_class_fpr = absent_pred_mask.float().mean().item()
                        else:
                            absent_class_mass = 0.0
                            absent_class_fpr = 0.0

                    step_idx = self.diag_div_step_offset + len(self.diag_div_stats)
                    self.diag_div_stats.append({
                        'step': step_idx,
                        'class_fractions': class_fractions.tolist(),
                        'hhi': hhi,
                        'num_active_classes': num_active,
                        'max_class_share': max_share,
                        'num_classes': num_classes,
                        'present_classes': present_classes,
                        'absent_classes': absent_classes,
                        'num_present_classes': num_present,
                        'num_absent_classes': num_absent,
                        'absent_class_mass': absent_class_mass,
                        'absent_class_fpr': absent_class_fpr,
                    })

        if safs_step_counts:
            n_steps = len(safs_step_counts)
            avg_total = sum(s[0] for s in safs_step_counts) / n_steps
            avg_kept = sum(s[1] for s in safs_step_counts) / n_steps
            avg_filtered = sum(s[2] for s in safs_step_counts) / n_steps
            self.safs_stats.append({
                'total': avg_total, 'kept': avg_kept, 'filtered': avg_filtered
            })

        t2 = time.time()
        if self.runtime:
            self.adapt_times.append(t2-t1)

        return loss_report

    def extract_text_embeddings(self, class_names, prompts, average=True):
        """
        Extracts text embeddings for given class names and prompts.
        Args:
            class_names: List of class names to generate text embeddings for.
            prompts: List of prompt templates to use for generating text embeddings.
            average: Boolean indicating whether to average the embeddings of different templates for each class.
        Returns:
            text_features: Tensor of text embeddings for the given class names and prompts.
        """
        text_features = []
        for class_name in class_names:
            texts = [p.format(class_name) for p in prompts]
            texts = self.tokenize(texts).to(self.device)
            class_embeddings = self.model.encode_text(texts)  # Shape: (#templates, 512)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            if average:
                class_embeddings_avg = class_embeddings.mean(dim=0)  # Shape: (512,)
                class_embeddings_avg = class_embeddings_avg / class_embeddings_avg.norm()
                # add the averaged embeddings to the original embeddings
                class_embeddings = torch.cat([class_embeddings, class_embeddings_avg.unsqueeze(0)], dim=0)
            text_features.append(class_embeddings)
        text_features = torch.stack(text_features, dim=1).to(self.device)
        return text_features

    @staticmethod
    def set_ln_grads(model):
        """
        Set gradient settings for LayerNorm layers within the model, disabling gradients globally except for these LN layers.
        Args:
            model: The model whose LayerNorm layers' gradients are to be set.
        Returns:
            The model with modified gradient settings.
        """
        model.requires_grad_(False)
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                m.requires_grad_(True)
        return model

    @staticmethod
    def collect_ln_params(model):
        """
        Collect the affine scale and shift parameters from LayerNorm layers.
        Args:
            model: The model from which to collect LayerNorm parameters.
        Returns:
            params: List of LayerNorm parameters.
            names: List of parameter names.
        """
        params = []
        names = []
        for nm, m in model.named_modules():
            if isinstance(m, nn.LayerNorm):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:
                        params.append(p)
                        names.append(f"visual.{nm}.{np}")
        return params, names

    @staticmethod
    def copy_model_and_optimizer(model, optimizer):
        """
        Copy the model and optimizer states for resetting after adaptation.
        Args:
            model: The model to copy.
            optimizer: The optimizer to copy.
        Returns:
            model_state: Copied state of the model.
            optimizer_state: Copied state of the optimizer.
        """
        model_state = copy.deepcopy(model.state_dict())
        optimizer_state = copy.deepcopy(optimizer.state_dict())
        return model_state, optimizer_state

    @staticmethod
    def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
        """
        Restore the model and optimizer states from copies.
        Args:
            model: The model to restore.
            optimizer: The optimizer to restore.
            model_state: The state to restore the model to.
            optimizer_state: The state to restore the optimizer to.
        """
        model.load_state_dict(model_state, strict=True)
        optimizer.load_state_dict(optimizer_state)

    @staticmethod
    def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
        """Entropy of softmax distribution from logits.
            x : torch.Tensor : logits of shape (#templates, batch_size, num_classes, H, W)
        """
        return -(x.softmax(-3) * x.log_softmax(-3)).sum(-3)

    @staticmethod
    def diversity_loss(logits: torch.Tensor) -> torch.Tensor:
        """Class-wise diversity loss to prevent model collapse.

        Maximizes the entropy of the marginal class distribution across pixels,
        preventing all pixels from collapsing to a single class.

        Args:
            logits: [#templates, batch, #classes, W, H]
        Returns:
            scalar loss (minimize to maximize class diversity)
        """
        p = logits.softmax(dim=-3)
        marginal = p.mean(dim=(1, 3, 4))  # [#templates, #classes]
        entropy = -(marginal * torch.log(marginal + 1e-8)).sum(dim=-1)  # [#templates]
        return -entropy.mean()

    def consistency_loss(self, logits_orig: torch.Tensor, logits_aug: torch.Tensor) -> torch.Tensor:
        """Pixel-wise consistency loss via KL divergence.

        Args:
            logits_orig: [#templates, batch, #classes, W, H]
            logits_aug: [#templates, batch, #classes, W, H]
        Returns:
            scalar loss
        """
        p = logits_orig.softmax(dim=-3)
        q = logits_aug.softmax(dim=-3)
        log_p = logits_orig.log_softmax(dim=-3)
        log_q = logits_aug.log_softmax(dim=-3)
        kl_pq = (p * (log_p - log_q)).sum(dim=-3)
        kl_qp = (q * (log_q - log_p)).sum(dim=-3)
        return 0.5 * (kl_pq + kl_qp).mean()

    def feature_consistency_loss(self, image_features_adapt: torch.Tensor,
                                 image_features_src: torch.Tensor) -> torch.Tensor:
        """Feature-level consistency loss between adapted and source features.

        Args:
            image_features_adapt: [batch, tokens, dim] from adapted model (normalized)
            image_features_src: [batch, tokens, dim] from frozen source (normalized, no grad)
        Returns:
            scalar loss
        """
        f_adapt = image_features_adapt[:, 1:]
        f_src = image_features_src[:, 1:]
        if self.feat_cons_type == 'l2':
            return ((f_adapt - f_src) ** 2).mean()
        else:
            return (1.0 - (f_adapt * f_src).sum(dim=-1)).mean()

    def cmac_loss(self, image_features_adapt: torch.Tensor,
                   image_features_src: torch.Tensor,
                   text_features: torch.Tensor) -> torch.Tensor:
        """Cross-Modal Anchor Consistency: feature-level text-anchored drift.

        Args:
            image_features_adapt: [batch, tokens, dim] from adapted model (normalized)
            image_features_src: [batch, tokens, dim] from frozen source (normalized, no grad)
            text_features: [#templates, #classes, dim] or [#classes, dim] (normalized)
        Returns:
            scalar loss
        """
        if text_features.dim() == 3:
            T = text_features.mean(dim=0)
        else:
            T = text_features
        num_classes = T.shape[0]

        f_adapt = image_features_adapt[:, 1:]
        f_src = image_features_src[:, 1:]

        logit_scale = self.model.logit_scale.exp()
        sim_adapt = logit_scale * (f_adapt @ T.t())
        sim_src = logit_scale * (f_src @ T.t())

        assigned = sim_src.argmax(dim=-1)

        sim_adapt_assigned = sim_adapt.gather(-1, assigned.unsqueeze(-1)).squeeze(-1)
        sim_src_assigned = sim_src.gather(-1, assigned.unsqueeze(-1)).squeeze(-1)

        loss_away = F.relu(sim_src_assigned - sim_adapt_assigned).mean()

        loss_toward_all = F.relu(sim_adapt - sim_src)
        mask = F.one_hot(assigned, num_classes=num_classes).bool()
        loss_toward_all = loss_toward_all.masked_fill(mask, 0)
        loss_toward = (loss_toward_all.sum(dim=-1) / (num_classes - 1)).mean()

        return self.lamb_cmac * (loss_away + loss_toward)

    @staticmethod
    def _compute_sample_miou(pred, label, num_classes, ignore_index=255):
        valid = label != ignore_index
        if valid.sum() == 0:
            return 0.0
        ious = []
        for c in range(num_classes):
            pred_c = (pred == c) & valid
            label_c = (label == c) & valid
            intersection = (pred_c & label_c).sum().item()
            union = (pred_c | label_c).sum().item()
            if union > 0:
                ious.append(intersection / union)
        if not ious:
            return 0.0
        return sum(ious) / len(ious)
