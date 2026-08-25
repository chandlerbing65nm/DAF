import time
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ovss import load_ovss
from optim import get_optimizer
from utils.misc import load_prompts_from_yaml, print_clip_parameters, print_optimizer_parameters

REFERENCE_PROMPT = 'a photo of a {}'


class METHOD:
    """
    Configurable TTA method for OVSS models.

    Extends TENT-style entropy minimization with selective parameter training:
    optionally train LayerNorm and/or attention layers in the visual encoder,
    restricted to the last K transformer blocks.
    """

    def __init__(self, ovss_type, ovss_backbone, lr, classes, steps=10,
                 prompt_dir=None, runtime_calculation=False, optimizer='adam',
                 reset_mode='episodic', device='cpu',
                 train_imag_norm=True, last_imag_k_norm=0,
                 train_imag_attn=False, last_imag_k_attn=0,
                 train_text_norm=False, last_text_k_norm=0,
                 loss_ent=True, lamb_ent=1.0,
                 loss_div=False, lamb_div=1.0,
                 loss_aug_cons=False, lamb_aug_cons=1.0,
                 loss_src_cons=False, lamb_src_cons=1.0,
                 loss_cmac=False, lamb_cmac=1.0,
                 loss_pba=False, lamb_pba=1.0,
                 updownsample=1.0,
                 prompt_average=False,
                 cons_type='sym_kl',
                 module_safs=False, alpha_safs=0.5,
                 diag_safs=False,
                 diag_cmac=False,
                 diag_div=False,
                 token_merge=False,
                 merge_type='algm',
                 algm_layers=(1, 7),
                 algm_threshold=0.8,
                 algm_window_size=(2, 2)):

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

        self.train_imag_norm = train_imag_norm
        self.last_imag_k_norm = last_imag_k_norm
        self.train_imag_attn = train_imag_attn
        self.last_imag_k_attn = last_imag_k_attn
        self.train_text_norm = train_text_norm
        self.last_text_k_norm = last_text_k_norm
        self.loss_ent = loss_ent
        self.lamb_ent = lamb_ent
        self.loss_div = loss_div
        self.lamb_div = lamb_div
        self.loss_aug_cons = loss_aug_cons
        self.lamb_aug_cons = lamb_aug_cons
        self.loss_src_cons = loss_src_cons
        self.lamb_src_cons = lamb_src_cons
        self.loss_cmac = loss_cmac
        self.lamb_cmac = lamb_cmac
        self.loss_pba = loss_pba
        self.lamb_pba = lamb_pba
        self.updownsample = updownsample
        self.prompt_average = prompt_average
        self.cons_type = cons_type
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
        self.eval_size = None
        self.token_merge = token_merge
        self.merge_type = merge_type
        self.algm_layers = algm_layers
        self.algm_threshold = algm_threshold
        self.algm_window_size = algm_window_size

        self.source_model = None

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
            self.prompt_templates = load_prompts_from_yaml(self.prompt_dir)
            print(f"Number of prompt templates: {len(self.prompt_templates)}")
        else:
            self.prompt_templates = [REFERENCE_PROMPT]

        # Freeze text encoder
        self.model.transformer.requires_grad_(False)
        self.model.ln_final.requires_grad_(False)
        self.model.token_embedding.requires_grad_(False)

        # Freeze visual encoder entirely, then selectively unfreeze
        self.model.visual.requires_grad_(False)

        params = []
        num_blocks = len(self.model.visual.transformer.resblocks)

        if self.train_imag_norm:
            ln_params = self._collect_ln_params(self.model.visual, num_blocks)
            for p in ln_params:
                p.requires_grad_(True)
            params.extend(ln_params)

        if self.train_imag_attn:
            attn_params = self._collect_attn_params(self.model.visual, num_blocks)
            for p in attn_params:
                p.requires_grad_(True)
            params.extend(attn_params)

        if self.train_text_norm:
            text_num_blocks = len(self.model.transformer.resblocks)
            text_ln_params = self._collect_text_ln_params(text_num_blocks)
            for p in text_ln_params:
                p.requires_grad_(True)
            params.extend(text_ln_params)

        if len(params) == 0:
            raise ValueError("No trainable parameters selected. Enable --train_imag_norm, --train_imag_attn, or --train_text_norm.")

        print_clip_parameters(self.model)
        self.optimizer = get_optimizer(params, optimizer_name=self.optimizer_name, lr=self.lr)
        print_optimizer_parameters(self.optimizer, self.model)

        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(self.model, self.optimizer)

        if self.loss_src_cons or self.loss_cmac or self.module_safs or self.diag_safs or self.diag_cmac:
            self.source_model = copy.deepcopy(self.model)
            self.source_model.requires_grad_(False)
            self.source_model.eval()

        if self.runtime:
            self.adapt_times = []
            self.eval_times = []

    def _collect_ln_params(self, visual, num_blocks):
        params = []
        for nm, m in visual.named_modules():
            if isinstance(m, nn.LayerNorm):
                for np_name, p in m.named_parameters():
                    if np_name in ['weight', 'bias']:
                        if self.last_imag_k_norm > 0:
                            block_idx = self._get_block_index(nm, num_blocks)
                            if block_idx is not None and block_idx < num_blocks - self.last_imag_k_norm:
                                continue
                        params.append(p)
        return params

    def _collect_attn_params(self, visual, num_blocks):
        params = []
        for nm, m in visual.named_modules():
            if isinstance(m, nn.MultiheadAttention):
                for np_name, p in m.named_parameters():
                    if np_name in ['in_proj_weight', 'in_proj_bias', 'out_proj.weight', 'out_proj.bias']:
                        if self.last_imag_k_attn > 0:
                            block_idx = self._get_block_index(nm, num_blocks)
                            if block_idx is not None and block_idx < num_blocks - self.last_imag_k_attn:
                                continue
                        params.append(p)
        return params

    def _collect_text_ln_params(self, num_blocks):
        params = []
        for nm, m in self.model.transformer.named_modules():
            if isinstance(m, nn.LayerNorm):
                for np_name, p in m.named_parameters():
                    if np_name in ['weight', 'bias']:
                        if self.last_text_k_norm > 0:
                            block_idx = self._get_text_block_index(nm, num_blocks)
                            if block_idx is not None and block_idx < num_blocks - self.last_text_k_norm:
                                continue
                        params.append(p)
        if self.last_text_k_norm == 0 or num_blocks <= self.last_text_k_norm:
            for np_name, p in self.model.ln_final.named_parameters():
                if np_name in ['weight', 'bias']:
                    params.append(p)
        return params

    @staticmethod
    def _get_text_block_index(module_name, num_blocks):
        for i in range(num_blocks):
            if f'resblocks.{i}.' in module_name:
                return i
        return None

    @staticmethod
    def _get_block_index(module_name, num_blocks):
        for i in range(num_blocks):
            if f'transformer.resblocks.{i}.' in module_name:
                return i
        return None

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

    def adapt(self, x, diag_labels=None, diag_ignore_index=255):
        if self.reset_mode == 'episodic':
            self.reset()
        loss_report = self.perform_adaptation(x, diag_labels=diag_labels, diag_ignore_index=diag_ignore_index)
        return loss_report

    @torch.no_grad()
    def evaluate(self, x):
        t1 = time.time()
        text_features = self.extract_text_embeddings(self.classes, self.prompt_templates, average=self.prompt_average).squeeze()
        logits, _, _ = self.model(x, text_features, True, interpolate=False)
        # logits: [#templates, batch, #classes, native_h, native_w]

        native_h, native_w = logits.shape[-2], logits.shape[-1]
        input_h, input_w = x.shape[-2], x.shape[-1]
        target_h = round(native_h + (input_h - native_h) * self.updownsample)
        target_w = round(native_w + (input_w - native_w) * self.updownsample)
        self.eval_size = target_h

        if target_h != native_h or target_w != native_w:
            temp_dim, b_dim, c_dim = logits.shape[0], logits.shape[1], logits.shape[2]
            logits = logits.reshape(-1, c_dim, native_h, native_w)
            logits = F.interpolate(logits, size=(target_h, target_w), mode='bilinear', align_corners=False)
            logits = logits.view(temp_dim, b_dim, c_dim, target_h, target_w)

        logits = logits[0]  # [batch, #classes, target_h, target_w]
        t2 = time.time()
        if self.runtime:
            self.eval_times.append(t2 - t1)
        return logits

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception('Cannot reset without saved model/optimizer state')
        self.load_model_and_optimizer(self.model, self.optimizer, self.model_state, self.optimizer_state)

    def perform_adaptation(self, x, diag_labels=None, diag_ignore_index=255):
        t1 = time.time()
        loss_report = []
        safs_step_counts = []
        for _ in range(self.steps):
            text_features = self.extract_text_embeddings(self.classes, self.prompt_templates, average=self.prompt_average).squeeze()
            logits, image_features, _ = self.model(x, text_features, True, interpolate=False)

            logits_src = None
            image_features_src = None
            if self.loss_src_cons or self.loss_cmac or self.module_safs or self.diag_safs or self.diag_cmac:
                with torch.no_grad():
                    logits_src, image_features_src, _ = self.source_model(x, text_features, True, interpolate=False)

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

            if self.loss_pba:
                pba = self.pba_loss(image_features, text_features)
                loss = loss + self.lamb_pba * pba

            if self.loss_div:
                div_loss = self.diversity_loss(logits)
                loss = loss + self.lamb_div * div_loss

            if self.loss_aug_cons:
                x_flip = torch.flip(x, dims=[-1])
                logits_flip, _, _ = self.model(x_flip, text_features, True, interpolate=False)
                logits_flip = torch.flip(logits_flip, dims=[-1])
                if keep_mask is not None:
                    logits_flip = logits_flip[:, keep_mask]
                cons_loss = self.consistency_loss(logits, logits_flip)
                loss = loss + self.lamb_aug_cons * cons_loss

            if self.loss_src_cons:
                src_cons_loss = self.consistency_loss(logits, logits_src)
                loss = loss + self.lamb_src_cons * src_cons_loss

            if self.loss_cmac:
                cmac_loss = self.cmac_loss(image_features, image_features_src, text_features)
                loss = loss + cmac_loss

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
                    if text_features.dim() == 3:
                        T = text_features.mean(dim=0)
                    else:
                        T = text_features
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
            self.adapt_times.append(t2 - t1)
        return loss_report

    def extract_text_embeddings(self, class_names, prompts, average=True):
        text_features = []
        for class_name in class_names:
            texts = [p.format(class_name) for p in prompts]
            texts = self.tokenize(texts).to(self.device)
            class_embeddings = self.model.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            if average:
                class_embeddings_avg = class_embeddings.mean(dim=0)
                class_embeddings_avg = class_embeddings_avg / class_embeddings_avg.norm()
                class_embeddings = torch.cat([class_embeddings, class_embeddings_avg.unsqueeze(0)], dim=0)
            text_features.append(class_embeddings)
        text_features = torch.stack(text_features, dim=1).to(self.device)
        return text_features

    @staticmethod
    def copy_model_and_optimizer(model, optimizer):
        model_state = copy.deepcopy(model.state_dict())
        optimizer_state = copy.deepcopy(optimizer.state_dict())
        return model_state, optimizer_state

    @staticmethod
    def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
        model.load_state_dict(model_state, strict=True)
        optimizer.load_state_dict(optimizer_state)

    @staticmethod
    def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
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

        Enforces prediction invariance between original and augmented/source
        predictions, stabilizing continual TTA by preventing drift.

        Args:
            logits_orig: [#templates, batch, #classes, W, H]
            logits_aug: [#templates, batch, #classes, W, H] (already de-augmented)
        Returns:
            scalar loss
        """
        p = logits_orig.softmax(dim=-3)
        q = logits_aug.softmax(dim=-3)
        log_p = logits_orig.log_softmax(dim=-3)
        log_q = logits_aug.log_softmax(dim=-3)
        kl_pq = (p * (log_p - log_q)).sum(dim=-3)
        kl_qp = (q * (log_q - log_p)).sum(dim=-3)
        if self.cons_type == 'for_kl':
            return kl_pq.mean()
        elif self.cons_type == 'rev_kl':
            return kl_qp.mean()
        else:
            return 0.5 * (kl_pq + kl_qp).mean()

    def cmac_loss(self, image_features_adapt: torch.Tensor,
                   image_features_src: torch.Tensor,
                   text_features: torch.Tensor) -> torch.Tensor:
        """Cross-Modal Anchor Consistency: feature-level text-anchored drift.

        Exploits CLIP's dual-encoder geometry by operating at the feature level.
        For each patch, finds its source-assigned text prototype and penalizes
        drift away from it (loss_away, weighted by lamb_cmac_p1) and drift toward
        unassigned prototypes (loss_toward, weighted by lamb_cmac_p2). Unlike
        logit-level KL, this allows the model to become more aligned with any
        prototype without penalty, enabling adaptation while preventing
        degenerate drift.

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

    def pba_loss(self, image_features: torch.Tensor,
                 text_features: torch.Tensor) -> torch.Tensor:
        """Prototypical Barycenter Alignment loss.

        Geometry-aware replacement for entropy minimization that exploits
        CLIP's dual-encoder hyperspherical geometry. Instead of minimizing
        categorical entropy (which treats all class uncertainty equally),
        PBA computes a probability-weighted barycenter of text prototypes
        and minimizes the angular distance between each patch feature and
        its barycenter.

        The barycenter norm acts as a geometry-aware concentration measure:
        - One-hot prediction: ||b|| = 1, loss = 1 - cos(f, T_c)
        - Spread across similar prototypes: ||b|| ~ 1, low penalty
        - Spread across distant prototypes: ||b|| << 1, high penalty

        Args:
            image_features: [batch, tokens, dim] from adapted model (L2-normalized)
            text_features: [#templates, #classes, dim] or [#classes, dim] (L2-normalized)
        Returns:
            scalar loss
        """
        if text_features.dim() == 3:
            T = text_features.mean(dim=0)
        else:
            T = text_features

        f = image_features[:, 1:]

        logit_scale = self.model.logit_scale.exp()
        sims = logit_scale * (f @ T.t())
        p = sims.softmax(dim=-1)

        barycenter = p @ T
        barycenter_norm = barycenter.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        barycenter_hat = barycenter / barycenter_norm

        loss = 1.0 - (f * barycenter_hat).sum(dim=-1)
        return loss.mean()
