import math
from typing import Callable, Tuple

import torch
import torch.nn.functional as F

from .model import VisionTransformer


class _IdentityMerge:
    def __call__(self, x: torch.Tensor, mode: str = None) -> torch.Tensor:
        return x


def _do_nothing_merge() -> Callable:
    return _IdentityMerge()


def conditional_pooling(
    feat: torch.Tensor,
    threshold: float,
    window_size: Tuple[int, int],
) -> Callable:
    with torch.no_grad():
        ws_h, ws_w = int(window_size[0]), int(window_size[1])
        stride_h, stride_w = ws_h, ws_w
        num_token_window = stride_h * stride_w
        if num_token_window <= 1:
            return _do_nothing_merge()

        feat = feat[:, 1:, :]
        bsz, num_tokens, channels = feat.shape
        base_grid_h = int(math.isqrt(num_tokens))
        base_grid_w = base_grid_h
        if base_grid_h * base_grid_w != num_tokens:
            raise ValueError("ALGM token merge requires a square ViT patch grid.")
        if base_grid_h % ws_h != 0 or base_grid_w % ws_w != 0:
            raise ValueError("ALGM window size must evenly divide the ViT patch grid dimensions.")

        gh = base_grid_h // ws_h
        gw = base_grid_w // ws_w

        feat = feat.view(bsz, base_grid_h, base_grid_w, channels)
        feat = feat.view(bsz, gh, ws_h, gw, ws_w, channels)
        feat = feat.permute(0, 1, 3, 5, 2, 4).contiguous()
        tensor_flattened = feat.view(bsz, gh, gw, channels, num_token_window)

        tensor_1 = tensor_flattened.unsqueeze(-1)
        tensor_2 = tensor_flattened.unsqueeze(-2)
        sims = F.cosine_similarity(tensor_1, tensor_2, dim=3)

        sims_mask = 1 - torch.eye(num_token_window, device=sims.device, dtype=sims.dtype)
        sims = sims * sims_mask.view(1, 1, 1, num_token_window, num_token_window)
        similarity_map = sims.sum(-1).sum(-1) / (num_token_window * (num_token_window - 1))
        similarity_map = similarity_map.view(bsz, -1)

        node_mean = torch.as_tensor(threshold, device=sims.device, dtype=similarity_map.dtype)
        node_mean = node_mean.view(1, 1).expand(bsz, similarity_map.shape[1])
        r = torch.ge(similarity_map, node_mean).sum(dim=1).min().item()
        if r <= 0:
            return _do_nothing_merge()

        _, sim_super_patch_idxs = similarity_map.topk(r, dim=-1)

        token_ids = torch.arange(base_grid_h * base_grid_w, device=feat.device)
        token_ids = token_ids.view(1, base_grid_h, base_grid_w).expand(bsz, -1, -1)
        windowed_tensor = token_ids.view(bsz, gh, ws_h, gw, ws_w)
        windowed_tensor = windowed_tensor.permute(0, 1, 3, 2, 4).contiguous()
        windowed_tensor = windowed_tensor.view(bsz, -1, num_token_window)

        gathered_tensor = torch.gather(
            windowed_tensor,
            1,
            sim_super_patch_idxs.unsqueeze(-1).expand(-1, -1, num_token_window),
        )

        mask = torch.ones((bsz, windowed_tensor.shape[1]), dtype=torch.bool, device=feat.device)
        mask.scatter_(1, sim_super_patch_idxs, False)
        remaining_tensor = windowed_tensor[mask.unsqueeze(-1).expand(-1, -1, num_token_window)]
        remaining_tensor = remaining_tensor.view(bsz, -1, num_token_window)
        unm_idx = remaining_tensor.reshape(bsz, -1).sort(dim=-1).values.unsqueeze(-1)

        dim_index = num_token_window - 1
        src_idx = gathered_tensor[:, :, :dim_index].reshape(bsz, -1).unsqueeze(-1)
        dst_idx = gathered_tensor[:, :, dim_index].reshape(bsz, -1).unsqueeze(-1)
        merge_idx = torch.arange(src_idx.shape[1] // dim_index, device=feat.device)
        merge_idx = merge_idx.repeat_interleave(dim_index).repeat(bsz, 1).unsqueeze(-1)

    def merge(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        x_cls, x_feat = x[:, :1, :], x[:, 1:, :]
        n, t1, c = x_feat.shape
        src = x_feat.gather(dim=-2, index=src_idx.expand(n, r * dim_index, c))
        dst = x_feat.gather(dim=-2, index=dst_idx.expand(n, r, c))
        unm = x_feat.gather(dim=-2, index=unm_idx.expand(n, t1 - (r * num_token_window), c))
        dst = dst.scatter_reduce(-2, merge_idx.expand(n, r * dim_index, c), src, reduce=mode)
        x_out = torch.cat([dst, unm], dim=1)
        return torch.cat((x_cls, x_out), dim=1)

    return merge


def turbo_matching(
    metric: torch.Tensor,
    layer_idx: int,
    source: torch.Tensor,
    class_token: bool = False,
    distill_token: bool = False,
) -> Callable:
    protected = 0
    if class_token:
        protected += 1
    if distill_token:
        protected += 1

    t = metric.shape[1]
    r = (t - protected) // 2
    if r <= 0:
        return _do_nothing_merge()

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True)
        a, b = metric[..., ::2, :], metric[..., 1::2, :]
        scores = a @ b.transpose(-1, -2)

        if class_token:
            scores[..., 0, :] = -math.inf
        if distill_token:
            scores[..., :, 0] = -math.inf

        node_max, node_idx = scores.max(dim=-1)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]

        node_mean = node_max[:, 1:].mean(dim=1).mean() + node_max[:, 1:].std(dim=1).mean() / layer_idx
        node_mean = node_mean.view(1, 1).expand(node_max.shape[0], node_max.shape[1])
        r = torch.ge(node_max, node_mean).sum(dim=1).min().item()
        if r <= 0:
            return _do_nothing_merge()

        unm_idx = edge_idx[..., r:, :]
        src_idx = edge_idx[..., :r, :]
        dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)

        if class_token:
            unm_idx = unm_idx.sort(dim=1)[0]

    def merge(x: torch.Tensor, mode: str = "mean") -> torch.Tensor:
        src, dst = x[..., ::2, :], x[..., 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=-2, index=unm_idx.expand(n, t1 - r, c))
        src_sel = src.gather(dim=-2, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, r, c), src_sel, reduce=mode)

        if distill_token:
            return torch.cat([unm[:, :1], dst[:, :1], unm[:, 1:], dst[:, 1:]], dim=1)
        return torch.cat([unm, dst], dim=1)

    return merge


def merge_wavg(
    merge: Callable,
    x: torch.Tensor,
    size: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if size is None:
        size = torch.ones_like(x[..., 0, None])

    x = merge(x * size, mode="sum")
    size = merge(size, mode="sum")
    x = x / size
    return x, size


def merge_source(
    merge: Callable,
    x: torch.Tensor,
    source: torch.Tensor = None,
) -> torch.Tensor:
    if source is None:
        n, t, _ = x.shape
        source = torch.eye(t, device=x.device, dtype=x.dtype)[None, ...].expand(n, t, t)
    source = merge(source, mode="amax")
    return source


class ALGMVisionTransformer(VisionTransformer):
    def _reset_algm_state(self, original_patch_tokens: int) -> None:
        self._algm_info["size"] = None
        self._algm_info["source"] = None
        self._algm_info["original_patch_tokens"] = int(original_patch_tokens)

    def _algm_has_merged_tokens(self, x: torch.Tensor) -> bool:
        expected_tokens = self._algm_info["original_patch_tokens"] + 1
        return x.shape[0] != expected_tokens

    def _restore_tokens(self, x: torch.Tensor) -> torch.Tensor:
        source = self._algm_info.get("source")
        if source is None:
            return x
        expected_tokens = self._algm_info["original_patch_tokens"] + 1
        if x.shape[0] == expected_tokens:
            return x

        x_nld = x.permute(1, 0, 2).contiguous()
        x_cls = x_nld[:, :1, :]
        x_feat = x_nld[:, 1:, :]
        idxs = source[:, 1:, 1:].argmax(dim=1)
        restored = torch.gather(x_feat, 1, idxs.unsqueeze(-1).expand(-1, -1, x_feat.shape[-1]))
        restored = torch.cat((x_cls, restored), dim=1)
        return restored.permute(1, 0, 2).contiguous()

    def _maybe_merge_tokens(self, x: torch.Tensor, layer_number: int) -> torch.Tensor:
        if layer_number not in self.algm_layers:
            return x

        x_nld = x.permute(1, 0, 2).contiguous()
        source = self._algm_info.get("source")
        if source is None:
            merge = conditional_pooling(x_nld, self.algm_threshold, self.algm_window_size)
        else:
            merge = turbo_matching(
                x_nld,
                layer_number,
                source,
                class_token=True,
                distill_token=False,
            )

        if self._algm_info["trace_source"]:
            self._algm_info["source"] = merge_source(merge, x_nld, self._algm_info["source"])
        x_nld, self._algm_info["size"] = merge_wavg(merge, x_nld, self._algm_info["size"])
        return x_nld.permute(1, 0, 2).contiguous()

    def forward(self, x: torch.Tensor, output_layers=(-1,), out_type="mean", return_vanilla_cls=False, weights=None):
        _, _, width, height = x.shape
        n_patches = (width // self.patch_size, height // self.patch_size)
        original_patch_tokens = n_patches[0] * n_patches[1]
        self._reset_algm_state(original_patch_tokens)

        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([
            self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x,
        ], dim=1)

        if x.shape[1] != self.positional_embedding.shape[0]:
            x = x + self.interpolate_pos_encoding(x, width, height).to(x.dtype)
        else:
            x = x + self.positional_embedding.to(x.dtype)

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)

        num_layers = len(self.transformer.resblocks)
        output_layers = tuple(num_layers + idx if idx < 0 else idx for idx in output_layers)
        last_layer_idx = max(output_layers)
        out_features = []

        for idx in range(last_layer_idx + 1):
            blk = self.transformer.resblocks[idx]
            layer_number = idx + 1

            if idx != last_layer_idx:
                reduced = self.custom_attn("vanilla", blk.attn, blk.ln_1(x), n_patches)
                x = x + reduced
                x = self._maybe_merge_tokens(x, layer_number)
                x = x + blk.mlp(blk.ln_2(x))
                current_feature = reduced if self.arch == "reduced" else x
            else:
                x_for_last = x
                if self.attn_strategy in ("naclip", "nonly") and self._algm_has_merged_tokens(x_for_last):
                    if layer_number in self.algm_layers and self._algm_info.get("source") is not None:
                        raise ValueError(
                            "ALGM token merging with naclip/nonly does not support selecting the final processed layer after earlier merging. "
                            "Choose algm_layers that stop before the final output layer."
                        )
                    x_for_last = self._restore_tokens(x_for_last)

                reduced = self.custom_attn(self.attn_strategy, blk.attn, blk.ln_1(x_for_last), n_patches)
                final_x = x_for_last + reduced
                final_x = self._maybe_merge_tokens(final_x, layer_number)
                final_x = final_x + blk.mlp(blk.ln_2(final_x))
                if self.attn_strategy != "vanilla" and return_vanilla_cls:
                    vanilla_cls = blk(x_for_last)[0]
                current_feature = reduced if self.arch == "reduced" else final_x

            if idx in output_layers:
                out_feature = self._restore_tokens(current_feature)
                out_features.append(out_feature)

        if out_type == "mean":
            if len(out_features) > 1:
                x = torch.mean(torch.stack(out_features), dim=0)
            else:
                x = out_features[0]

            x = x.permute(1, 0, 2)
            if return_vanilla_cls:
                return self.ln_post(x) @ self.proj, self.ln_post(vanilla_cls) @ self.proj
            return self.ln_post(x) @ self.proj

        if out_type == "all":
            out_features = [feat.permute(1, 0, 2) for feat in out_features]
            out_features = [self.ln_post(feat) @ self.proj for feat in out_features]
            out_features = torch.stack(out_features, dim=0)
            if return_vanilla_cls:
                return out_features, self.ln_post(vanilla_cls) @ self.proj
            return out_features

        if out_type == "weighted_mean":
            if len(weights) != len(out_features):
                raise ValueError("weights length must match number of output features")
            x = torch.stack(out_features, dim=0)
            x = torch.sum(x * weights.unsqueeze(1).unsqueeze(-1), dim=0)
            x = x.permute(1, 0, 2)
            return self.ln_post(x) @ self.proj

        raise ValueError(f"Unsupported out_type: {out_type}")


def apply_patch(
    visual: VisionTransformer,
    selected_layers,
    threshold: float = 0.8,
    window_size: Tuple[int, int] = (2, 2),
    merge_type: str = "algm",
    trace_source: bool = True,
):
    if merge_type != "algm":
        raise ValueError(f"Unsupported merge_type: {merge_type}")
    if not isinstance(visual, VisionTransformer):
        raise ValueError("ALGM token merging is only supported for CLIP ViT image encoders.")

    num_layers = len(visual.transformer.resblocks)
    layers = tuple(sorted({int(layer) for layer in selected_layers}))
    if len(layers) == 0:
        raise ValueError("algm_layers must contain at least one layer index.")
    if min(layers) < 1 or max(layers) > num_layers:
        raise ValueError(f"algm_layers must be within [1, {num_layers}] for this ViT backbone.")

    window_size = (int(window_size[0]), int(window_size[1]))
    if window_size[0] <= 0 or window_size[1] <= 0:
        raise ValueError("algm_window_size values must be positive integers.")

    visual.__class__ = ALGMVisionTransformer
    visual.algm_layers = layers
    visual.algm_threshold = float(threshold)
    visual.algm_window_size = window_size
    visual._algm_info = {
        "size": None,
        "source": None,
        "trace_source": trace_source,
        "original_patch_tokens": None,
    }
    return visual
