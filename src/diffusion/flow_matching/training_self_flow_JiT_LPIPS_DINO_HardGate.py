import torch
import lpips

from typing import Callable, Sequence

from src.utils.no_grad import freeze_model
from src.diffusion.base.training import *
from src.diffusion.base.scheduling import BaseScheduler


def inverse_sigma(alpha, sigma):
    return 1 / sigma**2


def snr(alpha, sigma):
    return alpha / sigma


def minsnr(alpha, sigma, threshold=5):
    return torch.clip(alpha / sigma, min=threshold)


def maxsnr(alpha, sigma, threshold=5):
    return torch.clip(alpha / sigma, max=threshold)


def constant(alpha, sigma):
    return 1


def time_shift_fn(t, timeshift=1.0):
    return t / (t + (1 - t) * timeshift)


class SelfFlowTrainer(BaseTrainer):
    def __init__(
        self,
        scheduler: BaseScheduler,
        encoder: nn.Module = None,
        loss_weight_fn: Callable = constant,
        lognorm_t: bool = False,
        timeshift: float = 1.0,
        P_mean: float = -0.8,
        P_std: float = 0.8,
        t_eps: float = 0.05,
        student_layer: int = 6,
        teacher_layer: int = 12,
        teacher_t_gap: float = 0.10,
        min_student_t: float = 0.02,
        mask_ratio: float = 0.25,
        self_flow_weight: float = 0.5,
        proj_dim_in: int = 256,
        proj_hidden_dim: int = 256,
        proj_dim_out: int = 256,
        dino_layers: Sequence[int] = (11,),
        lpips_weight: float = 1.0,
        dino_weight: float = 1.0,
        percept_t_threshold: float = 0.3,
        percept_ratio: float = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if teacher_layer <= student_layer:
            raise ValueError("teacher_layer should be deeper than student_layer")

        self.scheduler = scheduler
        self.encoder = encoder
        self.loss_weight_fn = loss_weight_fn
        self.lognorm_t = lognorm_t
        self.timeshift = timeshift
        self.P_mean = P_mean
        self.P_std = P_std
        self.t_eps = t_eps

        self.student_layer = student_layer
        self.teacher_layer = teacher_layer
        self.teacher_t_gap = teacher_t_gap
        self.min_student_t = min_student_t
        self.mask_ratio = mask_ratio
        self.self_flow_weight = self_flow_weight
        self.lpips_weight = lpips_weight
        self.dino_weight = dino_weight
        self.percept_t_threshold = percept_t_threshold
        self.percept_ratio = percept_ratio
        self.dino_layers = list(dino_layers)
        self.cached_percept_weight = 1.0

        freeze_model(self.encoder)

        self.lpips_loss_fn = lpips.LPIPS(net="vgg").eval()
        self.lpips_loss_fn.compile()
        freeze_model(self.lpips_loss_fn)

        self.projector = nn.Sequential(
            nn.Linear(proj_dim_in, proj_hidden_dim),
            nn.SiLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.SiLU(),
            nn.Linear(proj_hidden_dim, proj_dim_out),
        )

    def _sample_teacher_base_t(self, batch_size, device):
        if self.lognorm_t:
            base_t = (
                torch.randn(batch_size, device=device, dtype=torch.float32) * self.P_std + self.P_mean
            ).sigmoid()
        else:
            base_t = torch.rand(batch_size, device=device, dtype=torch.float32)
        return time_shift_fn(base_t, self.timeshift)

    def _sample_token_mask(self, batch_size, num_tokens, device):
        token_mask = torch.rand(batch_size, num_tokens, device=device) < self.mask_ratio

        empty = token_mask.sum(dim=1) == 0
        if empty.any():
            empty_indices = empty.nonzero(as_tuple=False).flatten()
            rand_indices = torch.randint(0, num_tokens, (empty_indices.numel(),), device=device)
            token_mask[empty_indices, rand_indices] = True

        full = token_mask.sum(dim=1) == num_tokens
        if full.any() and num_tokens > 1:
            full_indices = full.nonzero(as_tuple=False).flatten()
            rand_indices = torch.randint(0, num_tokens, (full_indices.numel(),), device=device)
            token_mask[full_indices, rand_indices] = False

        return token_mask

    def _build_dual_timesteps(self, teacher_base_t, token_mask):
        batch_size, num_tokens = token_mask.shape
        t_teacher_tokens = teacher_base_t[:, None].expand(batch_size, num_tokens).clone()
        t_student_tokens = t_teacher_tokens.clone()
        t_student_tokens[token_mask] = (
            t_teacher_tokens[token_mask] - self.teacher_t_gap
        ).clamp(min=self.min_student_t, max=1.0)
        return t_teacher_tokens, t_student_tokens

    def _tokens_to_spatial_map(self, t_tokens, height, width, patch_size):
        batch_size, num_tokens = t_tokens.shape
        grid_h = height // patch_size
        grid_w = width // patch_size
        if num_tokens != grid_h * grid_w:
            raise ValueError(
                f"Token count {num_tokens} does not match spatial grid {grid_h}x{grid_w}"
            )
        t_grid = t_tokens.view(batch_size, 1, grid_h, grid_w)
        return t_grid.repeat_interleave(patch_size, dim=2).repeat_interleave(patch_size, dim=3)

    def _build_percept_gate(self, t_student_tokens):
        t_min_student = t_student_tokens.min(dim=1).values
        percept_gate = (t_min_student >= self.percept_t_threshold).to(t_student_tokens.dtype)
        return percept_gate, t_min_student

    def _masked_cosine_loss(self, student_proj, teacher_feat, token_mask):
        student_proj = torch.nn.functional.normalize(student_proj.float(), dim=-1)
        teacher_feat = torch.nn.functional.normalize(teacher_feat.float(), dim=-1)
        if student_proj.shape[-1] != teacher_feat.shape[-1]:
            raise ValueError(
                f"Projected student dim {student_proj.shape[-1]} does not match teacher dim {teacher_feat.shape[-1]}"
            )
        cos = torch.nn.functional.cosine_similarity(student_proj, teacher_feat, dim=-1)
        mask = token_mask.float()
        return ((1.0 - cos) * mask).sum() / mask.sum().clamp_min(1.0)

    def _calculate_adaptive_weight(self, rec_loss, g_loss, last_layer):
        rec_grads = torch.autograd.grad(rec_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(rec_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 100.0).detach()
        return d_weight

    def compute_lpips_loss(self, pred_img, x, percept_gate=None):
        batch_size, _, height, width = pred_img.shape
        lpips_vals = self.lpips_loss_fn(pred_img, x).view(batch_size, -1).mean(dim=1)
        if percept_gate is None:
            return lpips_vals.mean()

        gate = percept_gate.view(batch_size).to(lpips_vals.dtype)
        gate_sum = gate.sum()
        if gate_sum.item() == 0:
            return lpips_vals.new_zeros(())
        return (lpips_vals * gate).sum() / gate_sum

    def compute_dino_loss(self, pred_dino_feats, gt_dino_feats, percept_gate=None):
        cos_losses = {}
        final_cos_loss = 0.0
        batch_size = pred_dino_feats[0].shape[0]

        if percept_gate is not None:
            gate = percept_gate.view(batch_size, 1).to(pred_dino_feats[0].dtype)
            gate_sum = gate.sum()
        else:
            gate = None
            gate_sum = None

        for i, (pred_feat, gt_feat) in enumerate(zip(pred_dino_feats, gt_dino_feats)):
            cos_map = torch.nn.functional.cosine_similarity(pred_feat.float(), gt_feat.float(), dim=-1)
            if gate is None:
                cos_loss = 1.0 - cos_map.mean()
            elif gate_sum.item() == 0:
                cos_loss = cos_map.new_zeros(())
            else:
                cos_loss = 1.0 - (cos_map * gate).sum() / (gate_sum * cos_map.shape[1])
            cos_losses[f"inter_cos_{i}"] = cos_loss
            final_cos_loss += cos_loss

        cos_losses["dino_percept_loss"] = final_cos_loss / len(pred_dino_feats)
        return cos_losses

    def _impl_trainstep(self, net, ema_net, solver, x, y, metadata=None):
        raw_images = metadata["raw_image"]
        current_step = metadata.get("global_step", 0)
        batch_size, c, height, width = x.shape
        patch_size = net.patch_size
        num_tokens = (height // patch_size) * (width // patch_size)

        self.lpips_loss_fn.eval()

        teacher_base_t = self._sample_teacher_base_t(batch_size, x.device)
        token_mask = self._sample_token_mask(batch_size, num_tokens, x.device)
        t_teacher_tokens, t_student_tokens = self._build_dual_timesteps(teacher_base_t, token_mask)
        percept_gate, t_min_student = self._build_percept_gate(t_student_tokens)

        t_teacher_map = self._tokens_to_spatial_map(t_teacher_tokens, height, width, patch_size)
        t_student_map = self._tokens_to_spatial_map(t_student_tokens, height, width, patch_size)

        eps = torch.randn_like(x)

        alpha_teacher = self.scheduler.alpha_value(t_teacher_map)
        sigma_teacher = self.scheduler.sigma_value(t_teacher_map)
        alpha_student = self.scheduler.alpha_value(t_student_map)
        sigma_student = self.scheduler.sigma_value(t_student_map)

        x_teacher = alpha_teacher * x + sigma_teacher * eps
        x_student = alpha_student * x + sigma_student * eps

        text_t = teacher_base_t
        pred_img, student_feat = net(
            x_student,
            t_student_tokens,
            y,
            text_t=text_t,
            return_after_layer=self.student_layer,
        )

        pred_v = (pred_img - x_student) / (1 - t_student_map).clamp_min(self.t_eps)
        target_v = (x - x_student) / (1 - t_student_map).clamp_min(self.t_eps)
        weight = self.loss_weight_fn(alpha_student, sigma_student)
        fm_loss = weight * (pred_v - target_v) ** 2

        with torch.no_grad():
            _, teacher_feat = ema_net(
                x_teacher,
                t_teacher_tokens,
                y,
                text_t=text_t,
                return_after_layer=self.teacher_layer,
            )

        student_proj = self.projector(student_feat)
        self_flow_loss = self._masked_cosine_loss(student_proj, teacher_feat.detach(), token_mask)

        with torch.no_grad():
            dst_features = self.encoder.get_intermediate_feats(raw_images, n=self.dino_layers)

        lpips_loss = self.compute_lpips_loss(pred_img, x, percept_gate)
        raw_pred_img = (pred_img + 1) / 2
        pred_feats = self.encoder.get_intermediate_feats(raw_pred_img, n=self.dino_layers)
        dino_losses = self.compute_dino_loss(pred_feats, dst_features, percept_gate)

        rec_loss = fm_loss.mean()
        percept_loss = self.lpips_weight * lpips_loss + self.dino_weight * dino_losses["dino_percept_loss"]

        if current_step >= 400000 and current_step % 50 == 0:
            last_layer = net.final_layer.linear.weight
            percept_weight = self._calculate_adaptive_weight(rec_loss, percept_loss, last_layer)
            self.cached_percept_weight = 0.8 * self.cached_percept_weight + 0.2 * percept_weight

        total_loss = (
            rec_loss
            + self.self_flow_weight * self_flow_loss
            + self.percept_ratio * self.cached_percept_weight * percept_loss
        )

        out = dict(
            loss=total_loss,
            fm_loss=rec_loss,
            self_flow_loss=self_flow_loss,
            lpips_loss=lpips_loss,
            dino_percept_loss=dino_losses["dino_percept_loss"],
            percept_weight=self.cached_percept_weight,
            sf_t_teacher_mean=t_teacher_tokens.mean(),
            sf_t_student_mean=t_student_tokens.mean(),
            sf_t_gap_mean=(t_teacher_tokens - t_student_tokens).mean(),
            sf_t_min_student=t_min_student.mean(),
            sf_percept_gate_on_rate=percept_gate.mean(),
            sf_mask_ratio=token_mask.float().mean(),
        )
        out.update(dino_losses)
        return out

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        if destination is None:
            destination = {}
        self.projector.state_dict(
            destination=destination,
            prefix=prefix + "projector.",
            keep_vars=keep_vars,
        )
        return destination
