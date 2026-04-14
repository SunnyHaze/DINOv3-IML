"""
DINOv3Forensics — frozen DINOv3 backbone with lightweight segmentation head.

Paper: "DINOv3 Beats Specialized Detectors: A Simple Foundation Model Baseline
        for Image Forensics"

This is the simplest baseline: the backbone is completely frozen; only the
3-conv segmentation head is trained.

Requires: torch, pillow, numpy
Optional:  imdlbenco (for training only — install with: pip install imdlbenco)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from IMDLBenCo.registry import MODELS
    def _register(cls): return MODELS.register_module()(cls)
except ImportError:
    def _register(cls): return cls


@_register
class DINOv3Forensics(nn.Module):
    def __init__(
        self,
        dinov3_repo_path: str,
        dinov3_weights_path: str,
        dinov3_model_type: str = "dinov3_vitb16",
        image_size: int = 512,
        freeze_backbone: bool = True,
        edge_lambda: float = 20.0,
    ):
        """DINOv3 frozen baseline for image manipulation localization.

        Args:
            dinov3_repo_path: Path to local DINOv3 repo (for torch.hub.load).
            dinov3_weights_path: Path to .pth pretrained weights file.
            dinov3_model_type: One of dinov3_vits16, dinov3_vitb16, dinov3_vitl16.
            image_size: Input image size (must be divisible by 16).
            freeze_backbone: Whether to freeze the DINOv3 backbone (default True).
            edge_lambda: Weight for edge-aware loss term.
        """
        super().__init__()
        self.image_size = image_size
        self.freeze_backbone = freeze_backbone
        self.edge_lambda = edge_lambda

        feat_dim_map = {
            "dinov3_vits16": 384,
            "dinov3_vitb16": 768,
            "dinov3_vitl16": 1024,
        }
        self.feat_dim = feat_dim_map.get(dinov3_model_type)
        if self.feat_dim is None:
            raise ValueError(
                f"Unsupported dinov3_model_type: {dinov3_model_type}. "
                f"Choose from {list(feat_dim_map.keys())}"
            )

        # Load DINOv3 backbone via torch.hub (local source)
        self.backbone = torch.hub.load(
            dinov3_repo_path,
            dinov3_model_type,
            source="local",
            weights=dinov3_weights_path,
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        # Lightweight segmentation head: feat_dim → feat_dim/2 → feat_dim/4 → 1
        self.seg_head = nn.Sequential(
            nn.Conv2d(self.feat_dim, self.feat_dim // 2, 3, padding=1),
            nn.BatchNorm2d(self.feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat_dim // 2, self.feat_dim // 4, 3, padding=1),
            nn.BatchNorm2d(self.feat_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat_dim // 4, 1, 1),
        )

        self.BCE_loss = nn.BCEWithLogitsLoss()
        self._init_seg_head()

    def _init_seg_head(self):
        for m in self.seg_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward_features(self, image: torch.Tensor) -> torch.Tensor:
        """Extract spatial feature map. Returns (B, feat_dim, H/16, W/16)."""
        return self.backbone.get_intermediate_layers(
            image, n=1, reshape=True, norm=True
        )[0]

    def predict(self, image: torch.Tensor) -> torch.Tensor:
        """Inference only — returns sigmoid probability mask (B, 1, H, W)."""
        with torch.no_grad():
            features = self.forward_features(image)
            logits = self.seg_head(features)
            logits = F.interpolate(
                logits, size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False,
            )
        return torch.sigmoid(logits)

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, **kwargs) -> "DINOv3Forensics":
        """Load model from a saved checkpoint.

        Args:
            checkpoint_path: Path to .pth checkpoint file.
            **kwargs: Constructor arguments (dinov3_repo_path, dinov3_weights_path, etc.)
        Returns:
            Model in eval mode with loaded weights.
        """
        model = cls(**kwargs)
        state = torch.load(checkpoint_path, map_location="cpu")
        # Handle both raw state_dict and IMDLBenCo checkpoint dicts
        sd = state.get("model", state)
        model.load_state_dict(sd)
        model.eval()
        return model

    def forward(self, image, mask, label, edge_mask=None, **kwargs):
        """IMDLBenCo training forward pass. Returns loss dict."""
        features = self.forward_features(image)

        logits = self.seg_head(features)
        logits = F.interpolate(
            logits, size=(self.image_size, self.image_size),
            mode="bilinear", align_corners=False,
        )

        predict_loss = self.BCE_loss(logits, mask)
        combined_loss = predict_loss
        edge_loss = torch.tensor(0.0, device=image.device)

        if edge_mask is not None:
            edge_loss = F.binary_cross_entropy_with_logits(
                input=logits, target=mask, weight=edge_mask,
            ) * self.edge_lambda
            combined_loss = predict_loss + edge_loss

        pred_mask = torch.sigmoid(logits)

        return {
            "backward_loss": combined_loss,
            "pred_mask": pred_mask,
            "pred_label": None,
            "visual_loss": {
                "predict_loss": predict_loss,
                "edge_loss": edge_loss,
                "combined_loss": combined_loss,
            },
            "visual_image": {"pred_mask": pred_mask},
        }
