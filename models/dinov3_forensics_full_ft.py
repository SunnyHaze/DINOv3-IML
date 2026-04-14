"""
DINOv3ForensicsFullFT — full fine-tuning of DINOv3 for image manipulation localization.

Paper: "DINOv3 Beats Specialized Detectors: A Simple Foundation Model Baseline
        for Image Forensics"

WARNING: Full fine-tuning on small forensics datasets (e.g. CASIA-v2, ~5K images)
causes the model to collapse into a trivial all-zero prediction. This model is
provided as a comparison baseline to demonstrate why LoRA is preferred.
See paper §4.3 and Figure 3 for the collapse analysis.

Requires: torch
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
class DINOv3ForensicsFullFT(nn.Module):
    """DINOv3 with full backbone fine-tuning.

    Same architecture as DINOv3ForensicsLoRA but all backbone parameters are
    trainable. Included as a baseline; collapses on small datasets.
    """
    def __init__(
        self,
        dinov3_repo_path: str,
        dinov3_weights_path: str,
        dinov3_model_type: str = "dinov3_vitl16",
        image_size: int = 512,
        edge_lambda: float = 20.0,
    ):
        """
        Args:
            dinov3_repo_path: Path to local DINOv3 repo (for torch.hub.load).
            dinov3_weights_path: Path to .pth pretrained weights file.
            dinov3_model_type: One of dinov3_vits16, dinov3_vitb16, dinov3_vitl16.
            image_size: Input image size (must be divisible by 16).
            edge_lambda: Weight for edge-aware loss term.
        """
        super().__init__()
        self.image_size = image_size
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

        # Load DINOv3 backbone — ALL parameters are trainable
        self.backbone = torch.hub.load(
            dinov3_repo_path,
            dinov3_model_type,
            source="local",
            weights=dinov3_weights_path,
        )

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

    def predict(self, image: torch.Tensor) -> torch.Tensor:
        """Inference only — returns sigmoid probability mask (B, 1, H, W)."""
        with torch.no_grad():
            features = self.backbone.get_intermediate_layers(
                image, n=1, reshape=True, norm=True
            )[0]
            logits = self.seg_head(features)
            logits = F.interpolate(
                logits, size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False,
            )
        return torch.sigmoid(logits)

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, **kwargs) -> "DINOv3ForensicsFullFT":
        """Load model from a saved checkpoint.

        Args:
            checkpoint_path: Path to .pth checkpoint file.
            **kwargs: Constructor args (dinov3_repo_path, dinov3_weights_path, etc.)
        Returns:
            Model in eval mode with loaded weights.
        """
        model = cls(**kwargs)
        state = torch.load(checkpoint_path, map_location="cpu")
        sd = state.get("model", state)
        model.load_state_dict(sd)
        model.eval()
        return model

    def forward(self, image, mask, label, edge_mask=None, **kwargs):
        """IMDLBenCo training forward pass. Returns loss dict."""
        features = self.backbone.get_intermediate_layers(
            image, n=1, reshape=True, norm=True
        )[0]

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
