"""CLIP model wrapper for the zero-shot / CHiLS / description-based baselines."""
import torch
from transformers import CLIPModel, CLIPProcessor


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class CLIPWrapper:
    """Thin wrapper around a HuggingFace CLIP model for text/image embedding."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = None):
        self.device = device or default_device()
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        """Return L2-normalized text embeddings, shape (len(texts), dim)."""
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        feats = self.model.get_text_features(**inputs)
        return feats / feats.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def encode_images(self, images: list) -> torch.Tensor:
        """Return L2-normalized image embeddings, shape (len(images), dim). `images` are PIL Images."""
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        feats = self.model.get_image_features(**inputs)
        return feats / feats.norm(dim=-1, keepdim=True)
