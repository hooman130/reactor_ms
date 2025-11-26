import os
from typing import Optional
import numpy as np
from PIL import Image
import torch

from scripts.reactor_logger import logger

GPEN_REPO_URL = "https://github.com/yangxy/GPEN.git"

class GPENFaceRestoration:
    """GPEN BFR 512 face restorer.

    Loads GPEN FullGenerator architecture from cloned repository and applies
    restoration to an RGB uint8 numpy image.
    """

    def __init__(self, model_path: str, device: str = "CPU"):
        self._name = "GPEN_BFR_512"
        self.model_path = model_path
        self.device = (
            "cuda"
            if device.upper() == "CUDA" and torch.cuda.is_available()
            else "cpu"
        )
        self.model: Optional[torch.nn.Module] = None
        self.repo_root = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..",
            "repositories",
            "gpen",
        )
        self._prepared = False

    def name(self) -> str:  # mimic FaceRestoration interface
        return self._name

    def _clone_repo(self):
        if not os.path.exists(self.repo_root):
            from subprocess import check_call
            os.makedirs(os.path.dirname(self.repo_root), exist_ok=True)
            logger.status("Cloning GPEN repository...")
            check_call([
                "git",
                "clone",
                "--depth",
                "1",
                GPEN_REPO_URL,
                self.repo_root,
            ])

    def _prepare(self):
        if self._prepared:
            return
        self._clone_repo()
        if self.repo_root not in os.sys.path:
            os.sys.path.append(self.repo_root)
        self._prepared = True

    def load(self):
        if self.model is not None:
            return
        self._prepare()
        try:
            from model import FullGenerator as GPENGenerator  # type: ignore
        except Exception as e:
            logger.error(f"Cannot import GPEN generator: {e}")
            raise
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"GPEN model not found at: {self.model_path}"
            )
        logger.status(f"Loading GPEN model from: {self.model_path}")
        # Parameters based on GPEN BFR 512 config
        self.model = GPENGenerator(
            size=512, style_dim=512, n_mlp=8, channel_multiplier=2
        ).to(self.device)
        sd = torch.load(self.model_path, map_location=self.device)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        cleaned = {}
        for k, v in sd.items():
            nk = k.replace("module.", "")
            cleaned[nk] = v
        missing, unexpected = self.model.load_state_dict(cleaned, strict=False)
        if missing:
            logger.debug(f"GPEN missing keys: {missing}")
        if unexpected:
            logger.debug(f"GPEN unexpected keys: {unexpected}")
        self.model.eval()
        logger.status("GPEN model loaded")

    @torch.no_grad()
    def restore(self, image_np: np.ndarray) -> np.ndarray:
        if self.model is None:
            self.load()
        h, w, c = image_np.shape
        if c != 3:
            raise ValueError("GPEN expects 3-channel RGB input")
        img_pil = Image.fromarray(image_np)
        img_resized = img_pil.resize((512, 512), Image.BICUBIC)
        arr = np.array(img_resized).astype(np.float32) / 255.0
        tensor = (
            torch.from_numpy(arr)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )
        tensor = tensor * 2 - 1  # normalize to [-1,1]
        result = self.model(tensor)
        result = (result.clamp(-1, 1) + 1) / 2  # back to [0,1]
        out_np = (
            result.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
        ).astype(np.uint8)
        out_np = np.array(
            Image.fromarray(out_np).resize((w, h), Image.BICUBIC)
        )
        return out_np
