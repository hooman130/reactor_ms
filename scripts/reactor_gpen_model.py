import os
from typing import Dict, List

import cv2
import numpy as np
import onnxruntime as ort

from modules import devices, errors, face_restoration, face_restoration_utils, modelloader, shared
from scripts.reactor_globals import MODELS_PATH
from scripts.reactor_logger import logger

GPEN_NAME = "GPEN-BFR-512"
GPEN_MODEL = f"{GPEN_NAME}.onnx"
GPEN_URL = "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GPEN-BFR-512.onnx"
MODEL_DIR = os.path.join(MODELS_PATH, "facerestore_models")


def _prepare_cropped_face(cropped_face: np.ndarray) -> np.ndarray:
    """
    Prepare the cropped face for GPEN ONNX inference.
    Converts BGR uint8 image to normalized float32 NCHW tensor.
    """
    cropped_face = cropped_face[:, :, ::-1] / 255.0
    cropped_face = (cropped_face - 0.5) / 0.5
    cropped_face = np.expand_dims(cropped_face.transpose(2, 0, 1), axis=0).astype(np.float32)
    return cropped_face


def _normalize_cropped_face(cropped_face: np.ndarray) -> np.ndarray:
    """
    Convert GPEN output back to BGR uint8 image.
    """
    cropped_face = np.clip(cropped_face, -1, 1)
    cropped_face = (cropped_face + 1) / 2
    cropped_face = cropped_face.transpose(1, 2, 0)
    cropped_face = (cropped_face * 255.0).round()
    cropped_face = cropped_face.astype(np.uint8)[:, :, ::-1]
    return cropped_face


class FaceRestorerGPEN(face_restoration.FaceRestoration):
    def __init__(self, model_dir: str = MODEL_DIR):
        super().__init__()
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.session: ort.InferenceSession | None = None
        self.providers: List[str] = []
        self.face_helper = None

    def name(self):
        return GPEN_NAME

    def _select_providers(self) -> List[str]:
        available = ort.get_available_providers()
        providers: List[str] = []
        try:
            if devices.device_codeformer.type == "cuda" and "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
        except Exception:
            # Fallback to CPU if device detection fails
            pass
        if "CPUExecutionProvider" in available:
            providers.append("CPUExecutionProvider")
        if len(providers) == 0:
            providers = available
        return providers

    def _load_session(self) -> ort.InferenceSession | None:
        try:
            face_restoration_utils.patch_facexlib(self.model_dir)
        except Exception:
            errors.report("Error patching facexlib for GPEN", exc_info=True)

        self.providers = self._select_providers()
        for model_path in modelloader.load_models(
            model_path=self.model_dir,
            model_url=GPEN_URL,
            command_path=self.model_dir,
            download_name=GPEN_MODEL,
            ext_filter=[".onnx"],
        ):
            try:
                logger.info("Loading GPEN model from %s with providers %s", model_path, self.providers)
                return ort.InferenceSession(model_path, providers=self.providers)
            except Exception:
                errors.report("Error setting up GPEN model", exc_info=True)
        return None

    def _ensure_resources(self) -> bool:
        if self.session is None:
            self.session = self._load_session()
        if self.session is None:
            return False
        if self.face_helper is None:
            try:
                self.face_helper = face_restoration_utils.create_face_helper(devices.device_codeformer)
            except Exception:
                errors.report("Error creating face helper for GPEN", exc_info=True)
                return False
        return True

    def _run_session(self, cropped_face: np.ndarray) -> np.ndarray:
        inputs: Dict[str, np.ndarray] = {}
        for ort_input in self.session.get_inputs():
            if ort_input.name == "input":
                inputs[ort_input.name] = _prepare_cropped_face(cropped_face)
            elif ort_input.name == "weight":
                inputs[ort_input.name] = np.array([1], dtype=np.double)
        if not inputs:
            inputs["input"] = _prepare_cropped_face(cropped_face)

        output = self.session.run(None, inputs)[0][0]
        return _normalize_cropped_face(output)

    def restore(self, np_image):
        if not self._ensure_resources():
            logger.error("GPEN resources are not ready; skipping face restoration")
            return np_image

        np_image_bgr = np_image[:, :, ::-1]
        original_resolution = np_image_bgr.shape[0:2]

        try:
            self.face_helper.clean_all()
            self.face_helper.read_image(np_image_bgr)
            self.face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
            self.face_helper.align_warp_face()

            if len(self.face_helper.cropped_faces) == 0:
                return np_image

            for cropped_face in self.face_helper.cropped_faces:
                try:
                    restored_face = self._run_session(cropped_face)
                except Exception:
                    errors.report("Failed GPEN face-restoration inference", exc_info=True)
                    restored_face = cropped_face

                self.face_helper.add_restored_face(restored_face)

            self.face_helper.get_inverse_affine(None)
            restored_img = self.face_helper.paste_faces_to_input_image()
            restored_img = restored_img[:, :, ::-1]

            if original_resolution != restored_img.shape[0:2]:
                restored_img = cv2.resize(
                    restored_img,
                    (original_resolution[1], original_resolution[0]),
                    interpolation=cv2.INTER_LINEAR,
                )

            return restored_img
        finally:
            self.face_helper.clean_all()


def setup_gpen_face_restorer():
    try:
        if len(shared.face_restorers) == 0:
            shared.face_restorers.append(face_restoration.FaceRestoration())
        if any(restorer.name() == GPEN_NAME for restorer in shared.face_restorers):
            return
        restorer = FaceRestorerGPEN()
        shared.face_restorers.append(restorer)
        logger.info("GPEN face restorer registered")
    except Exception:
        errors.report("Error initializing GPEN face restorer", exc_info=True)
