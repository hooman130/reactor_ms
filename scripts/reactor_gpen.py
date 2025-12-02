import os
import cv2
import numpy as np
import onnxruntime
import urllib.request
from tqdm import tqdm

# Common paths
try:
    from modules.paths_internal import models_path
except:
    try:
        from modules.paths import models_path
    except:
        models_path = os.path.abspath("models")

GPEN_MODELS_PATH = os.path.join(models_path, "facerestore_models")

GPEN_MODELS = {
    "GPEN-BFR-512": "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GPEN-BFR-512.onnx",
    "GPEN-BFR-1024": "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GPEN-BFR-1024.onnx",
    "GPEN-BFR-2048": "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GPEN-BFR-2048.onnx",
}

GPEN_SESSIONS = {}

class GPENRestorer:
    def __init__(self, model_name):
        self.model_name = model_name

    def name(self):
        return self.model_name

def download_gpen_model(model_name):
    if model_name not in GPEN_MODELS:
        print(f"Unknown GPEN model: {model_name}")
        return None

    if not os.path.exists(GPEN_MODELS_PATH):
        os.makedirs(GPEN_MODELS_PATH)

    file_path = os.path.join(GPEN_MODELS_PATH, f"{model_name}.onnx")

    if not os.path.exists(file_path):
        print(f"Downloading {model_name} to {file_path}...")
        url = GPEN_MODELS[model_name]
        try:
            with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=model_name) as t:
                def reporthook(blocknum, blocksize, totalsize):
                    t.total = totalsize
                    t.update(blocksize)
                urllib.request.urlretrieve(url, file_path, reporthook=reporthook)
            print(f"Downloaded {model_name}")
        except Exception as e:
            print(f"Failed to download {model_name}: {e}")
            if os.path.exists(file_path):
                os.remove(file_path)
            return None

    return file_path

def get_gpen_session(model_path):
    if model_path in GPEN_SESSIONS:
        return GPEN_SESSIONS[model_path]

    try:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = onnxruntime.InferenceSession(model_path, providers=providers)
        GPEN_SESSIONS[model_path] = session
        return session
    except Exception as e:
        print(f"Failed to create ONNX session for {model_path}: {e}")
        return None

def process_gpen_face(cropped_face, model_name):
    """
    Runs GPEN inference on a cropped face.
    cropped_face: HWC RGB numpy array (uint8)
    model_name: string
    Returns: HWC RGB numpy array (uint8) of restored face
    """
    model_path = download_gpen_model(model_name)
    if not model_path:
        return cropped_face

    if "512" in model_name:
        size = 512
    elif "1024" in model_name:
        size = 1024
    elif "2048" in model_name:
        size = 2048
    else:
        size = 512

    session = get_gpen_session(model_path)
    if not session:
        return cropped_face

    # Resize to model input
    # Use INTER_AREA for downscaling (if input is larger) or CUBIC for upscaling
    # Usually cropped face from detection might be small.
    img_resized = cv2.resize(cropped_face, (size, size), interpolation=cv2.INTER_CUBIC)

    # Normalize: (x/255.0 - 0.5) / 0.5
    img_float = img_resized.astype(np.float32) / 255.0
    img_norm = (img_float - 0.5) / 0.5

    # HWC to CHW
    img_chw = img_norm.transpose(2, 0, 1)
    img_batch = np.expand_dims(img_chw, axis=0)

    # Run
    input_name = session.get_inputs()[0].name
    try:
        output = session.run(None, {input_name: img_batch})[0]
    except Exception as e:
        print(f"GPEN Run Error: {e}")
        return cropped_face

    # Post process: CHW -> HWC, Denormalize
    output = output[0].transpose(1, 2, 0)
    output = (output * 0.5) + 0.5
    output = np.clip(output, 0, 1)
    output = (output * 255.0).astype(np.uint8)

    return output
