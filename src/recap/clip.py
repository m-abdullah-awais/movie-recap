"""CLIP ViT-B/32 image and text embedding through onnxruntime.

Vision is used only for retrieval: finding which shot matches a line of
narration. It never reads the story.

PyTorch is deliberately absent. The int8 ONNX export is roughly 150 MB against
two gigabytes, and the pipeline needs nothing else PyTorch provides.

Tensor names differ between ONNX exports of the same model, so inputs and outputs
are discovered from the loaded graph rather than hardcoded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .models import ClipAssets

# CLIP's own normalisation constants. Getting these wrong does not error, it just
# quietly degrades every similarity score, so they are stated explicitly.
_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

IMAGE_SIZE = 224
CONTEXT_LENGTH = 77
EMBED_DIM = 512


class ClipUnavailable(RuntimeError):
    """onnxruntime, the tokenizer library, or the model files are missing."""


def _pick_output(session, preferred: tuple[str, ...]) -> str:
    """Choose the embedding output from a session.

    Exports vary: some name it ``image_embeds``, some ``text_embeds``, some only
    expose pooled or last hidden state. The preferred names are tried first, then
    any output whose last dimension matches the embedding width, then the first
    output as a last resort.
    """
    outputs = session.get_outputs()
    by_name = {o.name: o for o in outputs}
    for name in preferred:
        if name in by_name:
            return name
    for output in outputs:
        shape = output.shape or []
        if shape and isinstance(shape[-1], int) and shape[-1] == EMBED_DIM:
            return output.name
    return outputs[0].name


class Clip:
    """Loaded CLIP encoders. Construct once and reuse, loading is not cheap."""

    def __init__(self, assets: ClipAssets, threads: int = 0):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ClipUnavailable("onnxruntime is not installed") from exc
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise ClipUnavailable("the tokenizers package is not installed") from exc

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads > 0:
            options.intra_op_num_threads = threads

        try:
            self._vision = ort.InferenceSession(
                str(assets.vision), options, providers=["CPUExecutionProvider"]
            )
            self._text = ort.InferenceSession(
                str(assets.text), options, providers=["CPUExecutionProvider"]
            )
        except Exception as exc:  # noqa: BLE001 - onnxruntime raises bare Exception
            raise ClipUnavailable(f"a CLIP model file could not be loaded: {exc}") from exc

        self._tokenizer = Tokenizer.from_file(str(assets.tokenizer))
        self._tokenizer.enable_truncation(max_length=CONTEXT_LENGTH)
        self._tokenizer.enable_padding(length=CONTEXT_LENGTH)

        self._vision_input = self._vision.get_inputs()[0].name
        self._vision_output = _pick_output(self._vision, ("image_embeds", "pooler_output"))
        self._text_inputs = [i.name for i in self._text.get_inputs()]
        self._text_output = _pick_output(self._text, ("text_embeds", "pooler_output"))

    # ---- images -----------------------------------------------------------

    def embed_images(self, batch: np.ndarray) -> np.ndarray:
        """Embed a batch of preprocessed images, shape (n, 3, 224, 224)."""
        result = self._vision.run([self._vision_output], {self._vision_input: batch})[0]
        return _l2_normalise(np.asarray(result, dtype=np.float32))

    # ---- text -------------------------------------------------------------

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        encoded = self._tokenizer.encode_batch([t or " " for t in texts])
        ids = np.array([e.ids for e in encoded], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)

        feed: dict[str, np.ndarray] = {}
        for name in self._text_inputs:
            if "mask" in name:
                feed[name] = mask
            else:
                feed[name] = ids
        result = self._text.run([self._text_output], feed)[0]
        return _l2_normalise(np.asarray(result, dtype=np.float32))


def preprocess(image_bgr: np.ndarray) -> np.ndarray:
    """Turn one decoded frame into a CLIP input tensor.

    Module level rather than a method, because callers preprocess frames while
    batching them up and should not need a loaded model to do it.

    Frames arrive from OpenCV in BGR with the channel axis last. CLIP expects
    RGB, channels first, scaled to 0 to 1, then normalised with its own
    constants. Getting the channel order or the constants wrong does not raise,
    it just quietly degrades every similarity score, which is why this is
    verified against known-brightness frames rather than assumed.
    """
    rgb = image_bgr[:, :, ::-1].astype(np.float32) / 255.0
    normalised = (rgb - _MEAN) / _STD
    return np.transpose(normalised, (2, 0, 1))


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """Unit-length rows, so a dot product is cosine similarity.

    Normalising once here means the selection stage can score with a single
    matrix multiply instead of dividing by norms for every comparison.
    """
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def load(assets: ClipAssets | None, threads: int = 0) -> Clip | None:
    """Load CLIP, or return None so the caller can fall back to time proximity."""
    if assets is None:
        return None
    try:
        return Clip(assets, threads=threads)
    except ClipUnavailable:
        return None


def read_frame(path: Path) -> np.ndarray | None:
    """Decode one keyframe image, already cropped to the model's input size."""
    try:
        import cv2
    except ImportError:
        return None
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    if image.shape[0] != IMAGE_SIZE or image.shape[1] != IMAGE_SIZE:
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return image
