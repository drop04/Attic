"""Local OCR: OpenCV preprocessing + Tesseract. Both are optional imports."""
from __future__ import annotations

from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


class OCRUnavailable(RuntimeError):
    pass


def _preprocess(path: Path):
    import cv2
    import numpy as np

    img = cv2.imread(str(path))
    if img is None:
        raise ValueError("could not read image")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if max(h, w) < 1600:  # phone photos are fine; tiny screenshots need upscaling
        scale = 1600 / max(h, w)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
    # deskew using the dark-pixel bounding box
    coords = np.column_stack(np.where(bw < 128))
    if len(coords) > 500:
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        if 0.5 < abs(angle) < 15:
            hh, ww = bw.shape
            m = cv2.getRotationMatrix2D((ww / 2, hh / 2), angle, 1.0)
            bw = cv2.warpAffine(bw, m, (ww, hh), flags=cv2.INTER_CUBIC, borderValue=255)
    return bw


def ocr_image(path: str | Path, lang: str = "eng") -> str:
    try:
        import pytesseract
        from PIL import Image
        import cv2  # noqa: F401
    except ImportError as exc:
        raise OCRUnavailable(
            "OCR needs: pip install pytesseract opencv-python-headless pillow, "
            "and the tesseract binary (apt install tesseract-ocr)"
        ) from exc
    try:
        pre = _preprocess(Path(path))
        return pytesseract.image_to_string(Image.fromarray(pre), lang=lang, config="--psm 6").strip()
    except pytesseract.TesseractNotFoundError as exc:
        raise OCRUnavailable("tesseract binary not found (apt install tesseract-ocr)") from exc
