import glob
import shutil

import pytest
from PIL import Image, ImageDraw, ImageFont

from attic import extract, ocr
from tests.samples import RECEIPT

pytestmark = pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")


def _render(text, path, rotate=0.0):
    fonts = glob.glob("/usr/share/fonts/**/DejaVuSans*.ttf", recursive=True)
    font = ImageFont.truetype(fonts[0], 30) if fonts else ImageFont.load_default()
    lines = text.splitlines()
    img = Image.new("RGB", (1100, 60 + 44 * len(lines)), "white")
    d = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        d.text((40, 30 + 44 * i), line, fill="black", font=font)
    if rotate:
        img = img.rotate(rotate, expand=True, fillcolor="white")
    img.save(path)


@pytest.mark.parametrize("rotate", [0.0, 3.0])
def test_photo_to_fields(tmp_path, rotate):
    p = tmp_path / "r.png"
    _render(RECEIPT, p, rotate)
    text = ocr.ocr_image(p)
    f, conf, _ = extract.finalize(extract.extract_local(text))
    assert f["purchase_date"] == "2025-03-14"
    assert f["warranty_months"] == 12
    assert f["amount"] == 36580.0
    assert "refrigerator" in f["product"].lower()
