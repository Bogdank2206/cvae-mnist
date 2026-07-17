import base64
import io

import torch
from PIL import Image


def tensor_to_base64_png(image_tensor: torch.Tensor) -> str:
    """Конвертирует тензор (1, 1, 28, 28) в data-URI base64 PNG."""
    pixels = image_tensor.squeeze().cpu().numpy()  # (28, 28)
    img = Image.fromarray((pixels * 255).astype("uint8"), mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"
