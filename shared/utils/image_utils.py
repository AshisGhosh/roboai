import io
import base64
from PIL import Image


def pil_to_b64(image: Image) -> str:
    """Converts a PIL image to a base64 string."""
    # Save the image to a bytes buffer
    buffer = io.BytesIO()
    image.save(
        buffer, format=image.format
    )  # You can change the format to PNG or other supported formats

    # Encode the buffer to base64
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # Optionally, prepend the URI scheme to make it ready for HTML or data transfer
    img_base64 = f"data:image/jpeg;base64,{img_str}"
    return img_base64


def b64_to_pil(image_b64: str) -> Image:
    """Converts a base64 string to a PIL image."""
    # Remove the URI scheme
    img_str = image_b64.split(",")[1]

    # Decode the base64 string
    img_bytes = base64.b64decode(img_str)

    # Convert bytes to PIL image
    image = Image.open(io.BytesIO(img_bytes))
    return image
