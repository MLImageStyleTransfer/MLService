from PIL import Image

from model import transfer_style
from app.utils import image_base64_encode, image_base64_decode


def controller(base64_encoded_content_image: str, base64_encoded_style_image: str,
               transfer_coefficient: float) -> str:
    """
    Controller function takes base64 encoded images with content and style, applies
    style transfer and returns result image encoded with base64
    :param base64_encoded_content_image: base64 string representation of content image
    :param base64_encoded_style_image: base64 string representation of style image
    :param transfer_coefficient: coefficient of transfer, takes value in [0, 1]
    :return: image with transferred style
    """
    assert 0 <= transfer_coefficient <= 1

    content_image: Image.Image = image_base64_decode(base64_encoded_content_image)
    style_image: Image.Image = image_base64_decode(base64_encoded_style_image)
    result: Image.Image = transfer_style(content_image, style_image,
                                         int(transfer_coefficient * 200))
    return image_base64_encode(result)
