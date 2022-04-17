import sys
import base64
import typing as tp
from io import BytesIO
from PIL import Image
from dotenv import dotenv_values
from .. import PATH_TO_APP


def get_port(place: str = "BACKEND") -> int:
    """
    This function returns port from config file or 5555
    :param place: BACKEND or FRONTEND
    :return: port
    """
    config: dict[str, tp.Optional[str]] = dotenv_values(PATH_TO_APP / ".env")
    port: int = 5555

    if place + "_PORT" in config:
        try:
            port = int(config[place + "_PORT"] or "5555")
        except ValueError:
            sys.stderr.write("Incorrect port in .env file!")
    return port


def image_base64_encode(image: Image.Image) -> str:
    """
    This function encodes PIL.Image into base64 string
    :param image: image for encoding
    :return: string with encoded image
    """
    buffer: BytesIO = BytesIO()
    image.save(buffer, format='JPEG')
    return str(base64.b64encode(buffer.getvalue()))


def image_base64_decode(encoded_image: str) -> Image.Image:
    """
    This function decodes and construct PIL.Image from base64 string
    :param encoded_image: string with encoded image
    :return: decoded PIL.Image
    """
    code_only: str = encoded_image[23:]
    img_data: bytes = base64.b64decode(code_only)
    decoded_image: Image.Image = Image.open(BytesIO(img_data))
    return decoded_image
