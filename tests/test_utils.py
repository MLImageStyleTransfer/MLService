import torch
import pytest
import typing as tp
from PIL import Image

from utils import normalize_tensor_image, denormalize_tensor_image, tensor_to_image, image_to_tensor


@pytest.fixture(scope="module")
def image() -> tp.Generator[Image.Image, None, None]:
    with Image.open("./test_data/img.png") as image:
        yield image


def test_tensor_to_image() -> None:
    tensor_image: torch.Tensor = torch.randn([3, 128, 128]).clamp_(0, 1)
    image: Image.Image = tensor_to_image(tensor_image)
    assert isinstance(image, Image.Image)
    assert (image.width == 128) and (image.height == 128)


def test_image_to_tensor(image: Image.Image) -> None:
    tensor_image: torch.Tensor = image_to_tensor(image)
    assert list(tensor_image.shape) == [3, image.height, image.width]


def test_normalize_and_denormalize_tensor_image(image: Image.Image) -> None:
    tensor_image: torch.Tensor = image_to_tensor(image.crop((50, 50, 100, 100)))
    normalized_tensor_image: torch.Tensor = normalize_tensor_image(tensor_image)
    denormalized_tensor_image: torch.Tensor = denormalize_tensor_image(normalized_tensor_image)
    assert denormalized_tensor_image.shape == tensor_image.shape
    assert denormalized_tensor_image == pytest.approx(tensor_image, abs=0.001)
