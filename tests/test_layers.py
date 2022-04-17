import torch
import pytest
import typing as tp
from PIL import Image

from utils import image_to_tensor, normalize_tensor_image
from layers import NormalizationLayer, ContentLossLayer, StyleLossLayer


@pytest.fixture(scope="module")
def tensor_image() -> tp.Generator[torch.Tensor, None, None]:
    with Image.open("./test_data/img.png") as image:
        tensor_image: torch.Tensor = image_to_tensor(image.crop((50, 50, 250, 250)))
        tensor_image = tensor_image.view(-1, *tensor_image.shape)
        yield tensor_image


def test_normalization_layer(tensor_image: torch.Tensor) -> None:
    layer: NormalizationLayer = NormalizationLayer()
    layer_output: torch.Tensor = layer(tensor_image)
    assert layer_output.shape == tensor_image.shape
    assert layer_output == pytest.approx(normalize_tensor_image(tensor_image), abs=1e-6)


def test_content_loss_layer(tensor_image: torch.Tensor) -> None:
    layer: ContentLossLayer = ContentLossLayer(tensor_image)
    layer_output: torch.Tensor = layer(tensor_image)

    assert layer_output.shape == tensor_image.shape
    assert layer_output == pytest.approx(tensor_image, abs=1e-6)
    assert layer.loss == pytest.approx(0.0, abs=1e-6)

    random_input: torch.Tensor = torch.randn_like(tensor_image)
    layer_output = layer(random_input)

    assert layer_output.shape == tensor_image.shape
    assert layer_output == pytest.approx(random_input, abs=1e-6)
    assert layer.loss != pytest.approx(0.0, abs=1e-6)


def test_style_loss_layer(tensor_image: torch.Tensor) -> None:
    layer: StyleLossLayer = StyleLossLayer(tensor_image)
    layer_output: torch.Tensor = layer(tensor_image)

    assert layer_output.shape == tensor_image.shape
    assert layer_output == pytest.approx(tensor_image, abs=1e-6)
    assert layer.loss == pytest.approx(0.0, abs=1e-6)

    random_input: torch.Tensor = torch.randn_like(tensor_image)
    layer_output = layer(random_input)

    assert layer_output.shape == tensor_image.shape
    assert layer_output == pytest.approx(random_input, abs=1e-6)
    assert layer.loss != pytest.approx(0.0, abs=1e-6)
