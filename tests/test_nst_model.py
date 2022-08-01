import torch
import pytest
import typing as tp
from PIL import Image
from torchvision import transforms

from config import Config
from model import NSTModel, transfer_style
from utils import image_to_tensor
from layers import NormalizationLayer, ContentLossLayer, StyleLossLayer


@pytest.fixture(scope="module")
def tensor_content_image() -> tp.Generator[torch.Tensor, None, None]:
    with Image.open("./test_data/img.png") as image:
        tensor_image: torch.Tensor = image_to_tensor(image.crop((50, 50, 250, 250)))
        tensor_image = tensor_image.view(-1, *tensor_image.shape)
        tensor_image = transforms.Resize(Config.working_image_size)(tensor_image)
        yield tensor_image


@pytest.fixture(scope="module")
def tensor_style_image() -> tp.Generator[torch.Tensor, None, None]:
    with Image.open("./test_data/img.png") as image:
        tensor_image: torch.Tensor = image_to_tensor(image.crop((400, 400, 650, 650)))
        tensor_image = tensor_image.view(-1, *tensor_image.shape)
        tensor_image = transforms.Resize(Config.working_image_size)(tensor_image)
        yield tensor_image


def test_nst_model_constructs(tensor_content_image: torch.Tensor,
                              tensor_style_image: torch.Tensor) -> None:
    NSTModel(tensor_content_image, tensor_style_image)


def test_nst_model_structure(tensor_content_image: torch.Tensor,
                             tensor_style_image: torch.Tensor) -> None:
    model: NSTModel = NSTModel(tensor_content_image, tensor_style_image)

    assert len(list(model.nst_model.children())) == 18
    assert isinstance(next(model.nst_model.children()), NormalizationLayer)
    assert len([layer for layer in model.nst_model.children() if
                isinstance(layer, ContentLossLayer)]) == len(Config.default_content_layers)
    assert len([layer for layer in model.nst_model.children() if
                isinstance(layer, StyleLossLayer)]) == len(Config.default_style_layers)


def test_nst_model_collects_gradients(tensor_content_image: torch.Tensor,
                                      tensor_style_image: torch.Tensor) -> None:
    model: NSTModel = NSTModel(tensor_content_image, tensor_style_image)
    input_img: torch.Tensor = torch.randn_like(tensor_content_image).requires_grad_()

    model(input_img)

    cumulative_style_loss: torch.Tensor = torch.tensor(0.0)
    for style_layer in model._style_loss_layers:
        cumulative_style_loss += style_layer.loss
        assert style_layer.loss.item() != pytest.approx(0.0, abs=1e-6)

    cumulative_content_loss: torch.Tensor = torch.tensor(0.0)
    for content_layer in model._content_loss_layers:
        cumulative_content_loss += content_layer.loss
        assert content_layer.loss.item() != pytest.approx(0.0, abs=1e-6)

    loss = cumulative_style_loss + cumulative_content_loss
    loss.backward()

    assert input_img.grad is not None
    assert input_img.grad.data != pytest.approx(torch.zeros_like(input_img.grad.data), abs=1e-6)


def test_transfer_style() -> None:
    content_image: Image.Image = Image.open("./test_data/sample_img.png")
    style_image: Image.Image = Image.open("./test_data/style_img.png")
    result: Image.Image = transfer_style(content_image, style_image, num_transfer_iterations=200)
    result.save("./test_data/result.png", "PNG")
