import torch
from PIL import Image

from config import Config
from model import NSTModel
from utils import tensor_to_image, image_to_tensor, resize_tensor_image


def transfer_style(content_image: Image.Image, style_image: Image.Image,
                   num_transfer_iterations: int) -> Image.Image:
    """
    This function transfers style from style and content image with
    num_transfer_iterations iterations
    :param content_image: image with content
    :param style_image: image with style
    :param num_transfer_iterations: num steps in transfer
    :return: image with transferred style
    """
    content_image_size: tuple[int, int] = content_image.size
    tensor_content_image: torch.Tensor = \
        resize_tensor_image(image_to_tensor(content_image)).unsqueeze(0)
    tensor_style_image: torch.Tensor = \
        resize_tensor_image(image_to_tensor(style_image)).unsqueeze(0)
    tensor_input_image: torch.Tensor = tensor_content_image.clone().to(Config.device)

    model: NSTModel = NSTModel(tensor_content_image, tensor_style_image).to(Config.device)
    optimizer: torch.optim.LBFGS = torch.optim.LBFGS([tensor_input_image.requires_grad_()])

    current_iteration: int = 0
    while current_iteration < num_transfer_iterations:
        def closure() -> float:
            nonlocal tensor_input_image
            optimizer.zero_grad()

            model(tensor_input_image)

            current_content_loss: torch.Tensor = torch.tensor(0.0, device=Config.device)
            for content_layer in model.content_loss_layers:
                current_content_loss += content_layer.loss

            current_style_loss: torch.Tensor = torch.tensor(0.0, device=Config.device)
            for style_layer in model.style_loss_layers:
                current_style_loss += style_layer.loss

            cumulative_loss: torch.Tensor = current_content_loss + Config.alpha * current_style_loss
            cumulative_loss.backward(retain_graph=True)

            nonlocal current_iteration
            current_iteration += 1
            return cumulative_loss.item()

        optimizer.step(closure)

    tensor_input_image = tensor_input_image.detach().clamp(0, 1)
    output_image: Image.Image = tensor_to_image(tensor_input_image).resize(content_image_size)
    return output_image
