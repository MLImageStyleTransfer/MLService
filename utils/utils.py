import torch
import torchvision
from PIL import Image

from config import Config


def normalize_tensor_image(image: torch.Tensor) -> torch.Tensor:
    """
    Normalizes tensor representation of image. Mean and std are taken from Config
    :param image: image for normalization
    :return: normalized image
    """
    return (image - Config.nn_input_mean) / Config.nn_input_std


def denormalize_tensor_image(image: torch.Tensor) -> torch.Tensor:
    """
    Denormalizes tensor representation of image. Mean and std are taken form Config
    :param image: normalized image
    :return: denormalized image
    """
    return image * Config.nn_input_std + Config.nn_input_mean


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """
    Converts PIL image to torch tensor
    :param image: PIL image
    :return: tensor representation of image [C, H, W]
    """
    return torchvision.transforms.ToTensor()(image)


def tensor_to_image(tensor_image: torch.Tensor) -> Image.Image:
    """
    Converts tensor image to PIL image
    :param tensor_image: tensor [3, H, W] or [1, 3, H, W]
    :return: PIL image
    """
    if list(tensor_image.shape) != [3, *Config.working_image_size]:
        tensor_image = tensor_image.view(3, *Config.working_image_size)
    return torchvision.transforms.ToPILImage()(tensor_image)


def resize_tensor_image(tensor_image: torch.Tensor,
                        new_size: tuple[int, int] = Config.working_image_size) -> torch.Tensor:
    """
    Resizes tensor representation of image into new size
    :param tensor_image: tensor representation of image [3, H, W] or [1, 3, H, W]
    :param new_size: new size of image
    :return: tensor representation of image with new size
    """
    result: torch.Tensor = torchvision.transforms.Resize(new_size)(tensor_image)
    return result
