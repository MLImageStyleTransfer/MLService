import torch
from pathlib import Path


class Config:
    """
    Config class contains main parameters of style transfer:

        PATH_TO_PROJECT - absolute path to the project root

        device - target device for computations

        working_image_size - size of input style and content images

        nn_input_mean - mean values for each channel for normalization (ImageNet)

        nn_input_std - str values for each channel for normalization (ImageNet)

        default_content_layers - list of layers after that ContentLossLayers is placed

        default_style_layers - list of layers after that  StyleLossLayers is placed
    """
    PATH_TO_PROJECT: Path = Path().cwd().parent.absolute().resolve()

    device: torch.device
    working_image_size: tuple[int, int]
    if torch.cuda.is_available():
        device = torch.device("cuda")
        working_image_size = (256, 256)
    else:
        device = torch.device("cpu")
        working_image_size = (128, 128)

    nn_input_mean: torch.Tensor = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(device)
    nn_input_std: torch.Tensor = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(device)

    default_content_layers: list[str] = ["conv_4"]
    default_style_layers: list[str] = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]
