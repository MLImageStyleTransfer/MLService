import torch
import torch.nn.functional

from config import Config
from utils import normalize_tensor_image


class NormalizationLayer(torch.nn.Module):
    """
    Normalize input with mean and std from Config
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        :param inp: [1, 3, H, W] tensor
        :return: [1, 3, H, W] tensor with normalized values
        """
        return normalize_tensor_image(inp)


class ContentLossLayer(torch.nn.Module):
    """
    Layer for computing content loss. It doesn't change input
    """
    def __init__(self, target: torch.Tensor) -> None:
        """
        :param target: feature map of original content image, [bs, C, H, W] tensor
        """
        assert len(target.shape) == 4

        super().__init__()
        self.target: torch.Tensor = target
        self.loss: torch.Tensor = torch.tensor(0.0, device=Config.device)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        :param inp: [1, C, H, W] tensor
        :return: inp without any changes
        """
        self.loss = torch.nn.functional.mse_loss(inp, self.target)
        return inp


class StyleLossLayer(torch.nn.Module):
    """
    Layer for computing style loss. It doesn't change input
    """
    def __init__(self, target: torch.Tensor) -> None:
        """
        :param target: feature map of original style image, [bs, C, H, W] tensor
        """
        assert len(target.shape) == 4

        super().__init__()
        self.target_gram_matrix = self.__gram_matrix(target)
        self.loss: torch.Tensor = torch.tensor(0.0, device=Config.device)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        :param inp: [1, C, H, W] tensor
        :return: inp without any changes
        """
        gram_matrix: torch.Tensor = self.__gram_matrix(inp)
        self.loss = torch.nn.functional.mse_loss(gram_matrix, self.target_gram_matrix)
        return inp

    @staticmethod
    def __gram_matrix(tensor: torch.Tensor) -> torch.Tensor:
        """
        Reshapes input tensor input 2d-matrix and computes the Gram matrix
        :param tensor: [1, C, H, W] tensor
        :return: [C, H * W] tensor - the Gram matrix
        """
        bs, c, h, w = tensor.size()
        square_matrix: torch.Tensor = tensor.view(bs * c, h * w)
        result: torch.Tensor = torch.mm(square_matrix, square_matrix.t()).div(bs * c * h * w)
        return result
