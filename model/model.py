import torch
import torch.nn as nn
import typing as tp
from pathlib import Path
from collections import defaultdict
from torchvision.models import vgg19

from config import Config
from layers import NormalizationLayer, ContentLossLayer, StyleLossLayer


class NSTModel(torch.nn.Module):
    """
    Model for neural style transfer
    """
    def __init__(self,
                 content_tensor: torch.Tensor,
                 style_tensor: torch.Tensor,
                 path_to_save_dir: Path = Path('model/pretrained')) -> None:
        """
        Initialize NSTModel
        :param content_tensor: tensor representation of content image, [1, 3, H, W]
        :param style_tensor: tensor representation of style image, [1, 3, H, W]
        :param path_to_save_dir: path for downloading pretrained model
        """
        super().__init__()
        self.nst_model: nn.Sequential = nn.Sequential()
        self.nst_model.add_module("normalization", NormalizationLayer())
        self._path_to_save_dir: Path = path_to_save_dir
        self._style_loss_layers: list[StyleLossLayer] = []
        self._content_loss_layers: list[ContentLossLayer] = []
        self._init_nst_model(content_tensor, style_tensor)

    def collect_style_transfer_loss(self) -> torch.Tensor:
        current_content_loss: torch.Tensor = torch.tensor(0.0, device=Config.device)
        for content_layer in self._content_loss_layers:
            current_content_loss += content_layer.loss

        current_style_loss: torch.Tensor = torch.tensor(0.0, device=Config.device)
        for style_layer in self._style_loss_layers:
            current_style_loss += style_layer.loss

        cumulative_loss: torch.Tensor = current_content_loss + Config.alpha * current_style_loss
        return cumulative_loss

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        output: torch.Tensor = self.nst_model(inp)
        return output

    def _init_nst_model(self,
                        content_tensor: torch.Tensor,
                        style_tensor: torch.Tensor) -> None:
        """
        initializes model that will be used for neural style_transfer.
        :param content_tensor: tensor with content image.
        :param style_tensor: tensor with style image
        """
        base_model: nn.Module = self._load_pretrained_base_model()
        layers_counter: defaultdict[str, int] = defaultdict(int)
        for layer in base_model.children():
            curr_layer_name: str = self._update_nst_model(layer, layers_counter)
            self._add_content_style_layers(curr_layer_name, layers_counter, content_tensor, style_tensor)
        self._crop_nst_model()

    def _load_pretrained_base_model(self, model_type: str = "vgg19") -> nn.Module:
        """
        Loads selected pretrained model from torch hub. Now only vgg19 is accessible.
        :param model_type: type of base model.
        :return: base model.
        """
        path_to_pretrained = Config.PATH_TO_PROJECT / self._path_to_save_dir
        if not path_to_pretrained.exists():
            path_to_pretrained.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(path_to_pretrained))
        base_model: tp.Optional[nn.Module] = None
        if model_type == "vgg19":
            base_model: nn.Module = vgg19(pretrained=True).features.to(Config.device).eval()
        assert base_model is not None
        return base_model

    def _update_nst_model(self,
                          layer: nn.Module,
                          layers_counter: defaultdict[str, int]) -> str:
        """
        Copy layers from base model to NSTModel
        :param layer: layer for coping
        :param layers_counter: counter for layers
        :return: layer name
        """
        layer_name: str = ""
        if isinstance(layer, nn.Conv2d):
            layers_counter['conv'] += 1
            layer_name = 'conv_' + str(layers_counter['conv'])
        elif isinstance(layer, nn.ReLU):
            layers_counter['relu'] += 1
            layer_name = 'relu_' + str(layers_counter['relu'])
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            layers_counter['pool'] += 1
            layer_name = 'pool_' + str(layers_counter['pool'])

        self.nst_model.add_module(layer_name, layer)
        return layer_name

    def _add_content_style_layers(self,
                                  layer_name: str,
                                  layers_counter: defaultdict[str, int],
                                  content_tensor: torch.Tensor,
                                  style_tensor: torch.Tensor) -> None:
        """
        Add content and style loss layers if it is needed.
        :param layer_name: previous layer name
        :param layers_counter: counter for layers
        :param content_tensor: tensor representation of content image, [1, 3, H, W]
        :param style_tensor: tensor representation of style image, [1, 3, H, W]
        """
        target_tensor: torch.Tensor
        loss_layer_name: str
        if layer_name in Config.default_content_layers:
            target_tensor = self.nst_model(content_tensor)
            content_loss_layer: ContentLossLayer = ContentLossLayer(target_tensor)

            loss_layer_name = 'content_loss_' + str(layers_counter['conv'])
            self.nst_model.add_module(loss_layer_name, content_loss_layer)
            self._content_loss_layers.append(content_loss_layer)

        if layer_name in Config.default_style_layers:
            target_tensor = self.nst_model(style_tensor)
            style_loss_layer: StyleLossLayer = StyleLossLayer(target_tensor)

            loss_layer_name = 'style_loss_' + str(layers_counter['conv'])
            self.nst_model.add_module(loss_layer_name, style_loss_layer)
            self._style_loss_layers.append(style_loss_layer)

    def _crop_nst_model(self) -> None:
        """
        Crops unnecessary layers after last content or style loss layer
        """
        for i in range(len(self.nst_model) - 1, -1, -1):
            if isinstance(self.nst_model[i], ContentLossLayer) or \
                    isinstance(self.nst_model[i], StyleLossLayer):
                self.nst_model = self.nst_model[:i + 1]
                break
