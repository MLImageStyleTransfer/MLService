import torch


class Config:
    device: torch.device
    working_image_size: tuple[int, int]
    if torch.cuda.is_available():
        device = torch.device('cuda')
        working_image_size = (256, 256)
    else:
        device = torch.device('cpu')
        working_image_size = (128, 128)

    nn_input_mean: torch.Tensor = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(device)
    nn_input_std: torch.Tensor = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(device)
