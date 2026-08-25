import torch.optim as optim


def get_optimizer(params, optimizer_name='adam', lr=1e-4):
    optimizer_name = optimizer_name.lower()

    if optimizer_name == 'adam':
        return optim.Adam(params, lr=lr, betas=(0.9, 0.999), weight_decay=0.0)

    if optimizer_name == 'adamw':
        return optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=0.0)

    if optimizer_name == 'sgd':
        return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0)

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")
