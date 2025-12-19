import torch


def test_tensor_shape_matches():
    x = torch.zeros((4, 5))
    assert x.shape == (4, 5)


def test_linear_layer_output_shape():
    layer = torch.nn.Linear(10, 3)
    x = torch.randn(2, 10)
    y = layer(x)
    assert y.shape == (2, 3)


def test_loss_is_scalar():
    pred = torch.tensor([0.2, 0.7, 0.1])
    target = torch.tensor([0.0, 1.0, 0.0])
    loss = torch.nn.functional.mse_loss(pred, target)
    assert loss.ndim == 0
