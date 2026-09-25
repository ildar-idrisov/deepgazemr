from pathlib import Path

import pytest
import torch
import torchvision

import deepgazemr.model as model_module

DATA = Path(__file__).resolve().parents[1] / 'data'


@pytest.fixture
def model(monkeypatch):
    # the checkpoint contains the VGG weights, so skip downloading the ImageNet ones
    vgg19 = torchvision.models.vgg19
    monkeypatch.setattr(model_module.models, 'vgg19', lambda pretrained=False: vgg19(pretrained=False))
    model = model_module.DeepGazeMR()
    model.load_state_dict(torch.load(DATA / 'deepgazemr-ledov.pt', map_location='cpu')['model_state_dict'])
    model.center_bias = torch.load(DATA / 'center-bias-ledov.pt', map_location='cpu')
    return model.eval()


def _video(frames):
    return torch.rand(frames, 3, 36, 64, generator=torch.Generator().manual_seed(0))


def test_forward_returns_normalized_log_density_in_input_dtype(model):
    with torch.no_grad():
        log_density = model.forward(_video(16))
    assert log_density.shape == (36, 64)
    assert log_density.dtype == torch.float32
    assert log_density.exp().sum().item() == pytest.approx(1.0, abs=1e-4)


def test_predict_matches_forward(model):
    video = _video(19)
    with torch.no_grad():
        predictions = list(model.predict(video))
        assert all(p is None for p in predictions[:15])
        for i in range(15, 19):
            assert torch.allclose(predictions[i], model.forward(video[i - 15:i + 1]), atol=1e-4)


def test_batched_clips_are_rejected(model):
    batch = _video(16)[None].repeat(2, 1, 1, 1, 1)
    with pytest.raises(ValueError):
        model.forward(batch)
    with pytest.raises(ValueError):
        next(model.predict(batch))
