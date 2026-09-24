import pytest

from liquidloc.pipelines.training_runtime import resolve_train_device


def test_resolve_train_device_auto_can_select_cuda():
    result = resolve_train_device(
        "auto",
        allow_auto_cuda=True,
        cuda_available_fn=lambda: True,
        cuda_runtime_fn=lambda: True,
    )

    assert result == {
        "requested_device": "auto",
        "selected_device": "cuda",
        "cuda_available": True,
        "cuda_runtime_available": True,
    }


def test_resolve_train_device_explicit_cuda_falls_back_to_cpu():
    result = resolve_train_device(
        "cuda",
        allow_auto_cuda=True,
        cuda_available_fn=lambda: False,
        cuda_runtime_fn=lambda: False,
    )

    assert result["selected_device"] == "cpu"
    assert result["cuda_runtime_available"] is False


def test_resolve_train_device_rejects_unknown_request():
    with pytest.raises(ValueError, match="device must be one of"):
        resolve_train_device(
            "tpu",
            allow_auto_cuda=False,
            cuda_available_fn=lambda: False,
            cuda_runtime_fn=lambda: False,
        )
