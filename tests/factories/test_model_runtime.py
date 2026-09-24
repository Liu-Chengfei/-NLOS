import torch
from torch import nn

from liquidloc.factories.model_runtime import build_module_runtime_resource_meta


def test_build_module_runtime_resource_meta_accepts_module_tuples():
    first = nn.Linear(3, 2)
    second = nn.Linear(2, 1)

    result = build_module_runtime_resource_meta((first, None, second))

    expected_params = float(sum(parameter.numel() for module in (first, second) for parameter in module.parameters()))
    assert result["params"] == expected_params
    assert result["ram_peak"] == result["ram_peak_mb"]
    assert result["ram_peak"] >= 1.0


def test_model_factory_uses_shared_runtime_resource_builder():
    from liquidloc.factories import model_factory

    assert model_factory._build_module_runtime_resource_meta is build_module_runtime_resource_meta
