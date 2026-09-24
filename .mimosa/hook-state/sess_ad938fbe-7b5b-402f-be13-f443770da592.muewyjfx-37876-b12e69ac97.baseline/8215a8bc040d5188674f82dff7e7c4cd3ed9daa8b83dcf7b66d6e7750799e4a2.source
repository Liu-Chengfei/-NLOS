"""LSTM 推理（lstm_inference）测试模块。

测试覆盖范围：
- LSTM 模型的推理流程
- 中间输出（bias/risk/scaling）的语义
- 检查点加载与设备兼容性

被测模块：liquidloc.models.lstm_inference"""

import pytest
import torch
import torch.nn.functional as F

from liquidloc.factories.model_factory import create_model
from liquidloc.models.lstm.inference import infer_intermediate
from liquidloc.models.lstm.trainer import (
    _build_head_mask_tensor,
    _compute_loss,
    _evaluate_model,
    _extract_window_and_target,
    _normalized_squared_error,
    _predict_batch,
    _project_train_outputs,
)


def _model_cfg():
    return {
        'name': 'lstm_ekf',
        'feature_order': ['dt', 'ax'],
        'window': {'size': 3, 'step': 1},
        'network': {'input_dim': 2, 'hidden_dim': 4, 'output_heads': ['bias', 'risk', 'uwb_scaling', 'vio_scaling']},
    }


def _structured_window():
    return {
        'current_modality': 'uwb',
        'feature_order': ['dt', 'ax'],
        'feature_values': [0.12, -0.03],
        'missing_mask': [0, 0],
        'dt': 0.1,
        'feature_window': [
            [0.10, -0.01],
            [0.11, -0.02],
            [0.12, -0.03],
        ],
        'missing_mask_window': [
            [0, 0],
            [0, 0],
            [0, 0],
        ],
    }


class _PrecomputedNetwork:
    def __init__(self, raw_outputs):
        self._raw_outputs = list(raw_outputs)

    def forward(self, window_tensor):
        return list(self._raw_outputs)


def test_normal_case():
    """正常场景测试。

    验证被测功能在标准输入下的正确行为，
    确保核心路径能正常执行并返回预期结果。
    """
    from liquidloc.common.constants import BRIDGE_SCALING_MIN
    model = create_model('lstm_ekf', _model_cfg())
    outputs = infer_intermediate(_structured_window(), model.network)
    assert torch.isfinite(torch.tensor(outputs.bias))
    assert 0.0 <= outputs.risk <= 1.0
    # v2: scaling 下界放宽至 BRIDGE_SCALING_MIN（0.5），配合 soft-mask 释放 Liquid 调节空间
    assert outputs.uwb_scaling >= BRIDGE_SCALING_MIN
    assert outputs.vio_scaling >= BRIDGE_SCALING_MIN


def test_boundary_case():
    """边界场景测试。
    
    验证被测功能在边界条件下的行为，
    确保极端输入不会导致异常或错误结果。
    """
    outputs = infer_intermediate(
        _structured_window(),
        _PrecomputedNetwork([-1.0, 1.5, -0.2, 0.7]),
    )
    assert outputs.bias == pytest.approx(0.0)  # bias 非负约束：max(0.0, -1.0) = 0.0
    assert outputs.risk == pytest.approx(float(torch.sigmoid(torch.tensor(1.5))))
    assert outputs.uwb_scaling == pytest.approx(1.0)
    assert outputs.vio_scaling == pytest.approx(1.0)  # 模态输出合约：current_modality='uwb'，非当前模态 scaling 强制为 1.0


def test_lstm_inference_is_invariant_to_liquid_specific_readout_context_fields():
    """读出上下文测试：lstm inference is invariant to liquid specific。\n\n验证 lstm inference is invariant to liquid specific 的读出上下文构建，\n确保协方差摘要和门控标志正确传递。
    """
    model = create_model('lstm_ekf', _model_cfg())
    base_window = _structured_window()
    augmented_window = {
        **base_window,
        'readout_context_by_name': {
            'state_cov_trace': 9.0,
            'pos_cov': 2.0,
            'last_innovation_norm': 4.0,
            'last_gate_skip_flag': 1.0,
        },
        'readout_context_observed_by_name': {
            'state_cov_trace': True,
            'pos_cov': True,
            'last_innovation_norm': True,
            'last_gate_skip_flag': True,
        },
        'context_vector': torch.tensor([0.7] * 28, dtype=torch.float32),
        'filter_context_vector': torch.tensor([0.9] * 8, dtype=torch.float32),
    }

    base_outputs = model.predict_intermediate_tensors(base_window)
    augmented_outputs = model.predict_intermediate_tensors(augmented_window)

    for key in ('bias', 'risk', 'uwb_scaling', 'vio_scaling'):
        assert float(base_outputs[key].detach().item()) == pytest.approx(
            float(augmented_outputs[key].detach().item()),
            abs=1e-6,
        )


def test_lstm_model_does_not_expose_liquid_specific_readout_modules():
    """不侵入测试：lstm model。\n\n验证 lstm model 不会产生副作用，\n确保功能隔离性。
    """
    model = create_model('lstm_ekf', _model_cfg())

    # LSTM 现在与 Liquid 对齐，拥有 risk_calibration 模块。
    assert hasattr(model, 'risk_calibration')
    # 但 LSTM 网络不应暴露 Liquid 特有的共享投影 / 池化模块。
    assert not hasattr(model.network, 'shared_projection')
    assert not hasattr(model.network, 'extract_shared_features')
    assert not hasattr(model.network, 'final_hidden')
    assert not hasattr(model.network, 'pooled_hidden')

    outputs = model.predict_intermediate_tensors(_structured_window())
    assert set(outputs.keys()) == {'bias', 'risk', 'uwb_scaling', 'vio_scaling'}


def test_invalid_case():
    """无效输入测试。
    
    验证被测功能对无效输入的拒绝行为，
    确保缺少必要参数时抛出 ValueError。
    """
    with pytest.raises(ValueError):
        infer_intermediate(
            {
                'current_modality': 'uwb',
                'feature_order': ['dt', 'ax'],
                'feature_values': [0.12, -0.03],
                'missing_mask': [0, 0],
            },
            _PrecomputedNetwork([0.1, 0.2, 1.0, 1.0]),
        )


def test_structured_window_missing_dt_is_rejected():
    """拒绝测试：structured window missing dt is。\n\n验证被测功能对不合法的 structured window missing dt is 输入正确抛出异常，\n防止无效参数通过验证。
    """
    window_tensor = _structured_window()
    window_tensor.pop('dt')

    with pytest.raises(ValueError, match='feature_window.dt must be provided explicitly'):
        infer_intermediate(window_tensor, _PrecomputedNetwork([0.1, 0.2, 1.0, 1.0]))


@pytest.mark.parametrize('window_tensor', [torch.zeros((3, 2), dtype=torch.float32), torch.zeros((1, 3, 2), dtype=torch.float32)])
def test_infer_intermediate_rejects_2d_and_3d_tensor_inputs(window_tensor):
    """拒绝测试：infer intermediate。\n\n验证被测功能对 infer intermediate 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    with pytest.raises(TypeError, match='window_tensor must be a structured feature window mapping'):
        infer_intermediate(window_tensor, _PrecomputedNetwork([0.0, 0.0, 0.0, 0.0]))


@pytest.mark.parametrize('window_tensor', [torch.zeros((3, 2), dtype=torch.float32), torch.zeros((1, 3, 2), dtype=torch.float32)])
def test_lstm_trainer_rejects_2d_and_3d_tensor_inputs(window_tensor):
    """拒绝测试：lstm trainer。\n\n验证被测功能对 lstm trainer 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    with pytest.raises(TypeError, match=r'^window_tensor must be a structured feature window mapping\.$'):
        _extract_window_and_target(
            {
                'window_tensor': window_tensor,
                'target_intermediate': {
                    'bias': 0.0,
                    'risk': 0.5,
                    'uwb_scaling': 0.75,
                    'vio_scaling': 0.8,
                },
            }
        )


def test_lstm_trainer_target_coercion_enforces_neutral_floor_scaling():
    """地板测试：lstm trainer target coercion enforces neutral。\n\n验证 lstm trainer target coercion enforces neutral 的地板值约束，\n确保输出不低于最小值。
    """
    projected = infer_intermediate(
        _structured_window(),
        _PrecomputedNetwork([0.0, 0.0, -1000.0, -1000.0]),
    )

    assert projected.uwb_scaling == pytest.approx(1.0)
    assert projected.vio_scaling == pytest.approx(1.0)

    _, target = _extract_window_and_target(
        {
            'window_tensor': _structured_window(),
            'target_intermediate': {
                'bias': -0.4,
                'risk': 1.2,
                'uwb_scaling': 1.25,
                'vio_scaling': 1.3,
            },
        }
    )
    assert target['bias'] == pytest.approx(0.0)  # bias 非负约束：max(0.0, -0.4) = 0.0
    assert target['risk'] == pytest.approx(1.0)
    assert target['uwb_scaling'] == pytest.approx(1.25)
    assert target['vio_scaling'] == pytest.approx(1.3)


def test_lstm_trainer_target_coercion_rejects_subunit_scaling():
    """拒绝测试：lstm trainer target coercion。\n\n验证被测功能对 lstm trainer target coercion 的拒绝行为，\n确保不合法输入被正确拦截。
    """
    with pytest.raises(ValueError, match=r'target_intermediate\.uwb_scaling must be finite and >= 1\.0\.'):
        _extract_window_and_target(
            {
                'window_tensor': _structured_window(),
                'target_intermediate': {
                    'bias': -0.4,
                    'risk': 1.2,
                    'uwb_scaling': 0.75,
                    'vio_scaling': 1.0,
                },
            }
        )


def test_lstm_train_and_inference_bias_keep_signed_semantics():
    projected = _project_train_outputs(torch.tensor([-0.75, 0.0, 0.0, 0.0]))
    assert projected['bias'].item() == pytest.approx(0.0)  # bias 非负约束：clamp(-0.75, min=0.0) = 0.0

    outputs = infer_intermediate(
        _structured_window(),
        _PrecomputedNetwork([-0.75, 0.0, 0.0, 0.0]),
    )
    assert outputs.bias == pytest.approx(0.0)  # bias 非负约束：max(0.0, -0.75) = 0.0


def test_lstm_trainer_loss_keeps_scaling_heads_trainable_with_neutral_floor_targets():
    """保持测试：lstm trainer loss。\n\n验证 lstm trainer loss 的保持行为，\n确保特定属性在处理过程中不变。
    """
    projected = _project_train_outputs(torch.tensor([0.0, 0.0, -1000.0, -1000.0]))
    assert projected['uwb_scaling'].item() == pytest.approx(1.0)
    assert projected['vio_scaling'].item() == pytest.approx(1.0)

    model = create_model('lstm_ekf', _model_cfg())
    with torch.no_grad():
        model.network.output_layer.weight.zero_()
        model.network.output_layer.bias.zero_()

    target = {
        'bias': 0.0,
        'risk': 0.0,
        'uwb_scaling': 1.4,
        'vio_scaling': 1.3,
    }

    loss = _compute_loss(model, _structured_window(), target)
    loss.backward()

    assert model.network.output_layer.bias.grad is not None
    assert model.network.output_layer.bias.grad[2].abs().item() > 0.0
    assert model.network.output_layer.bias.grad[3].abs().item() == pytest.approx(0.0)


def test_lstm_trainer_loss_masks_inactive_bias_and_uwb_heads_for_vio_targets():
    model = create_model('lstm_ekf', _model_cfg())
    with torch.no_grad():
        model.network.output_layer.weight.zero_()
        model.network.output_layer.bias.zero_()

    window_tensor = _structured_window()
    window_tensor['current_modality'] = 'vio'
    target = {
        'bias': 1.2,
        'risk': 0.0,
        'uwb_scaling': 1.8,
        'vio_scaling': 1.3,
    }

    loss = _compute_loss(model, window_tensor, target)
    loss.backward()

    assert model.network.output_layer.bias.grad is not None
    assert model.network.output_layer.bias.grad[0].abs().item() == pytest.approx(0.0)
    assert model.network.output_layer.bias.grad[2].abs().item() == pytest.approx(0.0)
    assert model.network.output_layer.bias.grad[3].abs().item() > 0.0


def test_lstm_eval_handles_mixed_sequence_lengths():
    model = create_model('lstm_ekf', _model_cfg())
    val_windows = [
        {
            'window_tensor': {
                'current_modality': 'uwb',
                'feature_order': ['dt', 'ax'],
                'feature_values': [0.11, -0.02],
                'missing_mask': [0, 0],
                'dt': 0.1,
                'feature_window': [[0.10, -0.01], [0.11, -0.02]],
                'missing_mask_window': [[0, 0], [0, 0]],
            },
            'target_intermediate': {
                'bias': 0.0,
                'risk': 0.0,
                'uwb_scaling': 1.0,
                'vio_scaling': 1.0,
            },
        },
        {
            'window_tensor': _structured_window(),
            'target_intermediate': {
                'bias': 0.0,
                'risk': 0.0,
                'uwb_scaling': 1.0,
                'vio_scaling': 1.0,
            },
        },
    ]

    score = _evaluate_model(model, val_windows)
    assert isinstance(score, float)
    assert score >= 0.0


def test_lstm_selection_score_is_tail_weighted_and_batch_size_invariant():
    model = create_model('lstm_ekf', _model_cfg())
    with torch.no_grad():
        model.network.output_layer.weight.zero_()
        model.network.output_layer.bias.zero_()

    bias_prediction = 0.0
    scaling_prediction = 1.0
    low_risk_window = {
        'window_tensor': _structured_window(),
        'target_intermediate': {
            'bias': bias_prediction,
            'risk': 0.5,
            'uwb_scaling': scaling_prediction,
            'vio_scaling': scaling_prediction,
        },
        'target_trace': {
            'alignment_risk': 0.0,
            'observation_risk': 0.0,
            'quality_risk': 0.0,
            'modality_signal': 0.0,
        },
    }
    high_risk_window = {
        'window_tensor': _structured_window(),
        'target_intermediate': {
            'bias': 0.0,
            'risk': 1.0,
            'uwb_scaling': 1.2,
            'vio_scaling': scaling_prediction,
        },
        'target_trace': {
            'alignment_risk': 1.0,
            'observation_risk': 1.0,
            'quality_risk': 1.0,
            'modality_signal': 1.0,
        },
    }

    model._cached_val_windows = None
    model._eval_batch_size = 1
    loss_batch_1 = _evaluate_model(model, [low_risk_window, high_risk_window])
    score_batch_1 = float(model._selection_score)
    model._cached_val_windows = None
    model._eval_batch_size = 2
    loss_batch_2 = _evaluate_model(model, [low_risk_window, high_risk_window])
    score_batch_2 = float(model._selection_score)
    model._cached_val_windows = None
    model._eval_batch_size = 1
    _evaluate_model(model, [high_risk_window])
    high_only_score = float(model._selection_score)

    assert loss_batch_1 == pytest.approx(loss_batch_2)
    assert score_batch_1 == pytest.approx(score_batch_2)
    assert score_batch_1 > (high_only_score / 2.0)


def test_lstm_batch_eval_uses_current_step_context_consistently():
    """使用测试：lstm batch eval。\n\n验证被测功能正确使用 lstm batch eval，\n确保内部依赖被正确调用。
    """
    model = create_model(
        'lstm_ekf',
        {
            'name': 'lstm_ekf',
            # 探针接 valid 数值位 (context: modal 0,1 | valid 2,3 | ...)
            'feature_order': ['valid', 'modality_gap_dt'],
            'window': {'size': 3, 'step': 1},
            'network': {
                'input_dim': 4,
                'hidden_dim': 4,
                'output_heads': ['bias', 'risk', 'uwb_scaling', 'vio_scaling'],
            },
        },
    )
    with torch.no_grad():
        for parameter in model.network.lstm.parameters():
            parameter.zero_()
        model.network.output_layer.weight.zero_()
        model.network.output_layer.bias.zero_()
        model.network.output_layer.weight[0, model.network.hidden_dim + 2] = 1.0

    base_window = {
        'current_modality': 'uwb',
        'feature_order': ['valid', 'modality_gap_dt'],
        'feature_values': [0.12, 0.1],
        'missing_mask': [0, 0],
        'dt': 0.1,
        'feature_window': [
            [0.10, 0.1],
            [0.11, 0.1],
            [0.12, 0.1],
        ],
        'missing_mask_window': [
            [0, 0],
            [0, 0],
            [0, 0],
        ],
    }
    richer_window = dict(base_window)
    richer_window['feature_values'] = [0.82, 0.1]
    richer_window['feature_window'] = [
        [0.10, 0.1],
        [0.11, 0.1],
        [0.82, 0.1],
    ]
    base_sequence = torch.cat(
        [
            torch.tensor(base_window['feature_window'], dtype=torch.float32),
            torch.tensor(base_window['missing_mask_window'], dtype=torch.float32),
        ],
        dim=1,
    )
    richer_sequence = torch.cat(
        [
            torch.tensor(richer_window['feature_window'], dtype=torch.float32),
            torch.tensor(richer_window['missing_mask_window'], dtype=torch.float32),
        ],
        dim=1,
    )

    predicted_batch = _predict_batch(
        model,
        torch.stack([base_sequence, richer_sequence], dim=0),
        modalities=['uwb', 'uwb'],
    )

    assert float(predicted_batch[1, 0].detach().item()) > float(predicted_batch[0, 0].detach().item())
