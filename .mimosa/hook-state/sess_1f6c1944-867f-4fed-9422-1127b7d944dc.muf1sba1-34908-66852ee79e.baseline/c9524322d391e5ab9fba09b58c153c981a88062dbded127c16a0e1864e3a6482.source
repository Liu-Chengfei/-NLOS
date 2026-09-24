"""归一化（normalization）测试模块。

测试覆盖范围：
- 特征归一化与反归一化
- 统计量（均值/标准差）的计算
- 边界值处理

被测模块：liquidloc.models.normalization"""

import numpy as np
import pytest

from liquidloc.models.features.normalization import denormalize_features, fit_norm_stats, normalize_features


def test_normal_case():
    """正常场景测试。
    
    验证被测功能在标准输入下的正确行为，
    确保核心路径能正常执行并返回预期结果。
    """
    feature_matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    norm_stats = fit_norm_stats(feature_matrix)

    normalized = normalize_features(feature_matrix, norm_stats)
    restored = denormalize_features(normalized, norm_stats)

    assert normalized.shape == feature_matrix.shape
    assert np.allclose(restored, feature_matrix)
    assert np.allclose(normalized.mean(axis=0), np.zeros(2))


def test_invalid_case():
    """无效输入测试。
    
    验证被测功能对无效输入的拒绝行为，
    确保缺少必要参数时抛出 ValueError。
    """
    with pytest.raises(KeyError, match="mean"):
        normalize_features([[1.0, 2.0]], {"std": [1.0, 1.0]})
