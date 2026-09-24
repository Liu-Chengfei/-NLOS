from __future__ import annotations

"""配置工具（config_utils）测试模块。

文件职责：验证 YAML 配置加载、合并和 TBD 路径收集功能。

测试覆盖范围：
- 正常场景：加载、合并、收集 TBD 路径
- 非映射 payload 拒绝
- 假值标量 payload 拒绝
- 零标量 payload 拒绝
- 空文档返回空字典
- 显式 null 顶层拒绝
- 波浪号 null 顶层拒绝
- 格式错误的内联序列回退解析

被测模块：liquidloc.common.config_utils"""


import pytest

from liquidloc.common.config_utils import (
    collect_tbd_paths,
    load_yaml_config,
    merge_configs,
)


def test_config_utils_load_merge_and_collect(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "seed: 7",
                "nested:",
                "  enabled: true",
                "  values: [1, 2, 3]",
                "paths:",
                "  checkpoint: TBD",
            ]
        ),
        encoding="utf-8",
    )

    payload = load_yaml_config(config_path)
    merged = merge_configs(
        [
            payload,
            {
                "nested": {
                    "enabled": False,
                    "extra": "ready",
                }
            },
        ]
    )

    assert payload["seed"] == 7
    assert payload["nested"]["values"] == [1, 2, 3]
    assert merged["nested"] == {
        "enabled": False,
        "values": [1, 2, 3],
        "extra": "ready",
    }
    assert collect_tbd_paths(payload) == ["paths.checkpoint"]


def test_load_yaml_config_rejects_non_mapping_payload(tmp_path):
    config_path = tmp_path / "not_mapping.yaml"
    config_path.write_text("- first\n- second\n", encoding="utf-8")

    with pytest.raises(TypeError, match="Top-level YAML payload must be a mapping"):
        load_yaml_config(config_path)


def test_load_yaml_config_rejects_falsey_scalar_payload(tmp_path):
    config_path = tmp_path / "scalar.yaml"
    config_path.write_text("false\n", encoding="utf-8")

    with pytest.raises(TypeError, match="Top-level YAML payload must be a mapping"):
        load_yaml_config(config_path)


def test_load_yaml_config_rejects_zero_scalar_payload(tmp_path):
    config_path = tmp_path / "zero.yaml"
    config_path.write_text("0\n", encoding="utf-8")

    with pytest.raises(TypeError, match="Top-level YAML payload must be a mapping"):
        load_yaml_config(config_path)


def test_load_yaml_config_keeps_empty_document_as_empty_dict(tmp_path):
    config_path = tmp_path / "empty.yaml"
    config_path.write_text("", encoding="utf-8")

    payload = load_yaml_config(config_path)
    assert payload == {}


def test_load_yaml_config_rejects_explicit_null_top_level_payload(tmp_path):
    config_path = tmp_path / "null.yaml"
    config_path.write_text("null\n", encoding="utf-8")

    with pytest.raises(TypeError, match="Top-level YAML payload must be a mapping"):
        load_yaml_config(config_path)


def test_load_yaml_config_rejects_tilde_null_top_level_payload(tmp_path):
    config_path = tmp_path / "tilde_null.yaml"
    config_path.write_text("~\n", encoding="utf-8")

    with pytest.raises(TypeError, match="Top-level YAML payload must be a mapping"):
        load_yaml_config(config_path)


def test_load_yaml_config_fallback_parses_malformed_inline_sequence_as_string(tmp_path, monkeypatch):
    import liquidloc.common.config_utils as config_utils

    monkeypatch.setattr(config_utils, "_yaml", None)
    config_path = tmp_path / "bad_inline.yaml"
    config_path.write_text("items: [1, 2\n", encoding="utf-8")

    payload = load_yaml_config(config_path)
    assert isinstance(payload, dict)
    assert "items" in payload
