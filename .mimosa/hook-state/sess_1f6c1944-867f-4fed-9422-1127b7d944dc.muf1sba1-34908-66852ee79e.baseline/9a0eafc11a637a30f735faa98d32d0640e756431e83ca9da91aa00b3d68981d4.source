from __future__ import annotations

"""状态定义（state_definition）测试模块。

测试覆盖范围：
- 状态向量的维度与字段映射
- 状态初始化与默认值
- 状态转换的正确性

被测模块：liquidloc.estimators.state_definition"""

import os
import tempfile

import pytest

from liquidloc.common.config_utils import find_project_root
from liquidloc.estimators import state_definition as sd

_PROJECT_ROOT = find_project_root()


def _build_task_yaml(*, state_items: str = "[px, py, vx, vy, yaw, bax, bay, bg, uwb_clock_bias, vio_scale]") -> str:
    return "\n".join(
        [
            "task_name: uwb_imu_vio_localization",
            "state_definition:",
            "  frame: planar_xy_yaw",
            f"  state_items: {state_items}",
            "sensor_roles:",
            "  uwb: absolute_range_constraint",
            "  imu: high_rate_propagation",
            "  vio: relative_pose_constraint",
            "vio_update_contract:",
            "  measurement_items: [dx, dy, dyaw]",
            "  updated_state_items: [px, py, yaw, uwb_clock_bias, vio_scale]",
            "  learned_control_entry: noise_multiplier",
            "  forbid_rewrite_measurement_fields_inside_update: true",
        ]
    )


def _write_task_yaml(content: str) -> str:
    tmp_dir = _PROJECT_ROOT / "outputs" / "_test_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fd, path = tempfile.mkstemp(suffix=".yaml", dir=str(tmp_dir))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
    except Exception:
        os.close(fd)
        raise
    return path


def _cleanup(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


class TestStateItems:
    def test_tuple_type(self):
        assert isinstance(sd.state_items, tuple)

    def test_length(self):
        assert len(sd.state_items) == 10

    def test_first_item_px(self):
        assert sd.state_items[0] == "px"

    def test_last_item_vio_scale(self):
        assert sd.state_items[-1] == "vio_scale"

    def test_all_strings(self):
        assert all(isinstance(item, str) for item in sd.state_items)

    def test_no_duplicates(self):
        """重复测试：no。\n\n验证 no 对重复输入的处理，\n确保重复项被正确检测。
        """
        assert len(sd.state_items) == len(set(sd.state_items))

    def test_required_keys_present(self):
        for key in ("px", "py", "vx", "vy", "yaw"):
            assert key in sd.state_items

    def test_order(self):
        assert sd.state_items == ("px", "py", "vx", "vy", "yaw", "bax", "bay", "bg", "uwb_clock_bias", "vio_scale")


class TestStateIndexMap:
    def test_mapping_values(self):
        expected = {"px": 0, "py": 1, "vx": 2, "vy": 3, "yaw": 4, "bax": 5, "bay": 6, "bg": 7, "uwb_clock_bias": 8, "vio_scale": 9}
        assert dict(sd.state_index_map) == expected

    def test_frozen_readonly(self):
        with pytest.raises(TypeError):
            sd.state_index_map["vio_scale"] = 0  # type: ignore[misc]

    def test_last_index(self):
        assert sd.state_index_map["vio_scale"] == sd.state_dim - 1

    def test_first_index(self):
        assert sd.state_index_map["px"] == 0

    def test_index_contiguous(self):
        indices = sorted(sd.state_index_map.values())
        assert indices == list(range(len(sd.state_items)))


class TestDimensions:
    def test_state_dim(self):
        assert sd.state_dim == 10

    def test_cov_dim_equals_state_dim(self):
        assert sd.cov_dim == sd.state_dim

    def test_noise_dim_equals_state_dim(self):
        assert sd.noise_dim == sd.state_dim

    def test_state_dim_equals_len_state_items(self):
        assert sd.state_dim == len(sd.state_items)


class TestGetStateIndexMap:
    def test_returns_dict(self):
        result = sd.get_state_index_map()
        assert isinstance(result, dict)

    def test_returns_correct_mapping(self):
        """映射报告测试：returns correct。\n\n验证 returns correct 的映射报告生成，\n确保锚点布局信息被正确记录。
        """
        result = sd.get_state_index_map()
        expected = {"px": 0, "py": 1, "vx": 2, "vy": 3, "yaw": 4, "bax": 5, "bay": 6, "bg": 7, "uwb_clock_bias": 8, "vio_scale": 9}
        assert result == expected

    def test_returns_new_dict_each_call(self):
        r1 = sd.get_state_index_map()
        r2 = sd.get_state_index_map()
        assert r1 is not r2

    def test_return_type_annotation(self):
        ann = sd.get_state_index_map.__annotations__.get("return")
        assert ann == "dict[str, int]" or ann == dict[str, int]


class TestLoadConfigStateItems:
    def _patch_and_call(self, content: str, monkeypatch):
        path = _write_task_yaml(content)
        try:
            monkeypatch.setattr(sd, "_TASK_CONFIG_PATH", type(sd._TASK_CONFIG_PATH)(path))
            return sd._load_config_state_items()
        finally:
            _cleanup(path)

    def _patch_and_expect_error(self, content: str, error_type, match: str, monkeypatch):
        path = _write_task_yaml(content)
        try:
            monkeypatch.setattr(sd, "_TASK_CONFIG_PATH", type(sd._TASK_CONFIG_PATH)(path))
            with pytest.raises(error_type, match=match):
                sd._load_config_state_items()
        finally:
            _cleanup(path)

    def test_valid_config(self, monkeypatch):
        result = self._patch_and_call(_build_task_yaml(), monkeypatch)
        assert result == sd.state_items

    def test_missing_config_file(self, monkeypatch):
        monkeypatch.setattr(sd, "_TASK_CONFIG_PATH", type(sd._TASK_CONFIG_PATH)(_PROJECT_ROOT / "nonexistent" / "path.yaml"))
        with pytest.raises((FileNotFoundError, ValueError)):
            sd._load_config_state_items()

    def test_missing_state_definition_block(self, monkeypatch):
        content = "\n".join(
            [
                "task_name: uwb_imu_vio_localization",
                "sensor_roles:",
                "  uwb: absolute_range_constraint",
                "  imu: high_rate_propagation",
                "  vio: relative_pose_constraint",
                "vio_update_contract:",
                "  measurement_items: [dx, dy, dyaw]",
                "  updated_state_items: [px, py, yaw]",
                "  learned_control_entry: noise_multiplier",
                "  forbid_rewrite_measurement_fields_inside_update: true",
            ]
        )
        self._patch_and_expect_error(content, TypeError, "state_definition must be a mapping", monkeypatch)

    def test_missing_state_items_key(self, monkeypatch):
        content = "\n".join(
            [
                "task_name: uwb_imu_vio_localization",
                "state_definition:",
                "  frame: planar_xy_yaw",
                "sensor_roles:",
                "  uwb: absolute_range_constraint",
                "  imu: high_rate_propagation",
                "  vio: relative_pose_constraint",
                "vio_update_contract:",
                "  measurement_items: [dx, dy, dyaw]",
                "  updated_state_items: [px, py, yaw]",
                "  learned_control_entry: noise_multiplier",
                "  forbid_rewrite_measurement_fields_inside_update: true",
            ]
        )
        self._patch_and_expect_error(content, TypeError, "state_definition.state_items must be a non-empty list", monkeypatch)

    def test_non_inline_list_rejected(self, monkeypatch):
        """拒绝测试：non inline list。\n\n验证被测功能对不合法的 non inline list 输入正确抛出异常，\n防止无效参数通过验证。
        """
        content = _build_task_yaml(state_items="px, py, vx")
        self._patch_and_expect_error(content, TypeError, "state_definition.state_items must be a non-empty list", monkeypatch)

    def test_empty_list_rejected(self, monkeypatch):
        """拒绝测试：empty list。\n\n验证被测功能对不合法的 empty list 输入正确抛出异常，\n防止无效参数通过验证。
        """
        content = _build_task_yaml(state_items="[]")
        self._patch_and_expect_error(content, TypeError, "state_definition.state_items must be a non-empty list", monkeypatch)

    def test_duplicate_state_items_rejected(self, monkeypatch):
        """拒绝测试：duplicate state items。\n\n验证被测功能对不合法的 duplicate state items 输入正确抛出异常，\n防止无效参数通过验证。
        """
        content = _build_task_yaml(state_items="[px, py, vx, vy, yaw, bax, bay, px]")
        self._patch_and_expect_error(content, ValueError, "duplicate entries", monkeypatch)

    def test_whitespace_in_items(self, monkeypatch):
        content = _build_task_yaml(state_items="[ px , py , vx , vy , yaw , bax , bay , bg , uwb_clock_bias , vio_scale ]")
        result = self._patch_and_call(content, monkeypatch)
        assert result == sd.state_items


class TestValidateUpstreamContracts:
    _ORIGINAL_PATH = sd._TASK_CONFIG_PATH

    def _patch_and_expect_error(self, content: str, error_type, match: str, monkeypatch):
        path = _write_task_yaml(content)
        try:
            monkeypatch.setattr(sd, "_TASK_CONFIG_PATH", type(sd._TASK_CONFIG_PATH)(path))
            with pytest.raises(error_type, match=match):
                sd._validate_upstream_contracts()
        finally:
            _cleanup(path)

    def test_valid_contracts(self, monkeypatch):
        """合同测试：valid。\n\n验证 valid 的接口合同，\n确保输入输出符合协议约定。
        """
        monkeypatch.setattr(sd, "_TASK_CONFIG_PATH", self._ORIGINAL_PATH)
        sd._validate_upstream_contracts()

    def test_config_mismatch_rejected(self, monkeypatch):
        """拒绝测试：config mismatch。\n\n验证被测功能对不合法的 config mismatch 输入正确抛出异常，\n防止无效参数通过验证。
        """
        content = _build_task_yaml(state_items="[px, py, vx]")
        self._patch_and_expect_error(content, ValueError, "state_definition.state_items must be", monkeypatch)

    def test_duplicate_config_items_rejected(self, monkeypatch):
        """拒绝测试：duplicate config items。\n\n验证被测功能对不合法的 duplicate config items 输入正确抛出异常，\n防止无效参数通过验证。
        """
        content = _build_task_yaml(state_items="[px, py, vx, vy, yaw, bax, bay, px]")
        self._patch_and_expect_error(content, ValueError, "duplicate entries", monkeypatch)

    def test_missing_required_keys_rejected(self, monkeypatch):
        """拒绝测试：missing required keys。\n\n验证被测功能对不合法的 missing required keys 输入正确抛出异常，\n防止无效参数通过验证。
        """
        content = _build_task_yaml(state_items="[px, py, bax, bay, bg]")
        self._patch_and_expect_error(content, ValueError, "state_definition.state_items must be", monkeypatch)
