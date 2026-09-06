"""Unit tests for ekf_init_protocol (P16) and normalization_state (P11)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from liquidloc.dataio.normalization_state import (
    NORMALIZATION_FIELDS,
    FieldStatistics,
    NormalizationState,
    compute_normalization_state,
    compute_z_score_normalization,
    load_im_meta_json,
    validate_no_test_leak,
    write_im_meta_json,
)
from liquidloc.estimators.ekf_init_protocol import (
    INITIAL_POSITION_UNCERTAINTY_M2,
    INITIAL_VELOCITY_UNCERTAINTY_MPS2,
    INITIAL_YAW_UNCERTAINTY_RAD2,
    TRILATERATION_MIN_ANCHORS,
    build_p0_diagonal,
    collect_first_frame_anchor_ranges,
    collect_first_vio_yaw,
    ekf_init_protocol_p16,
    trilateration_least_squares,
)


# =============================================================================
# P16: EKF init protocol
# =============================================================================


class TestTrilaterationLeastSquares:
    """最小二乘三边测量解。"""

    def test_successful_4_anchors(self):
        # Target at (10, 10), anchors K1 (2,2)(18,3)(6,17)(16,18)
        import math
        tx, ty = 10.0, 10.0
        anchors = [(2.0, 2.0), (18.0, 3.0), (6.0, 17.0), (16.0, 18.0)]
        ranges = [math.hypot(tx - a[0], ty - a[1]) for a in anchors]
        result = trilateration_least_squares(ranges, anchors)
        assert result is not None
        assert abs(result[0] - 10.0) < 0.01
        assert abs(result[1] - 10.0) < 0.01

    def test_insufficient_anchors(self):
        result = trilateration_least_squares([1.0, 2.0], [(0, 0), (1, 1)])
        assert result is None

    def test_singular_layout(self):
        # 3 anchors colinear → singular A^T A → returns None
        result = trilateration_least_squares(
            [3.0, 4.0, 5.0],
            [(0, 0), (2, 0), (4, 0)],
        )
        assert result is None


class TestCollectFirstFrameAnchorRanges:
    def test_only_valid_events(self):
        events = [
            {"modality": "uwb", "t": 0.5, "payload": {"anchor_id": 0, "measured_range": 5.0, "valid": True}},
            {"modality": "uwb", "t": 0.5, "payload": {"anchor_id": 1, "measured_range": 6.0, "valid": True}},
            {"modality": "uwb", "t": 0.5, "payload": {"anchor_id": 0, "measured_range": 5.5, "valid": False}},
        ]
        out = collect_first_frame_anchor_ranges(events, {0: (0, 0, 0), 1: (10, 0, 0)})
        assert out == {0: 5.0, 1: 6.0}


class TestCollectFirstVioYaw:
    def test_returns_first_event(self):
        events = [
            {"modality": "vio", "t": 0.3, "payload": {"dx": 0.1, "dy": 0.0, "dyaw": 0.05, "quality": 1.0}},
            {"modality": "vio", "t": 0.5, "payload": {"dx": 0.1, "dy": 0.0, "dyaw": 0.1, "quality": 1.0}},
        ]
        yaw, quality = collect_first_vio_yaw(events)
        assert yaw == pytest.approx(0.05)
        assert quality == pytest.approx(1.0)

    def test_no_events_returns_none(self):
        yaw, quality = collect_first_vio_yaw([])
        assert yaw is None
        assert quality is None


class TestBuildP0Diagonal:
    def test_trilated_first(self):
        diag, mode = build_p0_diagonal(delayed_init=False, initial_position_yaw=True)
        assert mode == "trilated_first"
        assert diag[0] == 0.25  # px var
        assert diag[1] == 0.25  # py var
        assert diag[4] == 0.01  # yaw var

    def test_delayed_init(self):
        diag, mode = build_p0_diagonal(delayed_init=True, initial_position_yaw=False)
        assert mode == "delayed_init_no_pos"
        assert diag[0] == INITIAL_POSITION_UNCERTAINTY_M2  # 100 m²
        assert diag[4] == INITIAL_YAW_UNCERTAINTY_RAD2  # π²

    def test_velocity_uncertainty(self):
        diag, _ = build_p0_diagonal(delayed_init=True, initial_position_yaw=False)
        assert diag[2] == INITIAL_VELOCITY_UNCERTAINTY_MPS2  # vx var
        assert diag[3] == INITIAL_VELOCITY_UNCERTAINTY_MPS2  # vy var


class TestEKFInitProtocolP16:
    def test_successful_trilateration_with_yaw(self):
        anchor_lookup = {
            0: (2.0, 2.0, 2.5),
            1: (18.0, 3.0, 2.5),
            2: (6.0, 17.0, 2.5),
            3: (16.0, 18.0, 2.5),
        }
        import math
        target = (10.0, 10.0)
        ranges = {
            0: math.hypot(8, 8),
            1: math.hypot(8, 7),
            2: math.hypot(4, 7),
            3: math.hypot(6, 8),
        }
        uwb_events = [
            {"modality": "uwb", "t": 0.1, "payload": {"anchor_id": a, "measured_range": r, "valid": True, "quality": 1.0}}
            for a, r in ranges.items()
        ]
        vio_events = [{"modality": "vio", "t": 0.05, "payload": {"dx": 0, "dy": 0, "dyaw": 0.5, "quality": 1.0}}]
        report = ekf_init_protocol_p16(uwb_events, vio_events, anchor_lookup=anchor_lookup)
        assert report["mode"] == "trilated_first"
        assert report["n_anchors_used"] == 4
        assert report["trilaterated_position_xy"] is not None
        assert report["vio_yaw_used"] == pytest.approx(0.5)
        # px, py should be close to (10, 10)
        px, py = report["trilaterated_position_xy"]
        assert abs(px - 10.0) < 0.5
        assert abs(py - 10.0) < 0.5

    def test_delayed_init_when_few_anchors(self):
        anchor_lookup = {0: (0, 0, 0), 1: (10, 0, 0)}
        uwb_events = [
            {"modality": "uwb", "t": 0.1, "payload": {"anchor_id": 0, "measured_range": 5.0, "valid": True, "quality": 1.0}},
            {"modality": "uwb", "t": 0.1, "payload": {"anchor_id": 1, "measured_range": 5.0, "valid": True, "quality": 1.0}},
        ]
        report = ekf_init_protocol_p16(uwb_events, [], anchor_lookup=anchor_lookup)
        assert report["mode"] == "delayed_init_no_pos_no_yaw"
        assert report["trilaterated_position_xy"] is None
        assert report["init_cov_diagonal"][0] == INITIAL_POSITION_UNCERTAINTY_M2
        assert report["init_cov_diagonal"][4] == INITIAL_YAW_UNCERTAINTY_RAD2

    def test_no_yaw_no_vio(self):
        anchor_lookup = {0: (0, 0, 0), 1: (10, 0, 0), 2: (5, 10, 0)}
        uwb_events = [
            {"modality": "uwb", "t": 0.1, "payload": {"anchor_id": a, "measured_range": 5.0, "valid": True, "quality": 1.0}}
            for a in [0, 1, 2]
        ]
        report = ekf_init_protocol_p16(uwb_events, [], anchor_lookup=anchor_lookup)
        assert report["mode"] == "trilated_init_no_yaw"
        assert report["trilaterated_position_xy"] is not None
        assert report["init_cov_diagonal"][4] == INITIAL_YAW_UNCERTAINTY_RAD2  # 大 yaw P0


class TestEKFCoreWiring:
    """EKFCore.apply_p16_init_from_first_frame 集成测试。"""

    def test_wired_to_ekfcore(self):
        from liquidloc.estimators.ekf_core import EKFCore

        ekf = EKFCore(init_cfg={
            "anchor_layout": {
                "anchor_ids": [0, 1, 2, 3],
                "anchor_positions": [(2.0, 2.0), (18.0, 3.0), (6.0, 17.0), (16.0, 18.0)],
            }
        })
        uwb_events = [
            {"modality": "uwb", "t": 0.1, "payload": {"anchor_id": i, "measured_range": 5.0, "valid": True, "quality": 1.0}}
            for i in range(4)
        ]
        vio_events = [{"modality": "vio", "t": 0.05, "payload": {"dx": 0, "dy": 0, "dyaw": 0.3, "quality": 1.0}}]
        report = ekf.apply_p16_init_from_first_frame(uwb_events, vio_events)
        assert report["mode"] == "trilated_first"
        # EKF state should have trilated position
        assert ekf._state["px"] != 0.0 or ekf._state["py"] != 0.0
        # EKF P0 should be tight (small) for position
        assert ekf._covariance[0, 0] == 0.25


# =============================================================================
# P11: Normalization state (im_meta.json)
# =============================================================================


class TestComputeNormalizationState:
    def test_basic_stats(self):
        events = [
            {"modality": "uwb", "t": i, "split_id": "train_001",
             "payload": {"anchor_id": 0, "measured_range": float(i), "valid": True, "quality": 1.0}}
            for i in range(10)
        ]
        state = compute_normalization_state(events, dataset_name="sim_e9")
        assert state.n_train_events == 10
        assert state.n_train_splits == 1
        assert state.by_modality_field["uwb.measured_range"].count == 10
        assert state.by_modality_field["uwb.measured_range"].mean == pytest.approx(4.5)

    def test_non_finite_counting(self):
        events = [
            {"modality": "uwb", "t": 0.0, "payload": {"anchor_id": 0, "measured_range": float("nan"), "valid": True}},
            {"modality": "uwb", "t": 0.0, "payload": {"anchor_id": 0, "measured_range": 5.0, "valid": True}},
        ]
        state = compute_normalization_state(events)
        stats = state.by_modality_field["uwb.measured_range"]
        assert stats.count == 1
        assert stats.non_finite_count == 1

    def test_modality_filter(self):
        events = [
            {"modality": "uwb", "t": 0.0, "payload": {"anchor_id": 0, "measured_range": 5.0, "valid": True}},
            {"modality": "imu", "t": 0.0, "payload": {"ax": 0.1, "ay": 0.0, "gz": 0.0}},
            {"modality": "vio", "t": 0.0, "payload": {"dx": 0.1, "dy": 0.0, "dyaw": 0.0}},
        ]
        state = compute_normalization_state(events)
        assert state.by_modality_field["uwb.measured_range"].count == 1
        assert state.by_modality_field["imu.ax"].count == 1
        assert state.by_modality_field["vio.dx"].count == 1


class TestValidateNoTestLeak:
    def test_no_leak_when_only_train(self):
        events = [{"modality": "uwb", "t": 0.0, "split_id": "train_001", "payload": {"measured_range": 5.0}}]
        n_train, n_suspect = validate_no_test_leak(events, allowed_split_ids={"train_001"})
        assert n_train == 1
        assert n_suspect == 0

    def test_suspect_when_test_included(self):
        events = [
            {"modality": "uwb", "t": 0.0, "split_id": "train_001", "payload": {"measured_range": 5.0}},
            {"modality": "uwb", "t": 0.0, "split_id": "test_001", "payload": {"measured_range": 8.0}},
        ]
        n_train, n_suspect = validate_no_test_leak(events, allowed_split_ids={"train_001"})
        assert n_train == 1
        assert n_suspect == 1


class TestWriteLoadImMetaJson:
    def test_roundtrip(self):
        events = [
            {"modality": "uwb", "t": 0.0, "split_id": "train_001",
             "payload": {"anchor_id": 0, "measured_range": float(i), "valid": True}}
            for i in range(20)
        ]
        state = compute_normalization_state(events, dataset_name="sim_e9_test")
        with tempfile.TemporaryDirectory() as tmp:
            path = write_im_meta_json(state, tmp)
            loaded = load_im_meta_json(path)
            assert loaded.dataset_name == "sim_e9_test"
            assert loaded.n_train_events == 20
            assert loaded.by_modality_field["uwb.measured_range"].mean == state.by_modality_field["uwb.measured_range"].mean


class TestComputeZScoreNormalization:
    def test_z_score_params(self):
        state = NormalizationState(
            n_train_events=10,
            n_train_splits=1,
            train_split_ids=["train"],
            dataset_name="sim",
            by_modality_field={
                "uwb.measured_range": FieldStatistics(count=10, mean=5.0, std=2.0, min=1.0, max=9.0),
            },
        )
        params = compute_z_score_normalization(state)
        assert params["uwb.measured_range"] == pytest.approx((5.0, 2.0))

    def test_zero_std_fallback(self):
        state = NormalizationState(
            by_modality_field={
                "imu.ax": FieldStatistics(count=5, mean=0.1, std=0.0, min=0.1, max=0.1),
            },
        )
        params = compute_z_score_normalization(state)
        # std=0 → fallback to epsilon=1e-6
        assert params["imu.ax"][1] >= 1e-6