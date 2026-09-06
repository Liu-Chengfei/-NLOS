import pytest
from liquidloc.scenarios.visual_levels import apply_visual_level


def _events():
    return [
        {"t": 0.0, "dt": 0.0, "modality": "imu", "meta": {"scene_id": "S(A1,N0,V1,K0,M0)", "seq_id": "mini_seq"}, "imu_payload": {"ax": 0.0, "ay": 0.0, "gz": 0.0}, "uwb_payload": None, "vio_payload": None},
        {"t": 0.1, "dt": 0.1, "modality": "vio", "meta": {"scene_id": "S(A1,N0,V1,K0,M0)", "seq_id": "mini_seq"}, "imu_payload": None, "uwb_payload": None, "vio_payload": {"dx": 0.5, "dy": -0.2, "dyaw": 0.1, "quality": 0.9, "tracked_features": 150, "reproj_err": 1.4}},
        {"t": 0.2, "dt": 0.1, "modality": "vio", "meta": {"scene_id": "S(A1,N0,V1,K0,M0)", "seq_id": "mini_seq"}, "imu_payload": None, "uwb_payload": None, "vio_payload": {"dx": 0.1, "dy": 0.0, "dyaw": 0.05, "quality": 0.8, "tracked_features": 50, "reproj_err": 0.2}},
        {"t": 0.3, "dt": 0.1, "modality": "vio", "meta": {"scene_id": "S(A1,N0,V1,K0,M0)", "seq_id": "mini_seq"}, "imu_payload": None, "uwb_payload": None, "vio_payload": {"dx": -0.2, "dy": 0.3, "dyaw": -0.04, "quality": 0.7, "tracked_features": 80, "reproj_err": 5.0}},
    ]


def test_dump_values(capsys):
    visual_cfg = {"V1": {"tracked_features_range": [60, 120], "reproj_err_max": 1.0, "blackout_prob": 0.0, "drift_bias_m": 0.05}}
    degraded, report = apply_visual_level(_events(), "V1", visual_cfg)
    vio = [e["vio_payload"] for e in degraded if e["modality"] == "vio"]
    for i, p in enumerate(vio):
        print(f"VIO {i}: dx={p['dx']}, dy={p['dy']}, dyaw={p['dyaw']}, quality={p['quality']}, tracked={p['tracked_features']}, reproj={p['reproj_err']}")
    print(f"consistency_checks: {report['consistency_checks']}")
    print(f"drift_bias_m: {report['drift_bias_m']}")
    print(f"drift_transform_plan:")
    for plan in report["drift_transform_plan"]:
        print(f"  {plan}")
    print(f"quality_plan:")
    for plan in report["quality_plan"]:
        print(f"  {plan}")
    print(f"reproj_err_plan:")
    for plan in report["reproj_err_plan"]:
        print(f"  {plan}")
