"""Fast contract checks for the repository YAML configuration surface."""

from __future__ import annotations

from pathlib import Path

import pytest

from liquidloc.common.config_utils import collect_tbd_paths, load_yaml_config


CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"
CONFIG_PATHS = tuple(sorted(CONFIG_ROOT.rglob("*.yaml")))


@pytest.mark.parametrize(
    "config_path",
    CONFIG_PATHS,
    ids=lambda path: path.relative_to(CONFIG_ROOT).as_posix(),
)
def test_repository_yaml_has_valid_top_level_shape(config_path: Path):
    payload = load_yaml_config(config_path)

    assert isinstance(payload, dict), config_path
    assert collect_tbd_paths(payload) == [], config_path


@pytest.mark.parametrize(
    ("relative_dir", "required_key", "exceptions"),
    [
        ("datasets", "dataset_name", {"public_dataset_registry.yaml"}),
        ("experiments", "experiment_id", set()),
        ("models", "name", set()),
        ("sweeps", "levels", set()),
    ],
)
def test_configuration_family_has_stable_identity_field(
    relative_dir: str,
    required_key: str,
    exceptions: set[str],
):
    family_paths = tuple(sorted((CONFIG_ROOT / relative_dir).glob("*.yaml")))
    assert family_paths, f"expected configuration files under {relative_dir}"

    for config_path in family_paths:
        if config_path.name in exceptions:
            continue
        payload = load_yaml_config(config_path)
        assert required_key in payload, config_path
