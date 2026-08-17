"""Load typed project settings from JSON configuration."""

import json
import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
_CONFIG: dict[str, Any] | None = None


def get_config_path() -> Path:
    """Return the configured JSON path or the repository-local default."""
    configured = os.environ.get("PHENOLOGY_CONFIG")
    if configured:
        return Path(configured).expanduser().resolve()
    return REPO_ROOT / "config.json"


def load_config(force_reload: bool = False) -> dict[str, Any]:
    """Load and cache the project configuration."""
    global _CONFIG
    if _CONFIG is not None and not force_reload:
        return _CONFIG

    config_path = get_config_path()
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration not found at {config_path}. Copy config.example.json "
            "to config.json or set PHENOLOGY_CONFIG."
        )

    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration root must be a JSON object: {config_path}")

    _CONFIG = config
    return config


def get_value(key: str, default: Any = None) -> Any:
    """Read a value using a dotted JSON key such as ``evaluation.stride``."""
    value: Any = load_config()
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            if default is not None:
                return default
            raise KeyError(f"Missing configuration key: {key}")
        value = value[part]
    return value


def get_path(key: str) -> str:
    """Read a path setting and resolve relative paths from the repository root."""
    path = Path(get_value(key)).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return str(path)


def get_mean_stds_dir() -> str:
    return get_path("storage.mean_stds")


def get_data_paths_dir() -> str:
    return get_path("storage.data_paths")


def get_data_hls_composites() -> str:
    return get_path("data.hls_composites")


def get_data_lsp_ancillary() -> str:
    return get_path("data.lsp_ancillary")


def get_data_geojson() -> str:
    return get_path("data.geojson")


def get_checkpoint_root() -> str:
    return get_path("storage.checkpoints")


def get_pixels_cache_dir() -> str:
    return get_path("storage.pixel_cache")


def get_qual_cache_dir() -> str:
    return get_path("storage.qualitative_cache")


def get_qual_tile_ids() -> list[tuple[str, str]]:
    tiles = get_value("evaluation.qualitative_tiles", [])
    return [(tile["site_id"], tile["hls_tile"]) for tile in tiles]


def get_wandb_project() -> str:
    return str(get_value("wandb.project", "phenology_paper_2"))


def get_eval_stride() -> int:
    return int(get_value("evaluation.stride", 2))


def get_eval_batch_size() -> int:
    return int(get_value("evaluation.batch_size", 64))


def get_model_weights(model_size: str) -> str:
    return get_path(f"model_weights.{model_size.lower()}")


if __name__ == "__main__":
    print(json.dumps(load_config(), indent=2))
