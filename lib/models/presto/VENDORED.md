# Vendored Presto

This directory is a vendored copy of [nasaharvest/presto](https://github.com/nasaharvest/presto), approximately tracking upstream `main` at commit `ba88a3f` (2025-09-26).

It is included directly in this repository (rather than as a git submodule) because we apply local modifications to make Presto importable without the `earthengine-api` / `openmapflow` dependencies, which are required by the upstream `dataops.pipelines` modules but not needed for inference here.

## Modifications

- **`presto/__init__.py`** — defer importing `construct_single_presto_input` (it pulls in `dataops.utils` at module load).
- **`presto/presto.py`** — replace the imports of `DynamicWorld2020_2021` and `BANDS_GROUPS_IDX` from `dataops.pipelines.*` with hardcoded inline definitions; comment out an upstream month-range assertion that fires for our DOY encoding.

No model architecture or weights have been changed.

## License

Presto is distributed under the MIT License; see `LICENSE` in this directory. All upstream copyright notices are preserved.
