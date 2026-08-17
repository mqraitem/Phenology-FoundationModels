# Vendored Presto

This directory contains the model-runtime subset of [nasaharvest/presto](https://github.com/nasaharvest/presto), approximately tracking upstream `main` at commit `ba88a3f` (2025-09-26).

It is included directly rather than as a submodule because the project applies small model changes and does not use Presto's upstream data pipelines, downstream evaluations, deployment tools, or training entry points. Those unused components are omitted from this runtime subset.

## Modifications

- **`presto/presto.py`** — replace the imports of `DynamicWorld2020_2021` and `BANDS_GROUPS_IDX` from `dataops.pipelines.*` with hardcoded inline definitions; comment out an upstream month-range assertion that fires for our DOY encoding.

No model architecture or weights have been changed.

## License

Presto is distributed under the MIT License; see `LICENSE` in this directory. All upstream copyright notices are preserved.
