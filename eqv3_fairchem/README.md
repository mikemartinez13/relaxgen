# eqv3_fairchem

The fairchem v1 fork that loads the fine-tuned EquiformerV3 checkpoint in
`ft_ckpts/eqv3/`. Copied on 2026-10-28 from the working tree of
`ConstrDiff_Material/models/equiformer_v3` (fork of fairchem, commit `0c2fd97`
plus uncommitted edits to `equiformer_v3.py`, `so3.py`, `transformer_block.py`
and `core/common/utils.py`). Only `src/fairchem/core` and the EqV3 model and
trainer code are kept, and from `core` the models EqV3 does not use (eSCN, eSEN,
GemNet variants, PaiNN, DimeNet++, SchNet), the atom-embedding tables, scripts and
the hydra CLI are removed. EquiformerV2 stays because the EqV3 trainer imports
its trainers, and SCN stays because EquiformerV2 imports its smearing module.
fairchem registers models by importing whatever files exist, so nothing else
was edited.

Install into the `fairchemv1` env from the relaxgen root:

```bash
pip install -e eqv3_fairchem --no-deps
```

On the UVA cluster the env still uses the ConstrDiff_Material submodule.
