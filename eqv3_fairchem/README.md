# eqv3_fairchem

The fairchem v1 fork that loads the fine-tuned EquiformerV3 checkpoint in
`ft_ckpts/eqv3/`. Copied on 2026-10-28 from the working tree of
`ConstrDiff_Material/models/equiformer_v3` (fork of fairchem, commit `0c2fd97`
plus uncommitted edits to `equiformer_v3.py`, `so3.py`, `transformer_block.py`
and `core/common/utils.py`). Only `src/fairchem/core` and the EqV3 model and
trainer code are kept.

Install into the `fairchemv1` env from the relaxgen root:

```bash
pip install -e eqv3_fairchem --no-deps
```

On the UVA cluster the env still uses the ConstrDiff_Material submodule.
