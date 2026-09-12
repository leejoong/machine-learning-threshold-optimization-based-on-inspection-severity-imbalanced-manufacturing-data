# Data layout

Raw datasets are intentionally excluded from Git because they are large and
may have their own redistribution terms. Keep them outside the repository.

The local layout used for this project is:

```text
D:/졸업논문 데이터/
├── 논문 공개 코드/       # this repository
├── cnc_milling_tool_life_2025/
├── rsw_gun/
└── wm811k/
```

For another machine, set the data root before running an experiment:

```powershell
$env:THESIS_DATA_ROOT = 'D:/path/to/datasets'
python -m experiments.run_cnc_milling_tool_life
```

The public reproduction package uses only these three datasets:

- `cnc_milling_tool_life_2025/`
- `rsw_gun/`
- `wm811k/`

Other datasets that may exist elsewhere on the D drive are not part of this
public project. Generated caches, trained models, and experiment outputs are written to
`outputs/` and are ignored by Git.
