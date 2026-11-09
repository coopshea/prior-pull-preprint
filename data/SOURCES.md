# Data sources

| World | Source | License | Retrieved |
|---|---|---|---|
| shields | Shields et al., Nature 2021, via github.com/b-shields/edbo `experiments/data/direct_arylation` | MIT | 2026-11-08 |
| suzuki | github.com/b-shields/edbo `experiments/data/suzuki`. Original source unconfirmed; it may derive from Perera et al., Science 2018. Only SMILES are distributed there, so SMILES are used as the labels | MIT | 2026-11-08 |
| efb | Zenodo 23953649, delignification of oil palm empty fruit bunches, posted 2026-11-04 | CC-BY-4.0 | 2026-11-08 |
| cernak_* | Open Reaction Database ord_dataset-805ad863feef48579d95d86a728035f4; Cernak group, JACS, doi 10.1021/jacs.6c05959. Decoded by scripts/decode_cernak.py | CC-BY-SA-4.0 | 2026-11-08 |

Inclusion rule (DESIGN.md section 3). Excluded after applying it: EDBO aryl amination, 3,955 of 3,960 cells, so incomplete; Cernak blocks JD-03-10 and XZ-01-55, duplicate level names within the block; Iron Mind lookup tables, not distributed in the repositories checked.

| evml | Human versus BO game on the Shields grid, github.com/b-shields/EvML `results/`, 50 players, 100 experiments each, batches of 5 | MIT | 2026-11-08 |
