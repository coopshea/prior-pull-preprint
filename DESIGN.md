# Prior Pull: replay study

Status: draft, 2026-11-08. No confirmatory results exist. Earlier plans are kept in `SPEC.md` and `DESIGN_survey_archive.md`.

## Question

When a scientist asks an LLM what to try next, how far do the data, the names of things, and the wording of the request each move the suggestion, compared with simply asking again?

This measures model behavior on measured chemistry and process grids. It does not measure what people do, and does not cover whole searches.

## Rules

1. Only inputs are designed. Worlds and seeds are not chosen for how they behave.
2. Every value standing for a real system comes from a cited source, recorded in `data/SOURCES.md`.
3. Once frozen, only a code bug can trigger a rerun, and each one is logged. All results are reported, including nulls.

## Worlds

Measured grids only: `shields`, `suzuki`, `efb`, and 29 `cernak` plate-temperature blocks, under the inclusion rule in `DESIGN_survey_archive.md`. Each of the four sources contributes equally.

## Unit of analysis

One unit is a world, a random history of 0, 3, 6, or 12 past results, and one prompt from the pool. Each unit is sent once per arm, with Opus 5.5 at default effort, as a single stateless request for the next 3 experiments.

| Arm | Change from the reference |
|---|---|
| A1 | Reference: real names, true data, no hint, no advice, rationale requested |
| A2 | Nothing. The noise floor |
| Scrambled | Results replaced by a fixed scramble of the grid (not at 0 results) |
| Coded | Settings and options shown as codes |
| Swapped | Real names randomly permuted among options |
| Prompt | A different prompt from the pool |
| Hunch | A hunch on a randomly chosen setting and option |
| Evidence, audit, caution | One asking strategy in one of 5 wordings |
| No rationale | The reason is not requested |

Every arm is compared with A1 of the same unit.

## Measures

- Change in picks: total-variation distance between the option distributions of two sets of picks, averaged over settings.
- True rank: mean grid percentile of the picks on the real values.
- Follows results: share of setting choices matching the option with the best mean in the results shown.
- Follows prior: mean zero-shot score of the names picked, from the model's own ranking of real names, with no results shown (3 repetitions per world). Listed options only, since long lists often come back incomplete. Undefined for coded names.
- Source guess: whether the model names the source of each grid, with a fabricated decoy (3 repetitions).

## Figures

- Figure 1: the design.
- Figure 2: change in picks for each arm, with the noise floor marked.
- Figure 3: change in picks against the number of results shown, for data, priors, wording, and noise.
- Figure 4: for each asking strategy and no rationale, the paired difference from the reference in true rank, follows results, and follows prior.

Intervals are bootstrapped over units within source, and sources are weighted equally. Each estimate is also reported per source.

## Preliminary check, 2026-11-08

16 units on Opus 5.5, cost $6.45 including rankings and source guesses. It found no failures in 172 decisions. Prompts were then cosmetically corrected, so these replies are not pooled with the main run.

## Freeze checklist

- [x] Worlds and pools
- [x] Unit table: 10 units per source and history size, 160 units, 1,720 decisions, seed 0, digest a11bcf1f9e66b7ef02db2c2c75928d4940e0c96c12b49aa7577dc40917b19463
- [x] Pools digest 175d4bbf3b42b030f28dcb952a5092d8a56c843a067934994d600414ca29474c
- [x] Spend cap $30, approved 2026-11-08, Batch API, output in `runs/core`

## Run record, 2026-11-08

Completed: 1,720 decisions, none unusable, $23.06. Deviation noted after the run: units cycle through Cernak blocks in order, so the first 10 of 29 blocks were used at this sample size. Figures are in `runs/core`, and all numbers in the paper come from `scripts/update_paper.py`. Zero-shot rankings were unparseable for 7 Cernak worlds, none of them among the 10 used, because their replies contained unescaped quotes in ligand names.
