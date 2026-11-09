# Prior Pull: initial input survey

Status: draft for sign-off, 2026-11-08. Signing off freezes sections 2 to 8. This document replaces SPEC.md as the plan. No results from this design exist yet.

## 1. Purpose

Find out how varied inputs change the experiments an LLM suggests, across many search problems. This first run tests whether the setup produces usable evidence, measures variability, and identifies candidate rules of thumb to examine at larger scale.

It makes no directional prediction. It measures model behavior, not the behavior of people using the model.

## 2. Rules

1. Only the inputs of the run are designed. Worlds, seeds, and labels are not chosen for how they behave.
2. No world is tuned, filtered, or dropped because of its landscape or any player's performance. Corners, needles, and flat surfaces stay in.
3. Seeds are consecutive integers. If results vary widely, the response is more instances or seeds.
4. Assignments come from a seeded sampler, not by hand.
5. Once frozen, only a code bug can trigger a rerun. Each bug, its effect, and the affected trajectories are logged. Surprising results stay in.
6. Every estimate in section 7 is reported, including nulls.
7. Every value or functional form that stands for a real system comes from a cited source, recorded with its provenance in the world file. Nothing comes from the author's or a model's memory. A world family that cannot be fully sourced is left out.

## 3. Worlds

Inclusion rule for real data: every complete measured grid that has at least 3 factors, at least 50 cells, an open license, and unique level names, and that we can download before the freeze. Sources checked: the EDBO repository, the Open Reaction Database's listing, the Iron Mind repositories, and Zenodo searches. See `data/SOURCES.md`.

| Source | Worlds | Construction |
|---|---|---|
| `shields` | 1 | 1,728 measured yields |
| `suzuki` | 1 | 3,696 measured yields. Labels are SMILES, as distributed |
| `efb` | 1 | 64 conditions in duplicate, returned in turn |
| `cernak` | 29 | Each plate and temperature is one grid of 48 ligands (or as listed) × 8 catalysts × 4 bases. Values as published, including those above 100 and blocks with all zeros |
| `surface` | 30 | Sum of Gaussians, presented with coded labels or an openly fictional story only. It stands for no real system |

Instance parameters come from world seeds 0 to 29. Surface shape, number of peaks, and noise (0%, 3%, or 10% of range) are drawn per instance as design inputs, not claims about any system. Surfaces have 3 or 4 factors to keep the BayBE baseline computable. Enzyme and photopolymer law worlds are excluded under rule 7. Story labels are fictional by construction and are disclosed as such.

Excluded under the rule: EDBO aryl amination (3,955 of 3,960 cells), Cernak blocks JD-03-10 and XZ-01-55 (duplicate level names), Iron Mind lookup tables (not distributed in the repositories checked).

## 4. Inputs varied

| Factor | Levels |
|---|---|
| Labels | Real names; coded; random permutation of real names, redrawn per trajectory; unrelated-domain story |
| Base prompt | Pool of 40 prompts varying tone, length, expertise, and structure, generated once and frozen. Real prompts from scientists are added if available before the freeze |
| Hint | None; or a randomly chosen level, crossed with source (self, PI, paper) and certainty (tentative, confident) |
| Asking strategy | None; 5 frozen wordings each of evidence-first, assumption audit, and caution; rationale requested or not |
| Setup | Single stateless request; multi-turn chat; orchestrator with blinded analyst and domain agent; analyst only, with labels decoded afterward |
| Feedback | True; scrambled across the grid |
| Budget | 4 rounds of 3; 8 rounds of 3; 12 rounds of 1 |
| Model | Opus 5.5, Sonnet 5.5, Haiku 5.5, each at its default effort |

Left for the scale-up, for simplicity: plain-text output, tools and web search, non-Anthropic models.

## 5. Assignment

Each trajectory takes one level of every factor. Levels are balanced by assigning each factor through an independently shuffled, equal-count list, seeded in advance, so any two factors are close to independent. Half of trajectories have no hint and 40% have no strategy. Each of the five sources gets one fifth of trajectories, split evenly among its worlds. On surfaces, "real" and "perm" assignments become "coded" and "story", and analyst setups always see coded labels, so analysis uses the labels actually shown.

Baselines (BayBE, random, one factor at a time) run on every world and budget, on true and scrambled feedback, using a label-blind encoding.

## 6. Two studies

### A. Search study, about 600 trajectories

Full searches sampled as in section 5.

### B. Replay study, about 1,500 single decisions

Histories come from a random policy, with no model involved, at 0, 3, 6, and 12 prior results. Each history is sent as a pair of single requests that differ in exactly one of: labels, base prompt, hint, strategy, rationale, feedback, or model. The two next decisions are compared. Setup and budget are not varied here, because a single request has no setup or budget to change. Feedback pairs use at least 3 prior results.

### Probes

Zero-shot rankings on every world and label condition. Source guess and row continuation on measured grids, with a fabricated decoy. Probes use Opus 5.5.

## 7. Measures and analysis

| Measure | Definition |
|---|---|
| Search quality | Mean grid percentile of best-so-far across the budget, on noiseless truth. Final percentile also reported |
| Decision shift | Total-variation distance between the factor-level distributions of paired decisions (study B) |
| Prior agreement | Spearman of the zero-shot ranking with the true marginal means, per world and label condition (probes) |
| Hint uptake | Share of picks at the randomly chosen hinted level, compared across hinted and unhinted trajectories |
| Failures | Retries, refusals, and random fills |

Analysis:

- Study A: regression of search quality on all factors with world fixed effects. Main-effect p-values come from randomization inference, reshuffling each factor among trajectories within a world, because cluster-robust errors gave false positives on mock data with no real effects. Intervals are HC3. Prespecified interactions: labels × feedback, labels × hint, labels × strategy, setup × hint, with HC3 p-values.
- Study B: mean paired effect per factor level, bootstrapped by world and history.
- Base prompt enters as a random effect, reported as variance across the 40 prompts.
- Each estimate is reported overall and by world family, with intervals and Benjamini-Hochberg control across all reported effects.
- Candidate rule of thumb: an effect with the same sign across world families whose interval excludes zero after correction. It is a candidate for the scale-up, not a conclusion.
- Rationales are descriptive text only and are not used as evidence of why a model acted.

The run also reports variance components, so the scale-up can be sized from them.

## 8. Freeze checklist

- [x] World generator, parameter ranges, and inclusion list
- [x] Prompt, hint, and strategy pools: `data/pools/pools.json`, generated once by Opus 5.5 and regenerated once before any run to correct first-person voice and how the objective reads
- [x] Assignment table: seeded `assign.build`, 600 trajectories, written to `runs/survey/assignment.json`
- [x] Analysis script: `survey/analyze.py`, tested on mock and simulated null data only. On 12 simulated null datasets of 600 trajectories, the share of main effects with p below .05 was 0.042 (HC3) and 0.073 (randomization), a small sample of simulations
- [ ] Spend cap

## 9. Cost, an estimate

Using debug-run rates and the Batch API: about $20 for study A, $7 for study B, and $5 for probes and generating pools. About $35 to $40 total, updated from a 20-trajectory check that cost $1.53 online. Budget sets the sample size, so a lower cap means fewer trajectories.

## 10. Standing of earlier work

The pilot, debug runs, and SPEC.md amendments are exploratory and informed this design. Pilot estimates are not used to set expectations or sample sizes here.
