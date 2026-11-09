# Prior Pull

What steers a language model's suggestion for the next experiment: the results it is shown, the names of things, or the wording of the request?

This is a working research project. It measures one model (Claude Opus 5.5) on measured chemistry data and has no human participants.

## Where to start

| Path | Contents |
|---|---|
| `DESIGN.md` | The replay study as run, with its run record |
| [`pages/prior-pull-paper.pdf`](pages/prior-pull-paper.pdf) | The paper. Preprint, work in progress |
| `pages/prior-pull-paper.html` | Source of the paper |
| `citations/EVIDENCE.md` | Page-level evidence for each citation, checked by `scripts/verify_citations.py` |
| `data/SOURCES.md` | Data sources and licenses |
| `runs/core/` | Requests, replies, and measures from the main run |
| `SPEC.md`, `DESIGN_survey_archive.md` | Earlier exploratory plans, kept as a record |

## Running it

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e .
.venv/bin/python -m pytest -q tests
.venv/bin/python -m priorpull.survey.core_cli run --mock --cap 1 --per-cell 3
```

Real runs need `ANTHROPIC_API_KEY` in a `.env` file and a spend cap passed with `--cap`.

## Third-party material

Cited papers are referenced by link and are not stored here. `citations/EVIDENCE.md` records short quotes and the page each was checked on. `data/raw/` contains published datasets under their own licenses, listed in `data/SOURCES.md`, with MIT notices in `data/licenses/`.
