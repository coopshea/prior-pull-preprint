import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from priorpull import llm
from priorpull.survey import analyze, assign, inputs, run, views
from priorpull.worlds import load_survey


@pytest.fixture(scope="module")
def worlds():
    return load_survey()


@pytest.fixture(scope="module")
def pools():
    return inputs.load()


def test_every_world_is_a_complete_unique_grid(worlds):
    assert sum(w.source == "cernak" for w in worlds.values()) == 29
    for w in worlds.values():
        assert not np.isnan(w.y).any(), w.id
        for f, g in enumerate(w.factors):
            assert len(set(g.levels)) == w.shape[f], (w.id, g.name)


def test_measured_values_are_returned_as_published(worlds):
    from priorpull.worlds import Oracle
    w = worlds["cernak_XZ-01-107_60"]
    assert w.y.max() > 100
    best = tuple(int(v) for v in np.unravel_index(int(np.argmax(w.y)), w.shape))
    assert Oracle(w, 0).observe(best) == round(float(w.y.max()), 1)


def test_no_world_selection_by_landscape(worlds):
    assert "cernak_JD-03-07_30" in worlds and worlds["cernak_JD-03-07_30"].y.max() == 0


@pytest.mark.parametrize("labels", ["real", "coded", "perm", "story"])
def test_views_are_bijections(worlds, labels):
    for w in worlds.values():
        if w.source == "surface" and labels in ("real", "perm"):
            continue
        p = views.view(w, labels, 3)
        for f, n in enumerate(w.shape):
            assert len(set(p.labels[f])) == n, (w.id, f)
            assert sorted(p.display_order[f]) == list(range(n))


def test_coded_view_hides_names(worlds, pools):
    w = worlds["shields"]
    p = views.view(w, "coded", 0)
    text = run.settings_text(p) + p.context
    assert "Ligand" not in text and "XPhos" not in text and "palladium" not in text


def test_assignment_balances_each_factor(worlds, pools):
    rows = assign.build(worlds, 600, len(pools["base"]))
    assert assign.table_digest(rows) == assign.table_digest(assign.build(worlds, 600, len(pools["base"])))
    for attr, levels in (("labels", assign.LABELS), ("setup", assign.SETUPS), ("model", assign.MODELS)):
        counts = [sum(getattr(r, attr) == x for r in rows) for x in levels]
        assert max(counts) - min(counts) <= 1, attr
    sources = {}
    for r in rows:
        sources[worlds[r.world].source] = sources.get(worlds[r.world].source, 0) + 1
    assert set(sources.values()) == {120}


def test_all_templates_render(worlds, pools):
    p = views.view(worlds["efb"], "real", 0)
    row = assign.Row(0, "efb", "real", "real", 0, "none", "none", 0, True, "single", "true", 4, 3, "mock")
    for i in range(len(pools["base"])):
        row.prompt = i
        text = run.base_prompt(p, row, [], 1, pools, "")
        assert "{" not in text and p.objective in text


def test_mock_end_to_end_and_blinding(worlds, pools, tmp_path):
    rows = assign.build(worlds, 80, len(pools["base"]))
    store = llm.Store(tmp_path, 1.0)
    be = llm.MockBackend(junk_rate=0.03)
    records = run.run_search(rows, worlds, pools, be, store, model_override="mock")
    assert all(len(r["picks"]) == row.rounds * row.q for r, row in zip(records, rows))
    log = [json.loads(l) for l in store.path.read_text().splitlines()]
    analyst = [x for x in log if x["tag"].get("role") == "analyst"]
    assert analyst and not any("palladium" in x["prompt"] or "Ligand" in x["prompt"] for x in analyst)
    table = analyze.search_table(rows, records, worlds)
    effects = analyze.main_effects(table, n_perm=19)
    assert {"estimate", "p_bh"} <= set(effects.columns)
    pairs = run.build_pairs(worlds, 28, rows)
    replay = run.run_replay(pairs, {r.id: r for r in rows}, worlds, pools, llm.MockBackend(junk_rate=0), store, model_override="mock")
    rt = analyze.replay_table(replay, worlds)
    assert set(rt.factor) == set(run.REPLAY_FACTORS)


def test_replay_pairs_differ_only_in_the_named_factor(worlds, pools):
    rows = assign.build(worlds, 50, len(pools["base"]))
    rng = np.random.default_rng(0)
    world = worlds[rows[0].world]
    for factor in run.REPLAY_FACTORS:
        other = run._variant(rows[0], factor, world, pools, rng)
        a, b = asdict(rows[0]), asdict(other)
        changed = {k for k in a if a[k] != b[k]}
        assert len(changed) == 1, (factor, changed)


def test_scrambled_and_true_histories_share_conditions(worlds):
    row = assign.Row(0, "shields", "real", "real", 0, "none", "none", 0, True, "single", "true", 4, 3, "mock")
    w = worlds["shields"]
    scram = assign.Row(**{**asdict(row), "feedback": "scrambled"})
    a, b = run.random_history(w, row, 6), run.random_history(w, scram, 6)
    assert [x[0] for x in a] == [x[0] for x in b]
    assert [x[1] for x in a] != [x[1] for x in b]
