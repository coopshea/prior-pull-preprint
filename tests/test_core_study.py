import json
from dataclasses import asdict

import numpy as np
import pytest

from priorpull import llm
from priorpull.survey import core, inputs, run, views
from priorpull.worlds import load_survey


@pytest.fixture(scope="module")
def worlds():
    return load_survey(0)


@pytest.fixture(scope="module")
def pools():
    return inputs.load()


@pytest.fixture(scope="module")
def unit(worlds, pools):
    return core.build_units(worlds, 1, len(pools["base"]))[5]


def test_each_arm_differs_from_reference_in_one_input(unit):
    base = asdict(core.row_for(unit, "A1"))
    expected = {"A2": set(), "scrambled": {"feedback"}, "coded": {"labels", "effective_labels"}, "swapped": {"labels", "effective_labels"},
                "prompt": {"prompt"}, "hunch": {"hint"}, "evidence": {"strategy"}, "audit": {"strategy"}, "caution": {"strategy"}, "no_rationale": {"rationale"}}
    for arm, fields in expected.items():
        other = asdict(core.row_for(unit, arm))
        assert {k for k in base if base[k] != other[k]} == fields, arm


def test_units_are_balanced_and_reproducible(worlds, pools):
    a = core.build_units(worlds, 4, len(pools["base"]))
    assert len(a) == 4 * 4 * 4
    assert core.units_digest(a) == core.units_digest(core.build_units(worlds, 4, len(pools["base"])))


def test_history_is_shared_across_arms_except_scrambled_values(worlds, unit):
    u = core.Unit(0, "shields", "shields", 6, 0, 1, "self_tentative", 0)
    w = worlds["shields"]
    h = {a: run.random_history(w, core.row_for(u, a), 6) for a in ("A1", "coded", "scrambled")}
    assert h["A1"] == h["coded"]
    assert [x[0] for x in h["A1"]] == [x[0] for x in h["scrambled"]]
    assert [x[1] for x in h["A1"]] != [x[1] for x in h["scrambled"]]


def test_scrambled_skipped_with_no_history():
    u = core.Unit(0, "shields", "shields", 0, 0, 1, "self_tentative", 0)
    assert "scrambled" not in core.arms_for(u)


def test_identical_picks_have_zero_change(worlds):
    w = worlds["shields"]
    picks = [[0, 1, 0, 0, 0], [1, 2, 1, 0, 1], [2, 3, 0, 1, 2]]
    assert core.change_in_picks(picks, picks, w) == 0


def test_follows_results_and_prior(worlds):
    w = worlds["efb"]
    history = [((0, 0, 0), 10.0), ((1, 1, 1), 90.0)]
    assert core.follows_results([[1, 1, 1]], history, w) == 1.0
    assert core.follows_results([[0, 0, 0]], history, w) == 0.0
    assert core.follows_results([[0, 0, 0]], [], w) is None
    shown = views.view(w, "real", 0).labels
    prior = {0: {shown[0][1]: 1.0}, 1: {shown[1][1]: 0.0}, 2: {}}
    assert core.follows_prior([[1, 1, 0]], shown, prior) == 0.5


def test_mock_run_end_to_end(worlds, pools, tmp_path):
    units = core.build_units(worlds, 2, len(pools["base"]))
    store = llm.Store(tmp_path, 1.0)
    results = core.run_units(units, worlds, pools, llm.MockBackend(junk_rate=0.02), store, model="mock")
    assert len(results) == sum(len(core.arms_for(u)) for u in units)
    m = core.measure(results, worlds, {})
    assert set(core.figure2_table(m)["change"]) == set(core.GROUP.values())
    assert not core.figure3_table(m).empty
    assert len(core.figure4_table(m)) == 4


def test_noise_floor_matches_changes_when_nothing_matters(worlds, pools, tmp_path):
    """With a model that ignores its inputs, no change should stand out from asking again."""
    units = core.build_units(worlds, 60, len(pools["base"]))
    store = llm.Store(tmp_path, 1.0)
    results = core.run_units(units, worlds, pools, llm.MockBackend(junk_rate=0), store, model="mock")
    t = core.figure2_table(core.measure(results, worlds, {})).set_index("change")
    noise = t.loc["Ask again (noise)"]
    # coded and swapped change the option names, so a random model still lands on different levels at the same rate
    for name, row in t.iterrows():
        assert abs(row["mean"] - noise["mean"]) < 0.05, name


def test_cap_blocks_a_batch_that_would_overrun(worlds, pools, tmp_path):
    units = core.build_units(worlds, 1, len(pools["base"]))
    store = llm.Store(tmp_path, 0.01)

    class Boom:
        def run(self, requests):
            raise AssertionError("nothing should be sent")

    with pytest.raises(llm.BudgetExceeded):
        core.run_units(units, worlds, pools, Boom(), store, model=core.MODEL)
