import json

import numpy as np
import pytest

from priorpull import llm, prompts, search
from priorpull.worlds import Oracle, load_all, present


@pytest.fixture(scope="module")
def worlds():
    return load_all()


def test_shields_matches_published_size(worlds):
    w = worlds["shields"]
    assert w.shape == (4, 12, 4, 3, 3)
    assert w.y.max() == 100.0
    assert not np.isnan(w.y).any()


def test_labels_are_bijections(worlds):
    for w in worlds.values():
        for s in w.semantics:
            p = present(w, s)
            for f, n in enumerate(w.shape):
                assert len(set(p.labels[f])) == n
                assert sorted(p.display_order[f]) == list(range(n))


def test_misleading_swap_gives_top_name_the_worst_results(worlds):
    w = worlds["shields"]
    p = present(w, "S2")
    marg = w.marginals()[1]
    best, worst = int(np.argmax(marg)), int(np.argmin(marg))
    assert p.labels[1][worst] == w.factors[1].levels[best]
    assert p.labels[1][best] == w.factors[1].levels[worst]


@pytest.mark.parametrize("wid", ["enzyme", "resin"])
def test_reversal_moves_the_apparent_optimum(worlds, wid):
    w = worlds[wid]
    best = np.unravel_index(int(np.argmax(w.y)), w.shape)
    p = present(w, "S2")
    moved = [
        p.labels[f][int(best[f])] != w.factors[f].levels[int(best[f])] for f in range(len(w.shape))
    ]
    assert sum(moved) >= 2, moved


def test_optimum_is_interior_on_formula_worlds(worlds):
    for wid in ("enzyme", "resin"):
        w = worlds[wid]
        best = np.unravel_index(int(np.argmax(w.y)), w.shape)
        interior = sum(0 < best[f] < w.shape[f] - 1 for f, g in enumerate(w.factors) if g.numeric)
        assert interior >= 2, (wid, best)


def test_formula_worlds_are_not_trivially_easy(worlds):
    for wid in ("enzyme", "resin", "gmm"):
        w = worlds[wid]
        assert (w.y > 90).mean() < 0.05, wid


def test_noise_is_common_across_arms(worlds):
    w = worlds["enzyme"]
    a, b = Oracle(w, 3), Oracle(w, 3)
    assert a.observe((4, 3, 4, 2)) == b.observe((4, 3, 4, 2))
    assert a.observe((4, 3, 4, 2)) != a.observe((4, 3, 4, 2)) or w.noise_sd == 0


def test_shuffle_keeps_the_value_distribution(worlds):
    w = worlds["shields"]
    o = Oracle(w, 1, "Fs")
    assert np.allclose(np.sort(o._y.ravel()), np.sort(w.y.ravel()))
    assert not np.allclose(o._y, w.y)


def test_prompt_never_leaks_truth_under_anonymization(worlds):
    w = worlds["shields"]
    p = present(w, "S1")
    text = prompts.build_prompt(p, [], 1, 8, 3, "T0", "W-")
    for f in w.factors:
        assert f.name not in text
    assert "palladium" not in text and "XPhos" not in text


def test_hint_names_worst_level_as_displayed(worlds):
    w = worlds["shields"]
    for sem in ("S0", "S1", "S2"):
        p = present(w, sem)
        f, level = prompts.hinted_level(p, "W-")
        assert level == w.hint_target()[2]
        assert p.labels[f][level] in prompts.hint_text(p, "W-")


def test_parse_rejects_wrong_count_and_bad_labels(worlds):
    p = present(worlds["shields"], "S0")
    row = {prompts.key_for(n): p.options(f)[0] for f, n in enumerate(p.names)}
    slots = lambda r, n: {f"experiment_{i + 1}": r for i in range(n)}
    good = llm.Reply("k", json.dumps({"reason": "r", "experiments": slots(row, 3)}), "end_turn", 1, 1, "mock")
    assert search.parse_reply(good, p, 3, False) is not None
    short = llm.Reply("k", json.dumps({"reason": "r", "experiments": slots(row, 1)}), "end_turn", 1, 1, "mock")
    assert search.parse_reply(short, p, 3, False) is None
    bad = dict(row)
    bad[prompts.key_for(p.names[0])] = "nope"
    wrong = llm.Reply("k", json.dumps({"reason": "r", "experiments": slots(bad, 3)}), "end_turn", 1, 1, "mock")
    assert search.parse_reply(wrong, p, 3, False) is None


def test_mock_run_is_resumable_and_free_on_replay(worlds, tmp_path):
    conds = [search.Condition("shields", "S0"), search.Condition("enzyme", "S2", "W-")]
    store = llm.Store(tmp_path, 10.0)
    first = search.run_llm("mock", worlds, conds, [0, 1], llm.MockBackend(), store)
    assert all(len(t.picks) == 24 for t in first)

    class Fail:
        def run(self, requests):
            raise AssertionError("resume should not call the backend")

    resumed = llm.Store(tmp_path, 10.0)
    second = search.run_llm("mock", worlds, conds, [0, 1], Fail(), resumed)
    assert [t.picks for t in first] == [t.picks for t in second]


def test_budget_cap_stops_the_run(worlds, tmp_path):
    store = llm.Store(tmp_path, 0.0)
    with pytest.raises(llm.BudgetExceeded):
        search.run_llm("mock", worlds, [search.Condition("shields", "S0")], [0], llm.MockBackend(), store)


def test_seeds_of_identical_prompts_are_separate_samples(worlds, tmp_path):
    store = llm.Store(tmp_path, 10.0)
    t = search.run_llm("mock", worlds, [search.Condition("shields", "S0")], [0, 3], llm.MockBackend(junk_rate=0), store, rounds=1)
    assert t[0].template == t[1].template
    assert t[0].picks != t[1].picks


def test_baselines_spend_exact_budget(worlds):
    for wid, w in worlds.items():
        for fn in (search.run_random, search.run_ofat):
            t = fn(w, 0)
            assert len(t.picks) == w.rounds * 3, (wid, fn.__name__)
            assert min(t.rounds) == 1 and max(t.rounds) == w.rounds


def test_client_never_uses_host_base_url():
    assert llm.API_BASE == "https://api.anthropic.com"


def test_hint_shares_and_recognition(worlds):
    from priorpull import analysis
    w = worlds["shields"]
    f, best, worst = w.hint_target()
    t = search.Trajectory("mock", "shields", "S0", "W0", "F0", 0, "T0", reasons=["This is the Shields/Doyle set"])
    p = [0] * len(w.shape)
    p[f] = worst
    t.picks = [p, p]
    assert analysis.hint_shares(t, w)["share_worst_level"] == 1.0
    assert analysis.names_dataset(t)


def test_continuation_does_not_count_zero_matches_as_recall(worlds):
    from priorpull import probes
    w = worlds["shields"]
    truth = probes._source_lines(w)[30:50]
    reply = llm.Reply("k", json.dumps({"rows": truth}), "end_turn", 1, 1, "mock")
    s = probes.score_continuation(reply, w, 30, 20)
    assert s["exact_rows"] == 20
    assert s["exact_nonzero"] == s["true_nonzero"] < 20


def test_efb_grid_is_complete_and_returns_measured_duplicates(worlds):
    w = worlds["efb"]
    assert w.shape == (4, 2, 8) and w.size == 64 and w.rounds == 4
    o = Oracle(w, 2)
    first, second = o.observe((1, 0, 3)), o.observe((1, 0, 3))
    assert {round(float(v), 1) for v in w.replicates[1, 0, 3]} == {first, second} or first == second


def test_new_strategies_run_and_orchestrator_stays_blind(worlds, tmp_path):
    conds = [
        search.Condition("shields", "S2", strategy=s) for s in ("M1", "NR", "OPEN", "ORCH")
    ] + [search.Condition("efb", "S0", "W-", strategy="ORCH")]
    store = llm.Store(tmp_path, 10.0)
    trajs = search.run_llm("mock", worlds, conds, [0], llm.MockBackend(junk_rate=0), store)
    assert all(len(t.picks) == worlds[t.world].rounds * 3 for t in trajs)
    assert {t.strategy: max(t.rounds) for t in trajs if t.world == "shields"}["OPEN"] == 1
    records = [json.loads(l) for l in store.path.read_text().splitlines()]
    analyst = next(r for r in records if r["tag"].get("role") == "analyst" and r["tag"]["round"] == 2)
    assert "Ligand" not in analyst["prompt"] and "palladium" not in analyst["prompt"]
    assert "hunch" not in "".join(r["prompt"] for r in records if r["tag"].get("role") == "analyst")
    orchestrator = next(r for r in records if r["tag"].get("strategy") == "ORCH" and not r["tag"].get("role") and r["tag"]["round"] == 2)
    assert "Results so far" not in orchestrator["prompt"]
    domain = next(r for r in records if r["tag"].get("role") == "domain")
    assert "Results so far" not in domain["prompt"]


def test_shuffled_oracle_ignores_measured_duplicates(worlds):
    o = Oracle(worlds["efb"], 1, "Fs")
    assert o.observe((0, 0, 0)) == o.observe((0, 0, 0))
