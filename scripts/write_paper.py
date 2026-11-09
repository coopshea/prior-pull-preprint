"""Compose the paper from saved runs. Numbers come from runs/core and data/raw, not typed by hand where it matters."""
import re
from pathlib import Path

import pandas as pd

root = Path(__file__).resolve().parents[1]
path = root / "pages" / "prior-pull-paper.html"
old = path.read_text()
head = old[: old.index("<article>")]
svgs = re.findall(r"<svg .*?</svg>", old, flags=re.S)
design_svg, change_svg, trend_svg = svgs[0], svgs[1], svgs[2]
tables = re.findall(r"<table>.*?</table>", old, flags=re.S)
strategy_table = next(t for t in tables if "Follows results" in t)
source_table = next(t for t in tables if "Oil palm" in t)

base = pd.read_csv(root / "runs/core/shields_baseline.csv")
t2 = pd.read_csv(root / "runs/core/figure2.csv").set_index("change")
t3 = pd.read_csv(root / "runs/core/figure3.csv")
f = lambda v, d=2: f"{v:.{d}f}"
g = lambda who, n, col="mean": base[(base.who == who) & (base.n == n)].iloc[0][col]
E_, B_, O_ = "Expert players", "BO from human starts", "Opus, real names (pilot)"
tk = lambda c, k: t3[(t3.category == c) & (t3.k == k)].iloc[0]["mean"]


def shields_fig():
    x0, x1, y0, y1 = 44.0, 330.0, 170.0, 16.0
    sx = lambda n: x0 + (x1 - x0) * n / 25
    sy = lambda v: y0 + (y1 - y0) * (v - 50) / 50
    series = [(E_, "var(--pending)", "Experts (50)"), (B_, "var(--accent)", "BO from the same starts"), (O_, "var(--muted)", "Opus 5.5 pilot (6)")]
    out = ['<svg viewBox="0 0 520 215" width="100%" role="img" aria-label="Best yield so far against experiments for expert players, BO, and a small Opus pilot on the Shields grid.">']
    out.append(f'<path class="grid" d="M{x0} {y1} V{y0} H{x1}"/>')
    for v in (50, 75, 100):
        out.append(f'<text class="sub" x="{x0 - 6}" y="{sy(v) + 4:.1f}" text-anchor="end">{v}</text>')
    for n in (5, 10, 15, 20, 25):
        out.append(f'<text class="sub" x="{sx(n):.1f}" y="{y0 + 15}" text-anchor="middle">{n}</text>')
    out.append(f'<text class="sub" x="{(x0 + x1) / 2}" y="{y0 + 33}" text-anchor="middle">Experiments run</text><text class="sub" x="{x0}" y="10">Best yield so far (%)</text>')
    for who, color, label in series:
        d = base[(base.who == who) & (base.n <= 25)].sort_values("n")
        pts = " ".join(f"{sx(r.n):.1f},{sy(r.mean):.1f}" for r in d.itertuples())
        dash = ' stroke-dasharray="3 3"' if who == O_ else ""
        out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"{dash}/>')
        for r in d.itertuples():
            out.append(f'<path d="M{sx(r.n):.1f} {sy(r.lo):.1f} V{sy(r.hi):.1f}" stroke="{color}" fill="none"/><circle cx="{sx(r.n):.1f}" cy="{sy(r.mean):.1f}" r="3" fill="{color}"/>')
        last = d.iloc[-1]
        offset = {E_: 12, B_: -4, O_: 4}[who]
        out.append(f'<text class="sub" x="{x1 + 8}" y="{sy(last["mean"]) + offset:.1f}">{label}</text>')
    out.append("</svg>")
    return "".join(out)


body = f"""<article>
<header>
  <h1>What steers an LLM's next experiment? Results, names, and wording in reaction optimization</h1>
  <p class="byline">Cooper Shea. Draft of 8 November 2026. Analysis code and text prepared with Claude Code.</p>
  <span class="status">Working draft. One model, measured chemistry data, no human participants.</span>
</header>

<section class="abstract">
  <h2>Abstract</h2>
  <p>Bayesian optimization (BO) has outperformed expert chemists at reaction optimization, and scientists now increasingly ask language models what to try next. Whether that brings BO-quality decisions is an open question. Before testing it, we measured what steers a model's suggestion. On 160 histories from measured reaction grids, we asked Claude Opus 5.5 for the next three experiments, then changed one input at a time. Repeating the request changed the picks by {f(t2.loc['Ask again (noise)', 'mean'])} on a 0 to 1 scale. A different prompt changed them by {f(t2.loc['Different prompt', 'mean'])}, advice by {f(t2.loc['Added advice', 'mean'])}, a hunch by {f(t2.loc['Added hunch', 'mean'])}, coded or swapped names by {f(t2.loc['Coded names', 'mean'])} and {f(t2.loc['Swapped names', 'mean'])}, and scrambled results by {f(t2.loc['Scrambled data', 'mean'])}. The influence of results grew with the amount of data, and the influence of names declined. Advice to rely on the data shifted picks toward the results shown without improving their true rank. We did not test BO itself, and propose a direct comparison as next steps.</p>
</section>

<div class="columns">
<h2>1. The claim</h2>
<p>Experimental optimization is often done by intuition and by changing one factor at a time. Shields et al. [1] tested this directly. Fifty chemists and engineers played an online game on a fully measured direct arylation grid, choosing five experiments per round, and BO outperformed them in efficiency and consistency. The released data include BO runs started from the players' own first choices. In the released data [2], by 25 experiments experts averaged a best yield of {f(g(E_, 25), 1)}% and BO {f(g(B_, 25), 1)}%, and {round(g(E_, 25, 'reach90') * 100)}% of experts versus {round(g(B_, 25, 'reach90') * 100)}% of BO runs had reached 90% (Figure 1). If trained experts have something to learn from BO, most practitioners have more.</p>
<p>BO is hard to adopt, however. It needs code, statistical judgment, and setup. Many researchers will instead paste their results into a chat and ask what to try next.</p>

<h2>2. The question</h2>
<p>How do language models affect this kind of decision? Do they enable better search, and what could go wrong? A model's suggestion can draw on the results it is shown, on knowledge from training, and on how the question is phrased. We start by measuring the relative influence of each.</p>

<h2>3. Prior work and the gap</h2>
<table>
<thead><tr><th>Work</th><th>Conclusion</th><th>Gap left</th></tr></thead>
<tbody>
<tr><td>Shields 2021 [1]</td><td>BO beat experts on one reaction grid</td><td>No LLMs</td></tr>
<tr><td>Gupta 2025 [3]</td><td>LLM agents barely reacted to shuffled feedback on gene tasks, and classical methods beat them</td><td>Older models. No test of names or wording</td></tr>
<tr><td>MacKnight 2025 [4]</td><td>LLMs beat BO on familiar reaction grids</td><td>Attributes this to prior knowledge without removing it</td></tr>
<tr><td>M&uuml;ller 2026 [5]</td><td>Agents can run BO campaigns on real platforms. Scripted loops beat step-by-step tool calls</td><td>All arms already use BO, so plain chat is not compared</td></tr>
<tr><td>Cooper 2026 [6]</td><td>An expert team's best catalyst was above both LLM runs on one problem, significantly so for one run. The final best was nearly absent from LLM proposals made without feedback</td><td>One problem, no test of inputs</td></tr>
<tr><td>SciSycEval 2026 [7]</td><td>Experimental design was the most sycophancy-prone task studied</td><td>Single-turn, scored on text</td></tr>
</tbody></table>
<p>In the papers we read, we did not find a controlled measurement of how results, names, and wording each move an LLM's choice of next experiment. A related paper [8] could not be accessed, so this claim is provisional.</p>

<h2>4. What we tested</h2>
<p>We used complete measured grids with open licenses: direct arylation (1,728 yields) [1], Suzuki coupling (3,696) [9], oil palm delignification (64 conditions in duplicate) [10], and blocks of a C&ndash;N coupling screen [11], of which the first 10 of 29 eligible blocks were used. Each of the four sources contributed equally.</p>
<p>One unit is a world, a random history of 0, 3, 6, or 12 measured results, and a prompt from a pool of 40 varied templates. Each unit was sent to Opus 5.5 once per condition (Figure 2), with no tools, 160 units and 1,720 decisions in all. Each comparison changes one input. Sampling and analysis were fixed before this run, after earlier exploratory work on the design.</p>
<p>We report change in picks (total-variation distance between the option distributions of two sets of three picks, averaged over settings), true rank (grid percentile of the picks), and agreement with the best option in the results shown and with the model's own no-results ranking.</p>
</div>

<figure class="plain"><div class="figscroll">{shields_fig()}</div><figcaption><b>Figure 1.</b> Best yield so far on the direct arylation grid. Expert and BO data are from the published game [1, 2]. Players could stop early, and keep their best result. The Opus points are a 6-run exploratory pilot in batches of 3, on a grid the model named as a known dataset, so they are illustrative. Intervals are 95%.</figcaption></figure>
<figure><div class="figscroll">{design_svg}</div><figcaption><b>Figure 2.</b> One unit of the replay study. Because the history is shared, each comparison isolates a single input. Repeating the reference request gives a noise floor.</figcaption></figure>

<div class="panels">
  <figure class="panel">{change_svg}<figcaption><b>Figure 3.</b> Change in picks by input, with 95% intervals. The dashed line is the noise floor.</figcaption></figure>
  <figure class="panel">{trend_svg}<figcaption><b>Figure 4.</b> Change in picks against the number of results shown. Names pools coded and swapped. Wording pools prompt, hunch, and advice.</figcaption></figure>
</div>

<div class="columns">
<h2>5. Results</h2>
<p>All 1,720 decisions were usable. Repeating the same request changed the picks by {f(t2.loc['Ask again (noise)', 'mean'])}. A different prompt changed them by {f(t2.loc['Different prompt', 'mean'])}, which exceeds the noise floor by 0.02 (interval 0.00 to 0.04). Advice, hunches, names, and scrambled results each changed picks by more (Figure 3).</p>
<p>Scrambling mattered more as data accumulated, rising from {f(tk('Data', 3))} at 3 results to {f(tk('Data', 12))} at 12. Names mattered less, falling from {f(tk('Priors', 0))} with no results to {f(tk('Priors', 12))} at 12 (Figure 4). Sources differed: on the C&ndash;N blocks, names moved picks about twice as far as scrambling, the reverse of the other three (Table 1).</p>
<p>Evidence-first and audit advice raised agreement with the best observed option and lowered agreement with the model's prior, but true rank fell by about 0.02 (Table 2).</p>
<p>The model's no-results rankings tracked measured option quality more often than not. Mean Spearman correlation with measured option means was 0.72 on direct arylation, 0.64 on Suzuki, and 0.38 on the C&ndash;N blocks, and its top choice matched the best option in 60%, 60%, and 27% of settings. On oil palm, its ranking of reaction time ran opposite to the measurements (&minus;0.52). It named the source of direct arylation and Suzuki data correctly in every try.</p>
<p><b class="tl">Table 1.</b> Mean change in picks by input and source.</p>
{source_table}
<p><b class="tl">Table 2.</b> Paired difference from the reference request, with 95% intervals.</p>
{strategy_table}

<h2>6. Significance</h2>
<p>These results suggest three practical points, each to be confirmed against outcomes. Rewording a request changed little beyond ordinary repetition. A user's results strongly influenced picks once several were available, so early suggestions lean more on the model's learned associations than later ones. Telling a model to rely on the data changed its choices but did not improve them here, so that advice alone is unlikely to substitute for an actual optimizer.</p>
<p>For model developers, shuffled-feedback and mislabeled-option tests are cheap checks of whether a model updates on evidence. For scientists, the Shields result suggests a practical default: use BO or a BO tool when the budget is small and the decision matters.</p>

<h2>7. Limits and next steps</h2>
<p>This study used one model, and chemistry data, two of whose grids the model recognizes. Replay measures single decisions on random histories. It does not measure a sequential optimization loop or what people do. The Figure 1 comparison spans different batch sizes and sample sizes.</p>
<p>Three studies follow directly. First, a head-to-head search on these grids: plain chat asked for help, chat asked for BO, the model with code execution, and the model with a real BO tool, scored against BO alone. Second, simulations validated against measured data, which would allow unlimited sequential experiments in settings the model has not seen. Third, a study with people guided in these different ways, using the Shields game as a template, since it is the only design here able to test whether model suggestions steer human reasoning.</p>

<h2>Data and code</h2>
<p>Grid sources, the prompt pool, the unit table, and analysis code are listed in <code>data/SOURCES.md</code> and the project repository. Every request and response is logged.</p>
</div>

<section>
  <h2>References</h2>
  <div class="refs">
    <p>[1] Shields BJ, Stevens J, Li J, et al. Bayesian reaction optimization as a tool for chemical synthesis. Nature 590, 2021.</p>
    <p>[2] Shields BJ. EvML. github.com/b-shields/EvML.</p>
    <p>[3] Gupta R, Hartford J, Liu B. LLMs for Bayesian optimization in scientific domains: are we there yet? Findings of EMNLP, 2025.</p>
    <p>[4] MacKnight R, Regio JE, Ethier JG, Baldwin LA, Gomes G. Pre-trained knowledge elevates large language models beyond traditional chemical reaction optimizers. arXiv:2509.00103, 2025.</p>
    <p>[5] M&uuml;ller M, et al. La Agente &Oacute;ptima: Towards agentic self-driving laboratories. arXiv:2609.04564, 2026.</p>
    <p>[6] Cooper AI, Yeow QJ, Xiao Z, et al. Large language models do not replace chemists in a closed-loop catalysis experiment. Research Square, 2026.</p>
    <p>[7] SciSycEval. Findings of EMNLP, 2026. Authors to be added.</p>
    <p>[8] Ciss&eacute; A, Cooper ME, Zhu M, et al. Can we automate scientific reasoning in closed-loop experiments using large language models? Digital Discovery 5(3), 2026.</p>
    <p>[9] Perera D, et al. A platform for automated nanomole-scale reaction screening and micromole-scale synthesis in flow. Science, 2018. Data via the EDBO repository.</p>
    <p>[10] Delignification of oil palm empty fruit bunches. Zenodo 23953649, 2026. Authors to be added.</p>
    <p>[11] Das, Zhang, Tan, et al. A 50,688-reaction data set reveals general ligands and mechanistic diversity in C&ndash;N couplings. J. Am. Chem. Soc., doi:10.1021/jacs.6c05959.</p>
  </div>
</section>
</article>
"""
head = head.replace("</style>", ".plain { background: transparent; }\n.tl { font-family: var(--sans); font-size: 13px; font-weight: 600; }\n</style>")
path.write_text(head + body)
print("words approx", len(re.sub("<[^>]+>", " ", body).split()), "chars", len(body))
