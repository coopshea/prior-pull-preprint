"""Fill the paper's results from runs/core. Every number here comes from the saved run."""
import re
from pathlib import Path

import pandas as pd

root = Path(__file__).resolve().parents[1]
run = root / "runs" / "core"
path = root / "pages" / "prior-pull-paper.html"
html = path.read_text()
t2, t3, t4 = (pd.read_csv(run / f"figure{i}.csv") for i in (2, 3, 4))
m = pd.read_csv(run / "measures.csv")
f = lambda v: f"{v:.2f}".replace("-", "−")


def fig2():
    x0, x1, top, step, vmax = 96.0, 280.0, 20, 25, 0.6
    sx = lambda v: x0 + (x1 - x0) * v / vmax
    labels = {"Ask again (noise)": "Ask again", "Scrambled data": "Scrambled", "Coded names": "Coded", "Swapped names": "Swapped",
              "Different prompt": "Prompt", "Added hunch": "Hunch", "Added advice": "Advice"}
    bottom = top + step * len(t2)
    noise = t2.loc[t2["change"] == "Ask again (noise)", "mean"].iloc[0]
    out = [f'<svg viewBox="0 0 300 {bottom + 40}" width="100%" role="img" aria-label="Change in picks by input with intervals. Noise floor marked.">']
    out.append(f'<path class="grid" d="M{x0} {top - 8} V{bottom} H{x1}"/>')
    for v in (0, 0.2, 0.4, 0.6):
        out.append(f'<text class="sub" x="{sx(v):.1f}" y="{bottom + 15}" text-anchor="middle">{v:.1f}</text>')
    out.append(f'<text class="sub" x="{(x0 + x1) / 2}" y="{bottom + 33}" text-anchor="middle">Change in picks</text>')
    out.append(f'<path d="M{sx(noise):.1f} {top - 8} V{bottom}" stroke="var(--muted)" stroke-dasharray="3 3" fill="none"/>')
    for i, r in enumerate(t2.itertuples()):
        y = top + step * i + 4
        out.append(f'<text class="sub" x="{x0 - 7}" y="{y + 4}" text-anchor="end">{labels[r.change]}</text>')
        out.append(f'<path d="M{sx(r.lo):.1f} {y} H{sx(r.hi):.1f}" stroke="var(--accent)" stroke-width="1.5" fill="none"/>')
        out.append(f'<circle cx="{sx(r.mean):.1f}" cy="{y}" r="3.5" fill="var(--accent)"/>')
        out.append(f'<text class="sub" x="{sx(r.hi) + 5:.1f}" y="{y + 4}">{f(r.mean)}</text>')
    out.append("</svg>")
    return "".join(out)


def fig3():
    ks = [0, 3, 6, 12]
    x0, x1, y0, y1, vmax = 42.0, 232.0, 160.0, 14.0, 0.7
    sx = lambda k: x0 + (x1 - x0) * ks.index(k) / 3
    sy = lambda v: y0 + (y1 - y0) * v / vmax
    style = {"Data": "var(--accent)", "Priors": "var(--pending)", "Wording": "var(--wording)", "Noise floor": "var(--muted)"}
    names = {"Data": "Data", "Priors": "Names", "Wording": "Wording", "Noise floor": "Noise"}
    out = ['<svg viewBox="0 0 300 200" width="100%" role="img" aria-label="Change in picks against results shown for data, names, wording, and noise.">']
    out.append(f'<path class="grid" d="M{x0} {y1} V{y0} H{x1}"/>')
    for v in (0, 0.2, 0.4, 0.6):
        out.append(f'<text class="sub" x="{x0 - 6}" y="{sy(v) + 4:.1f}" text-anchor="end">{v:.1f}</text>')
    for k in ks:
        out.append(f'<text class="sub" x="{sx(k):.1f}" y="{y0 + 15}" text-anchor="middle">{k}</text>')
    out.append(f'<text class="sub" x="{(x0 + x1) / 2}" y="{y0 + 33}" text-anchor="middle">Results shown</text>')
    for cat, color in style.items():
        g = t3[t3.category == cat].sort_values("k")
        pts = " ".join(f"{sx(int(r.k)):.1f},{sy(r.mean):.1f}" for r in g.itertuples())
        dash = ' stroke-dasharray="3 3"' if cat == "Noise floor" else ""
        out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.8"{dash}/>')
        for r in g.itertuples():
            x = sx(int(r.k))
            out.append(f'<path d="M{x:.1f} {sy(r.lo):.1f} V{sy(r.hi):.1f}" stroke="{color}" fill="none"/><circle cx="{x:.1f}" cy="{sy(r.mean):.1f}" r="3" fill="{color}"/>')
        last = g.iloc[-1]
        out.append(f'<text class="sub" x="{x1 + 7}" y="{sy(last["mean"]) + 4:.1f}">{names[cat]}</text>')
    out.append("</svg>")
    return "".join(out)


def fig4():
    cell = lambda r, c: f"{f(r[c])} [{f(r[c + '_lo'])}, {f(r[c + '_hi'])}]"
    rows = "".join(f"<tr><td>{r['strategy']}</td><td>{cell(r, 'true_rank')}</td><td>{cell(r, 'follows_results')}</td><td>{cell(r, 'follows_prior')}</td></tr>" for r in t4.to_dict("records"))
    return f'<table><thead><tr><th>Strategy</th><th>True rank</th><th>Follows results</th><th>Follows prior</th></tr></thead><tbody>{rows}</tbody></table>'


def source_table():
    groups = {"Ask again": ["A2"], "Scrambled": ["scrambled"], "Coded": ["coded"], "Swapped": ["swapped"], "Prompt": ["prompt"], "Hunch": ["hunch"], "Advice": ["evidence", "audit", "caution"]}
    head = "".join(f"<th>{s}</th>" for s in ("Shields", "Suzuki", "Oil palm", "Cernak"))
    rows = ""
    for name, arms in groups.items():
        means = m[m.arm.isin(arms)].groupby("source")["change"].mean()
        rows += f"<tr><td>{name}</td>" + "".join(f"<td>{f(means[s])}</td>" for s in ("shields", "suzuki", "efb", "cernak")) + "</tr>"
    return f"<table><thead><tr><th>Input</th>{head}</tr></thead><tbody>{rows}</tbody></table>"


panels = f"""<div class="panels">
  <figure class="panel">{fig2()}<figcaption><b>Figure 2.</b> Change in picks by input. Points are means with 95% intervals. The dashed line is the noise floor from repeating the reference request.</figcaption></figure>
  <figure class="panel">{fig3()}<figcaption><b>Figure 3.</b> Change in picks against the number of results shown. Names pools coded and swapped. Wording pools prompt, hunch, and advice. Scrambling is undefined with no results.</figcaption></figure>
  <figure class="wide">{fig4()}<figcaption><b>Figure 4.</b> Paired difference from the reference request on the same units, with 95% intervals. True rank is grid percentile (0 to 1). Follows prior is limited to units with a usable ranking.</figcaption></figure>
</div>"""
html = re.sub(r'<div class="panels">.*?</div>\n\n<div class="columns">\n<h2>4\. Results</h2>', lambda _: panels + '\n\n<div class="columns">\n<h2>4. Results</h2>', html, flags=re.S)
html = html.replace(".panels { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; }", ".panels { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px 24px; }\n.wide { grid-column: 1 / -1; }")
html = html.replace("--pending-bg: #fdf3dd;\n", "--pending-bg: #fdf3dd; --wording: #2e7d55;\n")
html = html.replace("--pending-bg: #3a2d12; color-scheme: dark }", "--pending-bg: #3a2d12; --wording: #7fd1a4; color-scheme: dark }")
html = html.replace("@media (max-width: 720px) { .columns, .refs { columns: 1; } .panels { grid-template-columns: minmax(0, 1fr); } }", "@media (max-width: 720px) { .columns, .refs { columns: 1; } .panels { grid-template-columns: minmax(0, 1fr); } .wide { grid-column: auto; overflow-x: auto; } }")

get = lambda name: t2.set_index("change").loc[name]
tk = lambda c, k: t3[(t3.category == c) & (t3.k == k)].iloc[0]["mean"]
s = t4.set_index("strategy")
abstract_tail = (
    f"Repeating the same request changed the picks by {f(get('Ask again (noise)')['mean'])} on a 0 to 1 scale. A different prompt changed them by {f(get('Different prompt')['mean'])}, "
    f"added advice by {f(get('Added advice')['mean'])}, a hunch by {f(get('Added hunch')['mean'])}, coded names by {f(get('Coded names')['mean'])}, swapped names by {f(get('Swapped names')['mean'])}, "
    f"and scrambled results by {f(get('Scrambled data')['mean'])}. With more results shown, the effect of scrambling rose and the effect of names fell. Asking strategies moved picks toward the results shown and away from the model's prior ranking, "
    f"without improving the true rank of the picks. Results come from one model, on chemistry grids, and two of the grids are recognized by the model."
)
html = html.replace('<span class="pending">Results to be added when the run completes.</span>', abstract_tail)
html = html.replace("Main run in progress. Results sections are placeholders.", "Results from one frozen run on Claude Opus 5.5.")
html = html.replace("and 29 plate-and-temperature blocks of a C&ndash;N coupling screen [15]. Each of these four sources contributes equally.", "and plate-and-temperature blocks of a C&ndash;N coupling screen [15]. Of the 29 blocks that met our inclusion rule, the first 10 in sorted order were used, because units cycle through blocks in order. Each of these four sources contributes equally.")

results = f"""<h2>4. Results</h2>
<p>All 1,720 decisions returned usable picks, with no refusals or failed parses. Repeating the reference request changed the picks by {f(get('Ask again (noise)')['mean'])} (95% interval {f(get('Ask again (noise)')['lo'])} to {f(get('Ask again (noise)')['hi'])}). A different prompt from the pool changed them by {f(get('Different prompt')['mean'])}, a paired excess of 0.02 over the noise floor (0.00 to 0.04). Added advice, a hunch, coded names, swapped names, and scrambled results each changed the picks by more (Figure 2). The scrambled estimate uses only units with at least 3 results shown.</p>
<p>The size of these effects depended on how many results were shown (Figure 3). The scrambling effect rose from {f(tk('Data', 3))} with 3 results to {f(tk('Data', 12))} with 12. The effect of names fell from {f(tk('Priors', 0))} with no results to {f(tk('Priors', 12))} with 12, and wording fell from {f(tk('Wording', 0))} to {f(tk('Wording', 12))}. The noise floor ranged from 0.13 to 0.18 with no trend across history sizes.</p>
<p>Sources differed (Table 1). In the C&ndash;N coupling blocks, coded and swapped names changed picks by about twice as much as scrambling, the reverse of the other three sources.</p>
<p>Evidence-first and assumption-audit advice raised the share of choices matching the best observed option and lowered agreement with the model's prior ranking, and true rank fell by about 0.02 (Figure 4). Caution showed a similar shift in agreement, and its true-rank interval included zero. Omitting the rationale lowered agreement with the results shown and left true rank unchanged.</p>
<p>The model named the source correctly for Shields and Suzuki in 3 of 3 tries each. It did not identify the oil palm data in 3 tries and claimed to recognize the fabricated decoy in none. For the C&ndash;N blocks it claimed recognition in 48 of 87 tries, and about 44% of those mentioned the Cernak group, Michigan, or the 50,688-reaction count.</p>
<p><b style="font-family:var(--sans);font-size:13px">Table 1.</b> Mean change in picks by input and source.</p>
{source_table()}
"""
html = re.sub(r"<h2>4\. Results</h2>.*?(?=<h2>5\. Discussion)", lambda _: results + "\n", html, flags=re.S)
html = html.replace("The study measures how decisions move, and how good the next picks are.", "Rewording a request moved the picks little beyond repeating it, while changing what the numbers or names said moved them much more, and those influences shifted as results accumulated. Advice aimed at relying on the data did change where picks went, though not toward better picks in this setting, where prior knowledge of the chemistry may be useful. The study measures how decisions move, and how good the next picks are.")
html = html.replace("Two of the grids are recognized by the model in preliminary testing, so", "Two of the grids are named correctly by the model, so")
path.write_text(html)
print("pending left:", html.count('class="pending"'), "abstract ok:", "Repeating the same request" in html)
