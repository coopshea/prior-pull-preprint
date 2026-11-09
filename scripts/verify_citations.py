"""Check every logged quote against the stated page of its PDF, recompute data claims, and write citations/EVIDENCE.md."""
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parents[1]
cit = root / "citations"


def pages(name):
    get = lambda *flags: subprocess.run(["pdftotext", *flags, str(cit / name), "-"], capture_output=True, text=True).stdout.split("\f")
    # Reading order suits sentences. Layout mode keeps table rows together.
    return [a + "\n" + b for a, b in zip(get(), get("-layout"))]


norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
S, C = "01_shields_nature_2021_full.pdf", "08_cisse_digital_discovery_2026.pdf"
# (ref, paper statement, file, page, quote, verdict, note)
E = [
 ("1 Shields 2021", "BO outperformed experts in efficiency and consistency", S, 1, "Bayesian optimization outperforms human decisionmaking in both average optimization efficiency", "Supported", "Full paper read."),
 ("1 Shields 2021", "Chemists and engineers played an online game", S, 1, "online game that links the decisions made by expert chemists and engineers", "Supported", ""),
 ("1 Shields 2021", "Fifty chemists and engineers", S, 5, "In total, 50 expert chemists and engineers", "Supported", "Also recomputed from the released data below."),
 ("1 Shields 2021", "Five experiments per round", S, 5, "run one batch of five experiments", "Supported", ""),
 ("1 Shields 2021", "Fully measured grid of 1,728", S, 5, "1,728 reactions including 12 ligands", "Supported", ""),
 ("1 Shields 2021", "BO started at random in the paper", S, 5, "each time starting with a different random initialization", "Supported", ""),
 ("1 Shields 2021", "Players made better first choices", S, 5, "significantly (p < 0.05) better initial", "Supported", "15% higher yield in the first batch than random selection."),
 ("1 Shields 2021", "BO's average passed theirs within three rounds", S, 5, "within three batches of five experiments the", "Supported", ""),
 ("1 Shields 2021", "Difference on the recorded data was significant after the fifth round", S, 6, "after the fifth batch of experiments the performance of", "Supported", "Applies to raw data and the lower bound. Under the authors' 'unrealistic' upper bound, there is no significant difference."),
 ("1 Shields 2021", "BO beat experts on one reaction grid", S, 6, "in the optimization of reaction 3", "Supported", ""),
 ("1 Shields 2021", "Players could stop early", S, 6, "most participants played fewer than 20 rounds", "Supported", "Carrying the best result forward is the paper's lower bound."),
 ("2 EvML", "Players chose five experiments per round", "02_evml_README.pdf", 1, "Your bench has room to run 5 experiments per", "Supported", "Game rules in the repository README."),
 ("2 EvML", "The released data include BO runs started from the players' first choices", "02_evml_analysis_notebook_headings.pdf", 1, "difference between BO when initialized by human and random choices", "Supported", "File name also says human_init. Recomputed: both groups have the same mean best yield after 5 experiments. These runs are in the repository. The Nature game comparison started BO at random."),
 ("3 Gupta 2025", "LLM agents barely reacted to shuffled feedback", "03_gupta_2025.pdf", 1, "permuted labels has no impact on performance", "Supported", "The paper states no impact. Our wording is slightly softer."),
 ("3 Gupta 2025", "Shuffling was on gene tasks", "03_gupta_2025.pdf", 4, "Cumulative number of hits secured by each method on the gene perturbation datasets", "Supported", "Table 1 caption. The abstract mentions both gene and molecule tasks."),
 ("3 Gupta 2025", "Classical methods beat them", "03_gupta_2025.pdf", 1, "consistently outperform LLM agents", "Supported", ""),
 ("3 Gupta 2025", "Gap: older models", "03_gupta_2025.pdf", 4, "Claude 3.5 Sonnet backbone", "Our reading", "Table 1 lists the models used for the shuffle comparison."),
 ("4 MacKnight 2025", "LLMs matched or exceeded BO on five reaction grids", "04_macknight_2025.pdf", 1, "Frontier LLMs consistently match or exceed BO performance across five single-objective datasets", "Supported", "Earlier draft said 'familiar' grids, which the source does not say. Corrected."),
 ("4 MacKnight 2025", "BO kept the advantage on the multi-objective case", "04_macknight_2025.pdf", 1, "BO retains superiority only for explicit multi-objective trade-offs", "Supported", ""),
 ("4 MacKnight 2025", "Gap: credited to prior knowledge without removing it", "04_macknight_2025.pdf", 1, "pre-trained knowledge in large language models", "Our reading", "No label-removal condition was found in the full-text read. Absence is our inference."),
 ("5 Muller 2026", "Agents can run BO campaigns on real platforms", "05_muller_2026.pdf", 1, "constructs and supervises Bayesian optimization campaigns across computational and experimental", "Supported", ""),
 ("5 Muller 2026", "Scripted loops beat step-by-step tool calls", "05_muller_2026.pdf", 28, "Tool-loop 0.319", "Supported on one benchmark", "Ackley 6D final 0.770 (specialist script) vs 0.319 (tool loop), 3 runs. On Shields arylation the gap is small, 0.865 vs 0.824 (MBSF@60)."),
 ("5 Muller 2026", "Gap: all arms already use BO", "05_muller_2026.pdf", 27, "the main agent implements and executes Bayesian optimization", "Our reading", "Quoted arm is the local BO comparator. All four arms are described on this page."),
 ("6 Cooper 2026", "The expert team's best matched the stronger LLM run within error", "06_cooper_2026.pdf", 7, "had the same catalytic activity, within error, as the LLM", "Supported", "Earlier draft said above both runs. Corrected to the source's wording."),
 ("6 Cooper 2026", "The expert team's best beat the weaker LLM run", "06_cooper_2026.pdf", 7, "was 22% more active than its weaker run", "Supported", "p = 3.5e-5 as reported."),
 ("6 Cooper 2026", "The final best was nearly absent from LLM proposals made without feedback", "06_cooper_2026.pdf", 5, "was essentially absent from the prior distribution of the repeat first batches", "Supported", "2 of 320 starting points. Applies to the best formulation at the end of Phase 1."),
 ("7 Zhang 2026 (SciSycEval)", "Experimental design was the most sycophancy-prone task", "07_scisyceval_2026.pdf", 8, "highest adjusted SR (65.3%) across all", "Supported", "Sentence names experimental design as highest."),
 ("7 Zhang 2026 (SciSycEval)", "Gap: single-turn, scored on text", "07_scisyceval_2026.pdf", 10, "interactive multi-turn scientific workflows", "Our reading", "Named as future work. The phrase single-turn does not appear in the paper."),
 ("8 Cisse 2026", "LLM search beat BO on the simulated photocatalysis problem", C, 10, "o3 only 26.2 2.9 9", "Supported", "Table 3, BO only 11.6. From identical starts, Table 4 gives o3 24.6 and BO 12.0."),
 ("8 Cisse 2026", "LLM search beat BO across batch sizes", C, 13, "the LLM search outperformed uncategorised BO for all batch sizes tested", "Supported", ""),
 ("8 Cisse 2026", "Problems are simulated and noise-free", C, 25, "noise-free ground truth models of a 10-D chemistry problem", "Supported", ""),
 ("8 Cisse 2026", "Bad hypotheses lowered performance", C, 15, "Both the bad and the mixed hypotheses reduced the performance of the optimisation", "Supported", "Table 6, 25.3 to 20.8 at 150 experiments."),
 ("8 Cisse 2026", "Most runs at least partly recovered", C, 18, "recovered, at least partially, from the bad hypotheses in 19/20 of the test runs", "Supported", ""),
 ("8 Cisse 2026", "With no attachment: 7.5 at 25, 25.1 at 150, optimum in 9 of 20", C, 20, "gpt-5 (med) 7.5 3.5 25.1 4.2 9", "Supported", "Table 8."),
 ("8 Cisse 2026", "With the 2020 paper: 18.5 at 25, 21.8 at 150, optimum in 0 of 20", C, 20, "+Nature 2020 18.5 1.9 21.8 1.5 0", "Supported", "Table 8."),
 ("8 Cisse 2026", "Salt explanation is the authors' speculation", C, 20, "scrutinising these logs and the associated data, we speculate", "Supported", ""),
 ("8 Cisse 2026", "Attaching the data raised the final result to 26.6", C, 20, "+Experimental data 23.2 3.3 26.6 2.7 11", "Supported", "Table 8."),
 ("8 Cisse 2026", "Model's own literature search made little net difference", C, 7, "of literature searching cancel out", "Supported", "The authors limit this to the photocatalysis problem."),
 ("8 Cisse 2026", "Gap: names stay real and results are not scrambled", C, 3, "maximizing the rate of hydrogen production from a photocatalyst mixture", "Our reading", "Quote shows real names in the prompt. Term search of the main text found no shuffled or anonymised condition. SI not read, including Table S3 on memorisation."),
 ("9 Perera 2018", "Source of the 3,696-reaction Suzuki table", "09_perera_2018_pubmed_abstract.pdf", 1, "5760 reactions", "NOT VERIFIED", "The abstract reports 5,760 reactions, and the EDBO repository code and notebooks do not name Perera. The paper now cites the EDBO repository for the table and states this is unconfirmed."),
 ("10 Aldana Rico 2026", "64 conditions in duplicate", "10_efb_zenodo_README.pdf", 1, "128 independent experimental runs", "Supported", ""),
 ("11 Das 2026", "50,688 C-N couplings in the Open Reaction Database", "11_cernak_ORD_record.pdf", 1, "50,688 C", "Supported", "Dataset record. The JACS version could not be downloaded."),
 ("11 Das 2026", "Authors of the C-N work", "11_cernak_preprint.pdf", 1, "Jayabrata Das", "Supported", "Preprint title differs from the JACS title. JACS title comes from OpenAlex."),
]
rows, bad = [], 0
for ref, claim, file, page, quote, verdict, note in E:
    assert len(quote.split()) <= 15, quote
    if not (cit / file).exists():  # papers are not redistributed, so a fresh clone has none
        rows.append((ref, claim, file, page, quote, verdict, note, "source not in repo"))
        continue
    ps = pages(file)
    ok = norm(quote) in norm(ps[page - 1])
    if not ok:
        bad += 1
        where = [i + 1 for i, p in enumerate(ps) if norm(quote) in norm(p)]
        print("NOT FOUND", ref, "|", quote, "| found on", where)
    rows.append((ref, claim, file, page, quote, verdict, note, "matched" if ok else "NOT MATCHED"))

g = pd.read_csv(root / "data/raw/evml/arylation_game_summary.csv")
b = pd.read_csv(root / "runs/core/shields_baseline.csv")
s = pd.read_csv(root / "data/raw/shields/experiment_index.csv")
get = lambda w, n, c: float(b[(b.who == w) & (b.n == n)].iloc[0][c])
data = [
 ("Fifty players", f"{len(g)} rows in data/raw/evml/arylation_game_summary.csv"),
 ("Direct arylation grid has 1,728 yields", f"{len(s)} rows in data/raw/shields/experiment_index.csv"),
 ("Suzuki table has 3,696 yields", f"{len(pd.read_csv(root / 'data/raw/suzuki/experiment_index.csv'))} rows in data/raw/suzuki/experiment_index.csv"),
 ("Experts 91.7% and BO 96.7% best yield at 25 experiments", f"{get('Expert players', 25, 'mean'):.1f} and {get('BO from human starts', 25, 'mean'):.1f}, scripts/shields_baseline.py"),
 ("50% of experts and 86% of BO runs reached 90%", f"{get('Expert players', 25, 'reach90'):.2f} and {get('BO from human starts', 25, 'reach90'):.2f}, scripts/shields_baseline.py"),
 ("BO runs share the players' first batch", f"mean best after 5: {get('Expert players', 5, 'mean'):.2f} and {get('BO from human starts', 5, 'mean'):.2f}"),
]
md = ["# Citation evidence log", "", "Generated by `scripts/verify_citations.py`, which checks that each quote occurs on the stated PDF page, ignoring spacing and punctuation. Page numbers are PDF pages. Quotes are kept short, so read the PDF for full context.", "",
      "A match shows the words are in the source. The verdict is a judgment about whether they support the sentence in our paper. 'Our reading' marks a gap we inferred.", "", f"Quotes checked: {len(rows)}. Not matched: {bad}.", "",
      "## Sources", "", "Papers are referenced by link and are not stored in this repository. The File column below names the local copy each quote was checked against.", "",
      "| Ref | Source | What was checked | Link |", "|---|---|---|---|",
      "| 1 | Shields, Nature 2021 | Full paper. Supplementary Information not read | https://doi.org/10.1038/s41586-021-03213-y |",
      "| 2 | EvML repository | README and notebook headings. Data in `data/raw/evml` | https://github.com/b-shields/EvML |",
      "| 3 | Gupta, EMNLP Findings 2025 | Full paper | https://doi.org/10.48550/arXiv.2509.21403 |",
      "| 4 | MacKnight, arXiv 2025 | Full paper | https://arxiv.org/abs/2509.00103 |",
      "| 5 | Muller, arXiv 2026 | Full paper | https://arxiv.org/abs/2609.04564 |",
      "| 6 | Cooper, Research Square 2026 | Full manuscript and SI | https://doi.org/10.21203/rs.3.rs-10032842/v1 |",
      "| 7 | Zhang et al. (SciSycEval), EMNLP Findings 2026 | Full paper | Title in the paper's reference list |",
      "| 8 | Cisse, Digital Discovery 2026 | Full paper. Supplementary Information not read | https://doi.org/10.1039/d5dd00520e |",
      "| 9 | Perera, Science 2018 | PubMed abstract only | https://doi.org/10.1126/science.aap9112 |",
      "| 10 | Zenodo 23953649 | README and record | https://zenodo.org/records/23953649 |",
      "| 11 | Das et al., JACS 2026 | Preprint and Open Reaction Database record. JACS version not read | https://doi.org/10.1021/jacs.6c05959 |", "",
      "## Claims and quotes", "", "| Ref | Statement in paper | File | Page | Quote | Verdict | Note | Check |", "|---|---|---|---|---|---|---|---|"]
md += ["| " + " | ".join(str(x).replace("|", "/") for x in r) + " |" for r in rows]
md += ["", "## Claims computed from data", "", "| Statement | Recomputed value |", "|---|---|"] + [f"| {a} | {v} |" for a, v in data]
(cit / "EVIDENCE.md").write_text("\n".join(md) + "\n")
print("quotes", len(rows), "not matched", bad)
for a, v in data:
    print(a, "=>", v)
