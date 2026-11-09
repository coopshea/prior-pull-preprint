"""Finite search worlds and the semantic conditions applied to them.

A world holds a canonical grid of true values. A semantic condition only changes
the labels shown for each canonical level, so the numbers stay fixed.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data" / "raw" / "shields"
SEMANTICS = ("S0", "S1", "S2", "S3")


@dataclass
class Factor:
    name: str
    levels: list[str]
    numeric: bool = False


@dataclass
class Transplant:
    context: str
    objective: str
    factors: list[Factor]


@dataclass
class World:
    id: str
    context: str
    objective: str
    factors: list[Factor]
    y: np.ndarray
    noise_sd: float
    transplant: Transplant
    domain: str = ""
    semantics: tuple[str, ...] = SEMANTICS
    _anon_seed: int = 7
    raw_lines: list[str] | None = None  # the source file as published, for recall probes
    rounds: int = 8
    source: str = ""
    replicates: np.ndarray | None = None  # measured duplicates, returned in turn instead of added noise

    @property
    def shape(self) -> tuple[int, ...]:
        return self.y.shape

    @property
    def size(self) -> int:
        return int(self.y.size)

    def marginals(self) -> list[np.ndarray]:
        out = []
        for ax in range(self.y.ndim):
            other = tuple(a for a in range(self.y.ndim) if a != ax)
            out.append(self.y.mean(axis=other))
        return out

    def hint_target(self) -> tuple[int, int, int]:
        """Return (factor, best level, worst level) for the factor with the largest spread."""
        marg = self.marginals()
        f = int(np.argmax([m.max() - m.min() for m in marg]))
        return f, int(np.argmax(marg[f])), int(np.argmin(marg[f]))


@dataclass
class Presentation:
    world: World
    semantic: str
    context: str
    objective: str
    names: list[str]
    labels: list[list[str]]
    display_order: list[list[int]] = field(default_factory=list)

    def label_to_index(self, f: int, label: str) -> int:
        return self.labels[f].index(label)

    def options(self, f: int) -> list[str]:
        return [self.labels[f][i] for i in self.display_order[f]]


def _anti_marginal(marg: np.ndarray) -> list[int]:
    order = np.argsort(-marg, kind="stable")
    n = len(order)
    pi = [0] * n
    for k, level in enumerate(order):
        pi[int(level)] = int(order[n - 1 - k])
    return pi


def present(world: World, semantic: str) -> Presentation:
    if semantic not in world.semantics:
        raise ValueError(f"{world.id} does not run {semantic}")
    n_f = len(world.factors)
    if semantic == "S0":
        names = [f.name for f in world.factors]
        labels = [list(f.levels) for f in world.factors]
        context, objective = world.context, world.objective
    elif semantic == "S1":
        rng = np.random.default_rng(world._anon_seed)
        names = [f"x{i + 1}" for i in range(n_f)]
        labels = []
        for f in world.factors:
            if f.numeric:
                labels.append([str(i + 1) for i in range(len(f.levels))])
            else:
                letters = [chr(ord("A") + i) for i in rng.permutation(len(f.levels))]
                labels.append(letters)
        context = "I'm trying to optimize a process with the settings below."
        objective = "output score (0 to 100)"
    elif semantic == "S2":
        names = [f.name for f in world.factors]
        labels = []
        for f, marg in zip(world.factors, world.marginals()):
            n = len(f.levels)
            pi = [n - 1 - i for i in range(n)] if f.numeric else _anti_marginal(marg)
            labels.append([f.levels[pi[i]] for i in range(n)])
        context, objective = world.context, world.objective
    else:
        t = world.transplant
        names = [f.name for f in t.factors]
        labels = [list(f.levels) for f in t.factors]
        context, objective = t.context, t.objective

    order = []
    for f_idx, f in enumerate(world.factors):
        if semantic == "S1":
            key = (lambda i: int(labels[f_idx][i])) if f.numeric else (lambda i: labels[f_idx][i])
            order.append(sorted(range(len(f.levels)), key=key))
        elif f.numeric:
            source = world.transplant.factors[f_idx].levels if semantic == "S3" else f.levels
            order.append(sorted(range(len(f.levels)), key=lambda i: source.index(labels[f_idx][i])))
        else:
            order.append(sorted(range(len(f.levels)), key=lambda i: labels[f_idx][i].lower()))
    return Presentation(world, semantic, context, objective, names, labels, order)


class Oracle:
    """Returns observed values. Noise is a pure function of world, seed, point, and replicate."""

    def __init__(self, world: World, seed: int, feedback: str = "F0"):
        self.world = world
        self.seed = seed
        self.feedback = feedback
        self.counts: dict[tuple[int, ...], int] = {}
        if feedback == "Fs":
            rng = np.random.default_rng(_stable_int(world.id, seed, "shuffle"))
            self._y = rng.permutation(world.y.ravel()).reshape(world.shape)
        elif feedback == "F0":
            self._y = world.y
        else:
            raise ValueError(feedback)

    def observe(self, idx: tuple[int, ...]) -> float:
        rep = self.counts.get(idx, 0)
        self.counts[idx] = rep + 1
        if self.world.replicates is not None and self.feedback == "F0":
            start = _stable_int(self.world.id, self.seed, idx, "first") % self.world.replicates.shape[-1]
            return round(float(self.world.replicates[idx][(start + rep) % self.world.replicates.shape[-1]]), 1)
        value = float(self._y[idx])
        if self.world.noise_sd > 0:
            rng = np.random.default_rng(_stable_int(self.world.id, self.seed, idx, rep))
            value = float(np.clip(value + rng.normal(0, self.world.noise_sd), 0, 100))
        return round(value, 1)  # measured values are returned as published, including any above 100


def _stable_int(*parts) -> int:
    return int(hashlib.sha256(repr(parts).encode()).hexdigest()[:16], 16)


def _normalize(y: np.ndarray) -> np.ndarray:
    return 100 * (y - y.min()) / (y.max() - y.min())


def load_shields() -> World:
    d = pd.read_csv(DATA / "experiment_index.csv")
    bases = pd.read_csv(DATA / "base-list.csv").set_index("Base_SMILES")["Base"].to_dict()
    ligands = pd.read_csv(DATA / "ligand-list.csv").set_index("Ligand_SMILES")["Ligand"].to_dict()
    solvents = pd.read_csv(DATA / "solvent-list.csv").set_index("Solvent_SMILES")["Solvent"].to_dict()
    d["Base"] = d["Base_SMILES"].map(bases)
    d["Ligand"] = d["Ligand_SMILES"].map(ligands)
    d["Solvent"] = d["Solvent_SMILES"].map(solvents)
    if d[["Base", "Ligand", "Solvent"]].isna().any().any():
        raise ValueError("unmapped SMILES in Shields data")

    base_levels = sorted(d["Base"].unique())
    lig_levels = sorted(d["Ligand"].unique())
    sol_levels = sorted(d["Solvent"].unique())
    conc_levels = sorted(d["Concentration"].unique())
    temp_levels = sorted(d["Temp_C"].unique())
    y = np.full((len(base_levels), len(lig_levels), len(sol_levels), len(conc_levels), len(temp_levels)), np.nan)
    for r in d.itertuples():
        y[
            base_levels.index(r.Base),
            lig_levels.index(r.Ligand),
            sol_levels.index(r.Solvent),
            conc_levels.index(r.Concentration),
            temp_levels.index(r.Temp_C),
        ] = d.at[r.Index, "yield"]
    if np.isnan(y).any():
        raise ValueError("Shields grid is incomplete")

    factors = [
        Factor("Base", base_levels),
        Factor("Ligand", lig_levels),
        Factor("Solvent", sol_levels),
        Factor("Concentration (M)", [str(c) for c in conc_levels], numeric=True),
        Factor("Temperature (°C)", [str(t) for t in temp_levels], numeric=True),
    ]
    roasts = ["Cinnamon", "New England", "American", "City", "City+", "Breakfast",
              "Full City", "Full City+", "Vienna", "French", "Italian", "Spanish"]
    transplant = Transplant(
        "I'm dialing in an espresso recipe for our cafe.",
        "tasting score (0 to 100)",
        [
            Factor("Bean origin", ["Brazil", "Colombia", "Ethiopia", "Sumatra"]),
            Factor("Roast", roasts),
            Factor("Water", ["filtered", "hard", "mineral", "soft"]),
            Factor("Dose (g)", ["16", "18", "20"], numeric=True),
            Factor("Brew temperature (°C)", ["90", "93", "96"], numeric=True),
        ],
    )
    return World(
        "shields",
        "I'm optimizing a palladium-catalyzed direct arylation reaction.",
        "reaction yield (%)",
        factors,
        y,
        0.0,
        transplant,
        domain="palladium catalysis",
        raw_lines=(DATA / "experiment_index.csv").read_text().splitlines(),
    )


def make_enzyme() -> World:
    ph = np.array([3, 4, 5, 6, 7, 8, 9], float)
    temp = np.array([5, 15, 25, 35, 45, 55, 65, 75, 85], float)
    sub = np.array([0.1, 0.3, 1, 3, 10, 30, 100, 300], float)
    cof = np.array([0.1, 0.3, 1.0, 0.75])  # Zn, Ca, Mg, Mn in level order below
    p, t, s = np.meshgrid(ph, temp, sub, indexing="ij")
    bell = np.exp(-(((p - 7.4) / 2.0) ** 2))
    thermal = np.exp(-(((t - 35) / 24) ** 2))
    haldane = s / (2.0 + s + s**2 / 50.0)
    base = bell * thermal * haldane
    y = _normalize(np.sqrt(base[..., None] * cof[None, None, None, :]))  # sqrt widens the peak so search has a gradient
    transplant = Transplant(
        "I'm developing a sourdough loaf for our bakery.",
        "loaf score (0 to 100)",
        [
            Factor("Hydration (%)", ["60", "65", "70", "75", "80", "85", "90"], numeric=True),
            Factor("Proof temperature (°C)", ["4", "8", "12", "16", "20", "24", "28", "32", "36"], numeric=True),
            Factor("Proof time (h)", ["1", "2", "3", "4", "6", "8", "12", "24"], numeric=True),
            Factor("Flour", ["bread", "einkorn", "rye", "spelt"]),
        ],
    )
    return World(
        "enzyme",
        "I'm tuning assay conditions for a human liver dehydrogenase.",
        "enzyme activity (relative, 0 to 100)",
        [
            Factor("pH", ["3", "4", "5", "6", "7", "8", "9"], numeric=True),
            Factor("Temperature (°C)", [str(int(v)) for v in temp], numeric=True),
            Factor("Substrate (mM)", ["0.1", "0.3", "1", "3", "10", "30", "100", "300"], numeric=True),
            Factor("Metal cofactor", ["Zn2+", "Ca2+", "Mg2+", "Mn2+"]),
        ],
        y,
        3.0,
        transplant,
        domain="enzymology",
    )


def make_resin() -> World:
    monomers = ["HDDA", "IBOA", "PEGDA-250", "TPGDA"]
    viscosity = np.array([7.0, 9.0, 18.0, 14.0])
    reactivity = np.array([1.0, 0.55, 0.85, 0.75])
    pi = np.array([0.25, 0.5, 1, 1.5, 2, 3, 4, 6], float)
    absorber = np.array([0, 0.02, 0.05, 0.1, 0.2, 0.4], float)
    exposure = np.array([1, 2, 4, 8, 16, 32], float)
    m, i, a, e = np.meshgrid(np.arange(4), pi, absorber, exposure, indexing="ij")
    dp = 1.0 / (0.004 + 0.0012 * i + 0.09 * a)  # penetration depth, um
    ec = 6.0 / (reactivity[m] * (0.3 + i))  # critical energy
    cd = dp * np.log(np.maximum(5.0 * e / ec, 1.0001))  # Jacobs working curve
    depth = np.exp(-(((cd - 160.0) / 80.0) ** 2))
    resolution = 1.0 / (1.0 + (dp / 45.0) ** 4)  # over-penetration blurs features
    brittle = np.exp(-((i / 3.5) ** 2))  # excess initiator embrittles and yellows
    speed = 1.0 / (1.0 + e / 40.0)
    flow = 1.0 / (1.0 + (viscosity[m] * (1 + 0.04 * i) / 30.0) ** 2)
    y = _normalize(depth * resolution * speed * flow * brittle)
    transplant = Transplant(
        "I'm tuning growing conditions for tomatoes in our greenhouse.",
        "harvest score (0 to 100)",
        [
            Factor("Cultivar", ["Celebrity", "Moneymaker", "Roma", "San Marzano"]),
            Factor("Fertilizer EC (mS/cm)", ["0.5", "1", "1.5", "2", "2.5", "3", "4", "5"], numeric=True),
            Factor("Shade cloth (%)", ["0", "10", "20", "30", "40", "60"], numeric=True),
            Factor("Supplemental light (h)", ["0", "2", "4", "6", "8", "12"], numeric=True),
        ],
    )
    return World(
        "resin",
        "I'm formulating an acrylate resin for a 405 nm DLP printer, printing 100 micron layers.",
        "print quality score (0 to 100)",
        [
            Factor("Monomer", monomers),
            Factor("Photoinitiator TPO (wt%)", ["0.25", "0.5", "1", "1.5", "2", "3", "4", "6"], numeric=True),
            Factor("UV absorber (wt%)", ["0", "0.02", "0.05", "0.1", "0.2", "0.4"], numeric=True),
            Factor("Exposure (s)", ["1", "2", "4", "8", "16", "32"], numeric=True),
        ],
        y,
        3.0,
        transplant,
        domain="photopolymer formulation",
    )


def make_gmm(seed: int = 11) -> World:
    rng = np.random.default_rng(seed)
    n = (8, 8, 8, 4)
    grids = np.meshgrid(*[np.linspace(0, 1, k) for k in n[:3]], indexing="ij")
    surface = np.zeros((8, 8, 8))
    for _ in range(7):
        mu = rng.uniform(0.05, 0.95, 3)
        width = rng.uniform(0.12, 0.3)
        amp = rng.uniform(0.3, 1.0)
        surface += amp * np.exp(-sum((g - m) ** 2 for g, m in zip(grids, mu)) / (2 * width**2))
    cat = rng.uniform(0.4, 1.0, 4)
    y = _normalize(surface[..., None] * cat[None, None, None, :])
    levels = [str(i + 1) for i in range(8)]
    transplant = Transplant(
        "I'm working out a cold brew recipe.",
        "tasting score (0 to 100)",
        [
            Factor("Steep time (h)", ["4", "6", "8", "10", "12", "16", "20", "24"], numeric=True),
            Factor("Grind setting", levels, numeric=True),
            Factor("Coffee to water (g per L)", ["50", "60", "70", "80", "90", "100", "120", "140"], numeric=True),
            Factor("Bean", ["Brazil", "Colombia", "Ethiopia", "Sumatra"]),
        ],
    )
    return World(
        "gmm",
        "I'm trying to optimize a process with the settings below.",
        "output score (0 to 100)",
        [
            Factor("x1", levels, numeric=True),
            Factor("x2", levels, numeric=True),
            Factor("x3", levels, numeric=True),
            Factor("x4", ["A", "B", "C", "D"]),
        ],
        y,
        3.0,
        transplant,
        domain="process optimization",
        semantics=("S1", "S3"),
    )


def load_efb() -> World:
    """Oil palm delignification, 64 conditions in duplicate. Zenodo 23953649, posted 2026-11-04, CC-BY-4.0."""
    d = pd.read_csv(DATA.parent / "efb" / "datos_originales.csv")
    temps = sorted(d["TEMPERATURA"].unique())
    loads = sorted(d["CARGA"].unique())
    times = sorted(d["TIEMPO"].unique())
    reps = np.full((len(temps), len(loads), len(times), 2), np.nan)
    for (t, c, m), g in d.groupby(["TEMPERATURA", "CARGA", "TIEMPO"]):
        if len(g) != 2:
            raise ValueError("EFB condition without exactly two replicates")
        reps[temps.index(t), loads.index(c), times.index(m)] = 100 * g["DeligYield"].to_numpy()
    if np.isnan(reps).any():
        raise ValueError("EFB grid is incomplete")
    transplant = Transplant(
        "I'm tuning a slow-cooked stock for our restaurant.",
        "flavor score (0 to 100)",
        [
            Factor("Simmer temperature (°C)", ["80", "85", "90", "95"], numeric=True),
            Factor("Salt (g per L)", ["5", "10"], numeric=True),
            Factor("Cook time (h)", ["2", "4", "6", "8", "10", "12", "14", "16"], numeric=True),
        ],
    )
    return World(
        "efb",
        "I'm optimizing subcritical alkaline delignification of oil palm empty fruit bunches with NaOH in a stirred batch reactor.",
        "lignin removal (%)",
        [
            Factor("Temperature (°C)", [str(int(v)) for v in temps], numeric=True),
            Factor("NaOH loading (wt%)", [str(int(v)) for v in loads], numeric=True),
            Factor("Reaction time (min)", [str(int(v)) for v in times], numeric=True),
        ],
        reps.mean(axis=-1),
        0.0,
        transplant,
        domain="biomass pretreatment",
        rounds=4,
        replicates=reps,
    )


def load_suzuki() -> World:
    d = pd.read_csv(DATA.parent / "suzuki" / "experiment_index.csv")
    cols = [("Electrophile_SMILES", "Electrophile"), ("Nucleophile_SMILES", "Nucleophile"), ("Ligand_SMILES", "Ligand"),
            ("Base_SMILES", "Base"), ("Solvent_SMILES", "Solvent")]
    levels = [sorted(d[c].astype(str).unique()) for c, _ in cols]
    y = np.full([len(v) for v in levels], np.nan)
    index = [d[c].astype(str).map({v: i for i, v in enumerate(lv)}).to_numpy() for (c, _), lv in zip(cols, levels)]
    y[tuple(index)] = d["yield"].to_numpy()
    if np.isnan(y).any():
        raise ValueError("Suzuki grid is incomplete")
    return World(
        "suzuki", "I'm optimizing a palladium-catalyzed Suzuki-Miyaura cross-coupling.", "reaction yield (%)",
        [Factor(name, lv) for (_, name), lv in zip(cols, levels)], y, 0.0, None, domain="palladium catalysis", source="suzuki",
        raw_lines=(DATA.parent / "suzuki" / "experiment_index.csv").read_text().splitlines(),
    )


def load_cernak() -> list[World]:
    """One world per plate and temperature with a complete, uniquely named ligand, catalyst, and base grid."""
    d = pd.read_csv(DATA.parent / "cernak" / "long.csv")
    out = []
    for (exp, temp), b in d.groupby(["experiment", "temp"]):
        if b.groupby(["Lig", "Cat", "Base"]).size().max() != 1:
            continue
        levels = [sorted(b[c].unique()) for c in ("Lig", "Cat", "Base")]
        if np.prod([len(v) for v in levels]) != len(b):
            continue
        y = np.full([len(v) for v in levels], np.nan)
        for c_i, (lig, cat, base, val) in enumerate(zip(b["Lig"], b["Cat"], b["Base"], b["y"])):
            y[levels[0].index(lig), levels[1].index(cat), levels[2].index(base)] = val
        solvents = " or ".join(sorted(b["Solv"].unique()))
        out.append(World(
            f"cernak_{exp}_{int(temp)}",
            f"I'm optimizing a C-N coupling of {b['Elec'].iloc[0]} with {b['Nuc'].iloc[0]} in {solvents} at {int(temp)} °C for 18 hours.",
            "assay yield (%)",
            [Factor("Ligand", levels[0]), Factor("Catalyst", levels[1]), Factor("Base", levels[2])],
            y, 0.0, None, domain="C-N cross-coupling", source="cernak",
        ))
    return out


def make_surface(i: int) -> World:
    """A meaningless smooth surface. Shape, peaks, and noise come only from the instance number."""
    rng = np.random.default_rng(_stable_int("surface", i))
    k = int(rng.integers(3, 5))  # 3 or 4 factors keeps grids small enough for the BayBE baseline
    shape = tuple(int(v) for v in rng.integers(4, 11, size=k))
    grids = np.meshgrid(*[np.linspace(0, 1, n) for n in shape], indexing="ij")
    y = np.zeros(shape)
    for _ in range(int(rng.integers(2, 11))):
        mu, width, amp = rng.uniform(0, 1, k), rng.uniform(0.05, 0.4), rng.uniform(0.1, 1.0)
        y += amp * np.exp(-sum((g - m) ** 2 for g, m in zip(grids, mu)) / (2 * width**2))
    factors = [Factor(f"x{j + 1}", [str(v + 1) for v in range(n)], numeric=True) for j, n in enumerate(shape)]
    noise = float(rng.choice([0.0, 3.0, 10.0]))
    return World(f"surface_{i}", "I'm trying to optimize a process with the settings below.", "output score (0 to 100)",
                 factors, _normalize(y), noise, None, domain="process optimization", source="surface")


def load_survey(n_surfaces: int = 30) -> dict[str, World]:
    shields, efb = load_shields(), load_efb()
    shields.source, efb.source = "shields", "efb"
    worlds = [shields, efb, load_suzuki(), *load_cernak(), *(make_surface(i) for i in range(n_surfaces))]
    return {w.id: w for w in worlds}


def load_all() -> dict[str, World]:
    return {w.id: w for w in (load_shields(), make_enzyme(), make_resin(), make_gmm(), load_efb())}
