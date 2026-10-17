"""The recovery measurement study's pure parts (configs/analysis/recovery_study.yaml;
docs/runs/2026-10-1x_recovery_study.md). Arrays in, numbers out; no simulator, no files.

* :func:`perturbation_bank` / :func:`effective` - the per-query perturbations of the executed prefix and
  their controller-relevant effect (clipped position/rotation change, the gripper's sign flips, the
  controller no-op flag).
* :func:`query_steps` - the replans nearest the rule's phases of a successful rollout.
* :func:`Table` - one row per continuation; :func:`perturbed` joins each perturbed continuation with its
  paired nominal continuation (same rollout, query and seed).
* :func:`s20_cells` - Q1: cross-fitted concentration over query x group cells, equally weighted held-out
  cell rates (injuries over at-risk effective trials), fractional ties.
* :func:`q2_crossfit` - Q2: ridge logistic models U (nominal failure + a fold-wise restricted cubic
  spline in log1p(size)) and R (U + gripper flip categories + phase x group), leave-one-rollout-out.
* :func:`bootstrap_task` - the full-procedure cluster bootstrap (rollouts resampled; Q1 re-ranked; U and R
  refitted with every copy of a rollout held out together).
* :func:`usable`, :func:`decide` - usability per regime and GO / NO-GO / INCONCLUSIVE.
* :func:`simulate_world` / :func:`simulate` - the pre-run validation on synthetic clustered worlds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

GROUPS = ("position", "rotation", "gripper")
DIMS = {"position": (0, 3), "rotation": (3, 6), "gripper": (6, 7)}
ALL_GROUPS = GROUPS + ("isotropic",)
GRIP = 6


# ---- the perturbations ----------------------------------------------------------------------------


def perturbation_bank(sigma: np.ndarray, execute_k: int, magnitudes, n_isotropic_per_magnitude: int,
                      seed: int) -> list[dict]:
    """The rule's perturbations of one query's executed prefix (``execute_k`` x 7 raw actions):
    per structured group, a fixed random unit direction within the group, at each sign and magnitude
    (m sigma, constant over the prefix); and isotropic directions over the whole prefix at the
    structured perturbations' mean raw norm per magnitude."""
    sigma = np.asarray(sigma, dtype=np.float64)
    rng = np.random.default_rng(seed)
    bank = []
    for g in GROUPS:
        lo, hi = DIMS[g]
        u = rng.normal(size=hi - lo)
        u = u / np.linalg.norm(u)
        for m in magnitudes:
            for sign in (1, -1):
                delta = np.zeros((execute_k, 7))
                delta[:, lo:hi] = sign * float(m) * sigma[lo:hi] * u
                bank.append({"kind": "structured", "group": g, "sign": sign, "magnitude": float(m), "delta": delta})
    for m in magnitudes:
        target = np.mean([np.linalg.norm(b["delta"]) for b in bank if b["magnitude"] == float(m)])
        for r in range(int(n_isotropic_per_magnitude)):
            v = rng.normal(size=(execute_k, 7))
            bank.append({"kind": "isotropic", "group": "isotropic", "sign": 0, "magnitude": float(m),
                         "delta": v / np.linalg.norm(v) * target, "index": r})
    return bank


def effective(chunk_prefix: np.ndarray, delta: np.ndarray, sigma: np.ndarray) -> dict:
    """The controller-relevant effect of ``delta`` on the clipped prefix: the change after clipping in
    sigma units (all dims: ``size``; position+rotation: ``posrot_size``), the gripper's sign flips (its
    controller sees only ``sign(action)``), the clipped fraction, and ``noop`` (no position/rotation
    change and no flip: identical controller inputs)."""
    a = np.asarray(chunk_prefix, dtype=np.float64)
    base = np.clip(a, -1.0, 1.0)
    pert = np.clip(a + delta, -1.0, 1.0)
    eff = pert - base
    sigma = np.asarray(sigma, dtype=np.float64)
    flips = int(np.sum(np.sign(pert[:, GRIP]) != np.sign(base[:, GRIP])))
    posrot = eff[:, :GRIP]
    noop = bool(np.all(posrot == 0.0) and flips == 0)
    changed = np.abs(delta) > 0
    clipped = float(np.mean(np.abs(eff[changed] - delta[changed]) > 1e-12)) if changed.any() else 0.0
    return {"size": float(np.linalg.norm(eff / sigma)), "posrot_size": float(np.linalg.norm(posrot / sigma[:GRIP])),
            "flips": flips, "clipped_fraction": clipped, "noop": noop}


def query_steps(replan_steps: list[int], length: int, phases) -> list[tuple[int, int]]:
    """``(phase_index, step)``: per phase, the replan step nearest ``phase * length``; a step already
    taken by an earlier phase is dropped (recorded by the caller)."""
    out, taken = [], set()
    steps = np.asarray(sorted(replan_steps))
    for i, ph in enumerate(phases):
        s = int(steps[np.argmin(np.abs(steps - ph * length))])
        if s in taken:
            continue
        taken.add(s)
        out.append((i, s))
    return out


# ---- the outcome table ---------------------------------------------------------------------------


@dataclass
class Table:
    """One row per continuation of one task. ``kind`` 0 = nominal continuation, 1 = perturbed;
    ``group`` indexes ALL_GROUPS (-1 for nominal)."""

    rollout: np.ndarray
    query: np.ndarray        # phase bin
    seed: np.ndarray
    kind: np.ndarray
    group: np.ndarray
    success: np.ndarray
    size: np.ndarray
    flips: np.ndarray
    noop: np.ndarray

    @classmethod
    def from_rows(cls, rows: list[dict]) -> "Table":
        def col(key, dtype):
            return np.asarray([r[key] for r in rows], dtype=dtype)
        return cls(col("rollout", np.int64), col("query", np.int64), col("seed", np.int64), col("kind", np.int64),
                   col("group", np.int64), col("success", bool), col("size", np.float64), col("flips", np.int64),
                   col("noop", bool))

    def take(self, idx: np.ndarray, rollout_labels: np.ndarray | None = None) -> "Table":
        t = Table(*(getattr(self, f)[idx] for f in self.__dataclass_fields__))
        if rollout_labels is not None:
            t.rollout = rollout_labels
        return t


def perturbed(t: Table) -> dict:
    """Perturbed continuations joined with their paired nominal continuation's success."""
    nominal = {(r, q, s): ok for r, q, s, k, ok in zip(t.rollout, t.query, t.seed, t.kind, t.success) if k == 0}
    idx = np.flatnonzero(t.kind == 1)
    nom = np.asarray([nominal[(t.rollout[i], t.query[i], t.seed[i])] for i in idx], dtype=bool)
    fail = ~t.success[idx]
    return {"idx": idx, "nominal_success": nom, "fail": fail, "injury": nom & fail, "rescue": ~nom & t.success[idx],
            "rollout": t.rollout[idx], "query": t.query[idx], "seed": t.seed[idx], "group": t.group[idx],
            "size": t.size[idx], "flips": t.flips[idx], "noop": t.noop[idx]}


# ---- Q1: concentration over query x group cells ----------------------------------------------------


def _top_share(rank_values: np.ndarray, held: np.ndarray, frac: float) -> float:
    """Sum of ``held`` over the top ``frac`` of cells ranked by ``rank_values`` (ties at the boundary get
    the expected share under random tie-breaking)."""
    n = len(rank_values)
    k = frac * n
    order = np.argsort(-rank_values, kind="stable")
    v = rank_values[order]
    h = held[order]
    total, taken, i = 0.0, 0.0, 0
    while i < n and taken < k - 1e-12:
        j = i
        while j < n and v[j] == v[i]:
            j += 1
        size = j - i
        share = min(1.0, (k - taken) / size)
        total += share * float(h[i:j].sum())
        taken += share * size
        i = j
    return total


def cell_rates(p: dict, keep: np.ndarray, seeds) -> dict:
    """Per (rollout, query, structured group) cell: injuries / at-risk effective trials over ``seeds``."""
    out = {}
    m = keep & np.isin(p["seed"], list(seeds)) & p["nominal_success"]
    for r, q, g, inj in zip(p["rollout"][m], p["query"][m], p["group"][m], p["injury"][m]):
        c = out.setdefault((int(r), int(q), int(g)), [0, 0])
        c[0] += int(inj)
        c[1] += 1
    return {k: v[0] / v[1] for k, v in out.items() if v[1] > 0}


def s20_cells(p: dict, frac: float = 0.2, halves=((0, 1), (2, 3)), by: str = "cell") -> float | None:
    """Cross-fitted S20: rank cells by their rate on one seed half, take the top ``frac``'s share of the
    other half's (equally weighted) cell rates, swap, average. Cells: effective structured perturbations
    (``by``: "cell" = rollout x query x group; "query" = rollout x query, the temporal version).
    None if either held-out total is zero or no cell has exposure in both halves."""
    keep = (~p["noop"]) & (p["group"] < len(GROUPS))
    if by == "query":
        p = {**p, "group": np.zeros_like(p["group"])}
    a, b = cell_rates(p, keep, halves[0]), cell_rates(p, keep, halves[1])
    cells = sorted(set(a) & set(b))
    if not cells:
        return None
    ra, rb = np.asarray([a[c] for c in cells]), np.asarray([b[c] for c in cells])
    vals = []
    for rank, held in ((ra, rb), (rb, ra)):
        if held.sum() <= 0:
            return None
        vals.append(_top_share(rank, held, frac) / held.sum())
    return float(np.mean(vals))


def group_shares(p: dict) -> dict:
    """Descriptive: injury rate per group among at-risk effective trials."""
    keep = (~p["noop"]) & p["nominal_success"]
    out = {}
    for gi, g in enumerate(ALL_GROUPS):
        m = keep & (p["group"] == gi)
        out[g] = {"trials": int(m.sum()), "injury_rate": float(p["injury"][m].mean()) if m.any() else None}
    return out


# ---- Q2: beyond magnitude -------------------------------------------------------------------------


def rcs_knots(x_positive: np.ndarray) -> np.ndarray | None:
    """Boundary knots at the min and max and interior knots at the 25/50/75% quantiles of the positive
    values; None (the linear fallback) unless all five are distinct."""
    x = np.asarray(x_positive, dtype=np.float64)
    if x.size < 5:
        return None
    k = np.concatenate([[x.min()], np.quantile(x, [0.25, 0.5, 0.75]), [x.max()]])
    return k if np.all(np.diff(k) > 1e-12) else None


def rcs_basis(x: np.ndarray, knots: np.ndarray | None) -> np.ndarray:
    """Harrell's restricted (natural) cubic spline: x and K - 2 nonlinear terms; linear if no knots."""
    x = np.asarray(x, dtype=np.float64)
    if knots is None:
        return x[:, None]
    t = knots
    K = len(t)
    cols = [x]
    scale = (t[-1] - t[0]) ** 2
    pos = lambda z: np.maximum(z, 0.0) ** 3   # noqa: E731
    for j in range(K - 2):
        cols.append((pos(x - t[j]) - pos(x - t[K - 2]) * (t[K - 1] - t[j]) / (t[K - 1] - t[K - 2])
                     + pos(x - t[K - 1]) * (t[K - 2] - t[j]) / (t[K - 1] - t[K - 2])) / scale)
    return np.column_stack(cols)


def designs(p: dict, sel: np.ndarray, knots, n_phase: int) -> tuple[np.ndarray, np.ndarray]:
    """U: intercept, nominal failure, the spline in log1p(size). R: U + gripper flip categories {1-3, 4-8}
    + phase x group indicators."""
    x = np.log1p(p["size"][sel])
    U = np.column_stack([np.ones(sel.sum()), (~p["nominal_success"][sel]).astype(float), rcs_basis(x, knots)])
    f = p["flips"][sel]
    extra = [((f >= 1) & (f <= 3)).astype(float), (f >= 4).astype(float)]
    q, g = p["query"][sel], p["group"][sel]
    for qi in range(n_phase):
        for gi in range(len(ALL_GROUPS)):
            extra.append(((q == qi) & (g == gi)).astype(float))
    return U, np.column_stack([U] + extra)


def fit_ridge_logistic(X: np.ndarray, y: np.ndarray, ridge: float, max_iter: int = 100, tol: float = 1e-8,
                       start: np.ndarray | None = None) -> np.ndarray:
    """Bernoulli logistic regression by Newton's method, ridge on every coefficient but the intercept
    (column 0). The ridge makes the objective strictly convex, so it converges."""
    beta = np.zeros(X.shape[1]) if start is None else start.copy()
    pen = np.full(X.shape[1], ridge)
    pen[0] = 1e-8
    y = y.astype(np.float64)
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-eta))
        grad = X.T @ (y - mu) - pen * beta
        hess = (X * (mu * (1 - mu))[:, None]).T @ X + np.diag(pen)
        step = np.linalg.solve(hess, grad)
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    return beta


def _logloss(X, beta, y):
    p = np.clip(1.0 / (1.0 + np.exp(-np.clip(X @ beta, -30, 30))), 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def q2_crossfit(p: dict, folds: np.ndarray, ridge: float, n_phase: int) -> tuple[float | None, np.ndarray]:
    """Leave-one-fold-out (folds = original rollout ids; copies stay together): the mean per-observation
    log-loss improvement of R over U on effective perturbations, and the per-observation improvements."""
    sel = ~p["noop"]
    y = p["fail"][sel].astype(float)
    fold = folds[sel]
    d = np.full(sel.sum(), np.nan)
    for f in np.unique(fold):
        tr, te = fold != f, fold == f
        if tr.sum() == 0 or te.sum() == 0:
            continue
        x_tr = np.log1p(p["size"][sel][tr])
        knots = rcs_knots(x_tr[x_tr > 0])
        U, R = designs(p, sel, knots, n_phase)
        bu = fit_ridge_logistic(U[tr], y[tr], ridge)
        br = fit_ridge_logistic(R[tr], y[tr], ridge)
        d[te] = _logloss(U[te], bu, y[te]) - _logloss(R[te], br, y[te])
    ok = np.isfinite(d)
    return (float(d[ok].mean()) if ok.any() else None), d


# ---- the full-procedure cluster bootstrap ------------------------------------------------------------


def _resample(t: Table, rng: np.random.Generator) -> tuple[Table, np.ndarray]:
    """Rollouts with replacement; each copy is relabelled (a distinct cell for Q1) but keeps its original
    id as the Q2 fold, so all copies of a rollout are held out together."""
    ids = np.unique(t.rollout)
    draw = rng.choice(ids, size=len(ids), replace=True)
    parts, labels, folds = [], [], []
    for c, r in enumerate(draw):
        idx = np.flatnonzero(t.rollout == r)
        parts.append(idx)
        labels.append(np.full(len(idx), c))
        folds.append(np.full(len(idx), r))
    idx = np.concatenate(parts)
    return t.take(idx, np.concatenate(labels)), np.concatenate(folds)


def _halves(rule: dict):
    return tuple(tuple(h) for h in rule["q1"].get("halves", ((0, 1), (2, 3))))


def task_point(t: Table, rule: dict) -> dict:
    p = perturbed(t)
    s20 = s20_cells(p, float(rule["q1"]["top_fraction"]), _halves(rule))
    q2, _ = q2_crossfit(p, p["rollout"], float(rule["q2"]["ridge"]), int(rule["n_phase"]))
    return {"s20": s20, "q2": q2}


def bootstrap_task(t: Table, rule: dict, n_boot: int, seed: int, need_q2=lambda s20: True) -> dict:
    """Resampled S20 and Q2 (refitted every resample). ``need_q2`` is a hook for skipping Q2 (unused:
    production and the validation compute both)."""
    rng = np.random.default_rng(seed)
    s20s, q2s = [], []
    for _ in range(int(n_boot)):
        tb, folds = _resample(t, rng)
        pb = perturbed(tb)
        s = s20_cells(pb, float(rule["q1"]["top_fraction"]), _halves(rule))
        s20s.append(np.nan if s is None else s)
        if need_q2(s):
            q, _ = q2_crossfit(pb, folds[pb["idx"]], float(rule["q2"]["ridge"]), int(rule["n_phase"]))
            q2s.append(np.nan if q is None else q)
        else:
            q2s.append(np.nan)
    return {"s20": np.asarray(s20s), "q2": np.asarray(q2s)}


def interval(samples: np.ndarray, level: float):
    s = np.asarray(samples, dtype=np.float64)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return None
    a = (1 - level) / 2
    return [float(np.quantile(s, a)), float(np.quantile(s, 1 - a))]


# ---- usability and the outcome ---------------------------------------------------------------------


def usable(t: Table, n_success_rollouts: int, regime: dict) -> dict:
    p = perturbed(t)
    keep = ~p["noop"]
    inj = p["injury"] & keep
    n_queries = len({(int(r), int(q)) for r, q in zip(t.rollout, t.query)})
    rollouts_with_injury = len(set(p["rollout"][inj].tolist()))
    checks = {"rollouts": n_success_rollouts >= int(regime["min_rollouts"]),
              "queries": n_queries >= int(regime["min_queries"]),
              "injuries": int(inj.sum()) >= int(regime["min_injuries"]),
              "injury_rollouts": rollouts_with_injury >= int(regime["min_injury_rollouts"])}
    return {"usable": all(checks.values()), "checks": checks, "n_rollouts": n_success_rollouts,
            "n_queries": n_queries, "injuries": int(inj.sum()), "injury_rollouts": rollouts_with_injury}


def task_verdict(point: dict, boot: dict, rule: dict, level: float) -> dict:
    s_ci, q_ci = interval(boot["s20"], level), interval(boot["q2"], level)
    q1 = point["s20"] is not None and s_ci is not None and point["s20"] >= float(rule["q1"]["min_s20"]) \
        and s_ci[0] > float(rule["q1"]["uniform"])
    q2 = point["q2"] is not None and q_ci is not None and q_ci[0] > 0
    diffuse = s_ci is not None and s_ci[1] < float(rule["q1"]["min_s20"])
    no_gain = q_ci is not None and q_ci[1] <= 0
    return {"s20": point["s20"], "s20_ci": s_ci, "q2": point["q2"], "q2_ci": q_ci, "q1_pass": bool(q1),
            "q2_pass": bool(q2), "pass": bool(q1 and q2), "clear_fail": bool(diffuse or no_gain)}


def decide(tasks: dict[str, dict], gate_ok: bool, negative_control_ok: bool) -> str:
    """GO / NO-GO / INCONCLUSIVE from per-task {"usable", "verdict"}."""
    if not gate_ok or not negative_control_ok:
        return "INCONCLUSIVE"
    use = [v for v in tasks.values() if v["usable"]]
    passing = [v for v in use if v["verdict"]["pass"]]
    if len(passing) >= 2:
        return "GO"
    failing = [v for v in use if not v["verdict"]["pass"]]
    if len(use) >= 2 and all(v["verdict"]["clear_fail"] for v in failing):
        return "NO-GO"
    return "INCONCLUSIVE"


def clopper_pearson(k: int, n: int, level: float = 0.95) -> list[float]:
    from scipy.stats import beta

    a = (1 - level) / 2
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))
    return [lo, hi]


# ---- the pre-run validation -------------------------------------------------------------------------


def simulate_world(spec: dict, rng: np.random.Generator) -> list[dict]:
    """One synthetic task's continuations, in the production layout (structured: 3 groups x 2 signs x 2
    magnitudes; isotropic: 2 x 2; M seeds; nominal continuations). ``spec["kind"]``:

    * ``null_uniform`` - one injury risk for every trial (no rollout, query, phase, group or size effect);
    * ``null_magnitude`` - risk a nonlinear function of size, with group-dependent size distributions and
      random rollout x query effects, but no systematic phase or group effect;
    * ``phase_only`` / ``component_only`` (the gripper's flips harmful at every phase) / ``mixed``.
    """
    n_q, n_seeds = int(spec["n_queries"]), int(spec["n_seeds"])
    n_roll = int(rng.binomial(int(spec["n_rollouts"]), float(spec["p_success"])))
    kind = spec["kind"]
    base = float(spec.get("base_logit", -2.5))
    rows = []
    for r in range(n_roll):
        u_r = rng.normal(0, 0.5) if kind != "null_uniform" else 0.0   # uniform: no heterogeneity at all
        for q in range(n_q):
            u_q = rng.normal(0, 0.7) if kind != "null_uniform" else 0.0
            for m in range(n_seeds):
                nominal_fail = rng.random() < 0.05
                rows.append({"rollout": r, "query": q, "seed": m, "kind": 0, "group": -1,
                             "success": not nominal_fail, "size": 0.0, "flips": 0, "noop": False})
                pert = [(gi, mag) for gi in range(3) for mag in (1, 3) for _ in (0, 1)] + \
                       [(3, mag) for mag in (1, 3) for _ in (0, 1)]
                for gi, mag in pert:
                    if gi == 2:   # gripper: 1 sigma mostly flips nothing (a no-op), 3 sigma flips
                        flips = int(rng.binomial(8, 0.05 if mag == 1 else 0.6))
                        size = float(mag * (2.0 + rng.random())) if flips else float(mag * 1.5)
                    else:
                        flips = int(rng.binomial(8, 0.3)) if gi == 3 and mag == 3 else 0
                        spread = {0: 1.0, 1: 0.6, 3: 1.4}[gi] if kind == "null_magnitude" else 1.0
                        size = float(mag * math.sqrt(8) * spread * (0.8 + 0.4 * rng.random()))
                    noop = gi == 2 and flips == 0
                    logit = base + u_r
                    if kind != "null_uniform":
                        logit += u_q
                    if kind in ("null_magnitude", "phase_only", "component_only", "mixed"):
                        logit += 1.2 * np.tanh((size - 6.0) / 3.0)
                    if kind in ("phase_only", "mixed") and q in spec.get("critical_phases", (2, 3)):
                        logit += float(spec.get("effect", 2.0))
                    if kind in ("component_only", "mixed") and gi == 2 and flips > 0:
                        logit += float(spec.get("effect", 2.0))
                    p_fail = 1 / (1 + math.exp(-logit))
                    if noop:
                        fail = nominal_fail          # identical controller inputs: the nominal outcome
                    else:
                        fail = nominal_fail or (rng.random() < p_fail)
                    rows.append({"rollout": r, "query": q, "seed": m, "kind": 1, "group": gi, "success": not fail,
                                 "size": size, "flips": flips, "noop": noop})
    return rows


def simulate_once(spec: dict, rule: dict, regime: dict, n_tasks: int, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    tasks = {}
    for k in range(n_tasks):
        rows = simulate_world(spec, rng)
        if not rows:
            tasks[f"t{k}"] = {"usable": False, "verdict": {"pass": False, "clear_fail": False}}
            continue
        t = Table.from_rows(rows)
        u = usable(t, len(np.unique(t.rollout)), regime)
        if not u["usable"]:
            tasks[f"t{k}"] = {"usable": False, "verdict": {"pass": False, "clear_fail": False}}
            continue
        point = task_point(t, rule)
        boot = bootstrap_task(t, rule, n_boot, seed * 31 + k)
        verdicts = {lvl: task_verdict(point, boot, rule, lvl) for lvl in (0.95, 0.99)}
        tasks[f"t{k}"] = {"usable": True, "verdicts": verdicts}
    out = {}
    for lvl in (0.95, 0.99):
        view = {k: {"usable": v["usable"], "verdict": v["verdicts"][lvl] if v["usable"] else v["verdict"]}
                for k, v in tasks.items()}
        out[str(lvl)] = decide(view, True, True)
    return out


def simulate(spec: dict, rule: dict, regime: dict, n_worlds: int, n_tasks: int, n_boot: int, seed: int,
             workers: int = 1) -> dict:
    """Decision rates over ``n_worlds`` synthetic studies, with Clopper-Pearson intervals."""
    args = [(spec, rule, regime, n_tasks, n_boot, seed * 100003 + w) for w in range(int(n_worlds))]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=int(workers)) as ex:
            results = list(ex.map(_simulate_star, args))
    else:
        results = [simulate_once(*a) for a in args]
    summary = {}
    for lvl in ("0.95", "0.99"):
        counts = {o: sum(r[lvl] == o for r in results) for o in ("GO", "NO-GO", "INCONCLUSIVE")}
        summary[lvl] = {**counts, "go_rate": counts["GO"] / len(results),
                        "go_rate_cp95": clopper_pearson(counts["GO"], len(results))}
    return {"spec": spec, "n_worlds": int(n_worlds), "n_boot": int(n_boot), "rates": summary}


def _simulate_star(a):
    return simulate_once(*a)


# ---- E1-B's inputs (configs/analysis/twosided_angleb.yaml) -------------------------------------------


def phase_group_weights(rows: list[dict], n_bins: int, magnitude: float) -> dict:
    """The recovery map as E1-B's metric weights. Per phase bin (the query index) and structured group: the
    injury rate among effective (not a controller no-op), at-risk (the paired nominal continuation
    succeeded) perturbations at ``magnitude``, pooled over rollouts, seeds and signs. A cell without such
    trials takes its group's rate over all bins (``pooled``, also the phase-pooled variant's weights)."""
    nominal = {(r["rollout"], r["query"], r["seed"]): bool(r["success"]) for r in rows if r["kind"] == 0}
    trials = np.zeros((int(n_bins), len(GROUPS)), dtype=np.int64)
    injuries = np.zeros_like(trials)
    for r in rows:
        if r["kind"] != 1 or not 0 <= r["group"] < len(GROUPS) or r["noop"]:
            continue
        if float(r["magnitude"]) != float(magnitude) or not nominal[(r["rollout"], r["query"], r["seed"])]:
            continue
        b = int(r["query"])
        if not 0 <= b < n_bins:
            raise ValueError(f"query {b} is outside the {n_bins} phase bins")
        trials[b, r["group"]] += 1
        injuries[b, r["group"]] += int(not r["success"])
    total = trials.sum(axis=0)
    if (total == 0).any():
        empty = [GROUPS[g] for g in np.flatnonzero(total == 0)]
        raise ValueError(f"no effective at-risk trials at {magnitude} sigma for {empty}")
    pooled = injuries.sum(axis=0) / total
    w = np.where(trials > 0, injuries / np.maximum(trials, 1), pooled[None, :])
    return {"w": w.tolist(), "pooled": pooled.tolist(), "trials": trials.tolist(), "injuries": injuries.tolist(),
            "fallback_cells": [[int(b), GROUPS[g]] for b, g in zip(*np.nonzero(trials == 0))],
            "groups": list(GROUPS), "magnitude": float(magnitude)}


def gripper_closed_by_bin(rows: list[dict], direction, n_bins: int, magnitude: float) -> dict:
    """Command-level phase evidence from the recovery rollouts: at each (rollout, query, seed), the gripper
    command's sign inferred from which ``magnitude``-sigma gripper perturbation changed the controller's
    input. Pushing a closed (+) command down flips it, pushing it up does not; ``direction(rollout, query,
    bank_index)`` is the perturbation's net sign on the gripper. Undetermined when both or neither flip.
    Returns the closed fraction per bin among determined continuations."""
    grip = ALL_GROUPS.index("gripper")
    seen: dict = {}
    for r in rows:
        if r["kind"] != 1 or r["group"] != grip or float(r["magnitude"]) != float(magnitude):
            continue
        key = (r["rollout"], r["query"], r["seed"])
        seen.setdefault(key, []).append((direction(r["rollout"], r["query"], r["bank_index"]), not r["noop"]))
    closed = np.zeros(int(n_bins))
    determined = np.zeros(int(n_bins), dtype=np.int64)
    undetermined = 0
    for (_, q, _), pairs in seen.items():
        effective = [d for d, eff in pairs if eff]
        if len(effective) != 1:
            undetermined += 1
            continue
        determined[int(q)] += 1
        closed[int(q)] += float(effective[0] < 0)
    frac = [float(c / n) if n else None for c, n in zip(closed, determined)]
    return {"closed_fraction": frac, "determined": determined.tolist(), "undetermined": undetermined}


def demo_gripper_closed_by_bin(gripper_actions: list[np.ndarray], n_bins: int) -> dict:
    """The same per-bin closed fraction in the demonstrations (raw gripper action > 0), frames binned by
    t / L."""
    closed = np.zeros(int(n_bins))
    count = np.zeros(int(n_bins), dtype=np.int64)
    for a in gripper_actions:
        a = np.asarray(a, dtype=np.float64)
        bins = np.minimum((np.arange(len(a)) * n_bins) // len(a), n_bins - 1)
        np.add.at(count, bins, 1)
        np.add.at(closed, bins, (a > 0).astype(np.float64))
    return {"closed_fraction": [float(c / n) if n else None for c, n in zip(closed, count)],
            "frames": count.tolist()}
