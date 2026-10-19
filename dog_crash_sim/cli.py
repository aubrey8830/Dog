"""Command-line application.

Examples::

    python -m dog_crash_sim presets
    python -m dog_crash_sim template -o my_test.yaml
    python -m dog_crash_sim run cps_sled_frontal
    python -m dog_crash_sim run my_test.yaml --set collision.impact_speed_kph=40 --render
    python -m dog_crash_sim sweep frontal_full_width_56 --param harness.style \
        --values padded_vest,y_front,norwegian,step_in,collar_only
    python -m dog_crash_sim compare results/run_a results/run_b
    python -m dog_crash_sim view results/run_a
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

from .config import ScenarioConfig, commented_template, config_from_dict, deep_merge, params_table
from .presets import PRESETS, preset


def _set_path(d: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def scenario_dict(source: str, sets=()) -> dict:
    """Load a scenario from a preset name or a YAML/JSON file and apply ``key=value`` overrides."""
    if source in PRESETS:
        data = preset(source)
    else:
        path = Path(source)
        if not path.exists():
            raise SystemExit(f"'{source}' is neither a preset ({', '.join(PRESETS)}) nor a file")
        text = path.read_text()
        data = json.loads(text) if path.suffix == ".json" else (yaml.safe_load(text) or {})
        base = data.pop("preset", None)
        if base:
            data = deep_merge(preset(base), data)
        data.setdefault("name", path.stem)
    for s in sets:
        if "=" not in s:
            raise SystemExit(f"--set expects key=value, got {s!r}")
        k, v = s.split("=", 1)
        _set_path(data, k.strip(), yaml.safe_load(v))
    return data


def _progress(quiet):
    last = {"msg": None, "t": time.time()}

    def cb(msg, frac):
        if quiet:
            return
        if msg != last["msg"] or time.time() - last["t"] > 2.0 or frac >= 1.0:
            sys.stderr.write(f"\r  [{frac * 100:5.1f} %] {msg:<40}")
            sys.stderr.flush()
            last.update(msg=msg, t=time.time())
            if frac >= 1.0:
                sys.stderr.write("\n")
    return cb


def run_one(data: dict, out_root: Path, render: bool = False, report: bool = True, quiet: bool = False,
            out_dir: Path = None) -> Path:
    from .results_io import save_results
    from .simulation import run_scenario
    cfg = config_from_dict(data)
    if render:
        cfg.simulation.render.enabled = True
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(out_dir) if out_dir else out_root / f"{cfg.name}_{stamp}"
    if not quiet:
        print(f"Scenario '{cfg.name}': {cfg.description or ''}".rstrip())
    res = run_scenario(cfg, progress=_progress(quiet))
    save_results(res, out)
    videos = []
    if cfg.simulation.render.enabled:
        from .render import render_video
        try:
            videos = render_video(res, out)
        except Exception as exc:  # needs OpenGL (a display, MUJOCO_GL=egl/osmesa, or xvfb-run)
            print(f"  video rendering failed: {exc}")
    if report:
        from .report import build_report
        build_report(res, out, keyframes=True, videos=videos)
    if not quiet:
        print_summary(res)
        print(f"  results: {out}")
    return out


def print_summary(res) -> None:
    mt = res.metrics
    s = mt.get("summary", {})
    print(f"  overall: {s.get('overall')}   injury index: {s.get('injury_index', 0):.2f} (lower is better)   "
          f"worst: {s.get('worst_criterion')} ({s.get('worst_ratio', 0):.2f} x limit)")
    for f in mt.get("flags", []):
        print(f"  ! {f}")
    print(f"  {'criterion':<58} {'value':>12} {'limit':>10}  status")
    for c in mt.get("criteria", []):
        lim = "" if c["limit"] is None else f"{c['limit']:.4g}"
        print(f"  {c['label'][:58]:<58} {c['value']:>9.4g} {c['unit']:<3}{lim:>9}  {c['status']}")


def cmd_run(a):
    data = scenario_dict(a.scenario, a.set)
    run_one(data, Path(a.out), render=a.render, report=not a.no_report, quiet=a.quiet,
            out_dir=Path(a.dir) if a.dir else None)


def _sweep_worker(args):
    data, out_dir = args
    try:
        run_one(data, out_dir.parent, render=False, report=True, quiet=True, out_dir=out_dir)
        return str(out_dir), None
    except Exception as exc:  # keep the sweep going
        return str(out_dir), repr(exc)


def cmd_sweep(a):
    base = scenario_dict(a.scenario, a.set)
    values = [yaml.safe_load(v) for v in a.values.split(",")]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(a.out) / f"sweep_{base.get('name', 'scenario')}_{a.param.replace('.', '-')}_{stamp}"
    root.mkdir(parents=True, exist_ok=True)
    jobs = []
    for v in values:
        d = copy.deepcopy(base)
        _set_path(d, a.param, v)
        d["name"] = f"{base.get('name', 'scenario')}__{a.param.split('.')[-1]}={v}"
        jobs.append((d, root / f"{a.param.split('.')[-1]}={v}"))
    print(f"Sweep over {a.param}: {values} ({len(jobs)} runs, {a.jobs} parallel)")
    if a.jobs > 1:
        from multiprocessing import get_context
        with get_context("spawn").Pool(a.jobs) as pool:
            results = pool.map(_sweep_worker, jobs)
    else:
        results = [_sweep_worker(j) for j in jobs]
    for path, err in results:
        print(f"  {'FAILED ' + err if err else 'ok'}  {path}")
    dirs = [p for p, e in results if not e]
    table = compare_dirs(dirs, root / "comparison")
    print(table)


def compare_dirs(dirs, out_base: Path) -> str:
    rows = []
    keys = []
    for d in dirs:
        s = json.loads((Path(d) / "summary.json").read_text())
        crit = {c["key"]: c for c in s["metrics"]["criteria"]}
        for k in crit:
            if k not in keys:
                keys.append(k)
        summ = s["metrics"]["summary"]
        rows.append((Path(d).name, f"{summ['overall']} / index {summ.get('injury_index', 0):.2f}", crit,
                     s["metrics"].get("flags", [])))
    out_base.parent.mkdir(parents=True, exist_ok=True)
    with open(out_base.with_suffix(".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "overall"] + keys)
        for name, ov, crit, _ in rows:
            w.writerow([name, ov] + [f"{crit[k]['value']:.4g}" if k in crit else "" for k in keys])
    labels = {}
    for _, _, crit, _ in rows:
        for k, c in crit.items():
            labels[k] = f"{c['label']} ({c['unit']})"
    head = "| criterion | " + " | ".join(r[0] for r in rows) + " |\n|---|" + "---|" * len(rows) + "\n"
    body = "| **overall / injury index (lower is better)** | " + " | ".join(f"**{r[1]}**" for r in rows) + " |\n"
    for k in keys:
        cells = []
        for _, _, crit, _ in rows:
            c = crit.get(k)
            cells.append("" if c is None else f"{c['value']:.3g} ({c['status']})")
        body += f"| {labels[k]} | " + " | ".join(cells) + " |\n"
    md = head + body
    out_base.with_suffix(".md").write_text(md)
    return md


def cmd_compare(a):
    out = Path(a.out) / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(compare_dirs(a.dirs, out))
    print(f"written {out}.md / .csv")


def cmd_presets(a):
    for k, v in PRESETS.items():
        print(f"{k:<26} {v['description']}")


def cmd_template(a):
    data = preset(a.preset) if a.preset else {}
    cfg = config_from_dict(data)
    text = commented_template(cfg)
    if a.output:
        Path(a.output).write_text(text)
        print(f"written {a.output}")
    else:
        print(text)


def cmd_params(a):
    print(params_table())


def cmd_view(a):
    from .results_io import load_results
    from .render import interactive_replay
    res = load_results(a.results)
    interactive_replay(res, speed=a.speed)


def cmd_render(a):
    from .render import render_video
    from .results_io import load_results
    res = load_results(a.results)
    cams = a.cameras.split(",") if a.cameras else None
    for p in render_video(res, Path(a.results), cameras=cams, fps=a.fps, slow=a.slow):
        print(f"written {p}")


def cmd_live(a):  # pragma: no cover - interactive
    """Run a scenario and show it in the MuJoCo viewer as it is computed."""
    from .render import interactive_replay
    from .simulation import run_scenario
    res = run_scenario(config_from_dict(scenario_dict(a.scenario, a.set)), progress=_progress(False))
    print_summary(res)
    interactive_replay(res, speed=a.speed)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dog-crash-sim", description="Rear-seat dog harness crash simulator")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a scenario (preset name or YAML/JSON file)")
    r.add_argument("scenario")
    r.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a parameter (repeatable)")
    r.add_argument("--out", default="results", help="results root directory")
    r.add_argument("--dir", default=None, help="exact output directory")
    r.add_argument("--render", action="store_true", help="render videos")
    r.add_argument("--no-report", action="store_true", help="skip the HTML report")
    r.add_argument("--quiet", action="store_true")
    r.set_defaults(func=cmd_run)
    s = sub.add_parser("sweep", help="run a scenario for several values of one parameter and compare")
    s.add_argument("scenario")
    s.add_argument("--param", required=True, help="dotted parameter name, e.g. collision.impact_speed_kph")
    s.add_argument("--values", required=True, help="comma-separated values")
    s.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    s.add_argument("--out", default="results")
    s.add_argument("--jobs", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)))
    s.set_defaults(func=cmd_sweep)
    c = sub.add_parser("compare", help="compare saved runs")
    c.add_argument("dirs", nargs="+")
    c.add_argument("--out", default="results")
    c.set_defaults(func=cmd_compare)
    sub.add_parser("presets", help="list scenario presets").set_defaults(func=cmd_presets)
    t = sub.add_parser("template", help="write a fully commented scenario file")
    t.add_argument("--preset", default=None)
    t.add_argument("-o", "--output", default=None)
    t.set_defaults(func=cmd_template)
    sub.add_parser("params", help="list every parameter with unit and description").set_defaults(func=cmd_params)
    v = sub.add_parser("view", help="replay saved results in the interactive MuJoCo viewer (needs a display)")
    v.add_argument("results")
    v.add_argument("--speed", type=float, default=0.05, help="playback speed (0.05 = 20x slow motion)")
    v.set_defaults(func=cmd_view)
    rv = sub.add_parser("render", help="render videos from saved results")
    rv.add_argument("results")
    rv.add_argument("--cameras", default=None, help="side,dog,top,front,world")
    rv.add_argument("--fps", type=int, default=None)
    rv.add_argument("--slow", type=float, default=None)
    rv.set_defaults(func=cmd_render)
    lv = sub.add_parser("live", help="simulate and open the interactive viewer")
    lv.add_argument("scenario")
    lv.add_argument("--set", action="append", default=[])
    lv.add_argument("--speed", type=float, default=0.05)
    lv.set_defaults(func=cmd_live)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
