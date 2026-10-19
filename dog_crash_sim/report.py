"""Engineering report: figures (PNG) and a self-contained HTML page."""

from __future__ import annotations

import base64
import dataclasses
import html
import json
from pathlib import Path

import numpy as np

from .config import G, ScenarioConfig, dog_mass_and_height, road_friction
from .analysis.signals import cfc_filter
from .scene.builder import DOG_PARTS, VEH_PARTS

# reference palette (validated; slots in fixed order) and chart chrome
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
STATUS = {"good": "#0ca30c", "acceptable": "#fab219", "marginal": "#ec835a", "poor": "#d03b3b"}
STATUS_ICON = {"good": "✔", "acceptable": "●", "marginal": "▲", "poor": "✖"}
SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

CATEGORIES = [
    ("collision", "Collision", "Impact speed, direction, angle, overlap and collision partner"),
    ("vehicle", "Vehicle", "Mass, dimensions, structural stiffness, crush zones and rear-compartment geometry"),
    ("rear_passenger", "Rear passenger (dog)", "Body size, seating position, posture and whether the seat belt is used"),
    ("restraints", "Restraints", "Belt geometry, pretensioner, load limiter, airbags, seat design and head restraints"),
    ("trunk_cargo", "Trunk / cargo", "Cargo mass, dimensions, position, packing arrangement and restraint method"),
    ("environment", "Environment", "Road friction, pre-impact braking and vehicle movement before impact"),
    ("harness", "Harness under test", "Architecture, straps, attachment and tether"),
]


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASE, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 10,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelsize": 9,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round", "legend.frameon": False,
        "legend.fontsize": 8, "legend.labelcolor": INK2, "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    })
    return plt


def _small_multiples(plt, n, title, height=2.2):
    fig, axes = plt.subplots(n, 1, figsize=(8.0, height * n + 0.4), sharex=True, squeeze=False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    return fig, axes[:, 0]


def _finish(fig, axes, path, xlabel="time after first contact (ms)"):
    axes[-1].set_xlabel(xlabel)
    for ax in axes:
        ax.axvline(0.0, color=BASE, linewidth=1.0, zorder=0)
        labelled = [ln for ln in ax.get_lines() if not ln.get_label().startswith("_")]
        if len(labelled) > 1:
            ax.legend(handles=labelled, loc="best", ncol=min(4, len(labelled)))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=110)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


def make_figures(res, out_dir) -> list:
    plt = _style()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    C = res.channels
    t = res.t
    dt = float(np.median(np.diff(t)))
    tm = t * 1e3
    mo = res.motion
    vt = mo.t * 1e3
    figs = []
    win = (max(tm[0], -200.0), tm[-1])

    # 1. vehicle pulse
    fig, ax = _small_multiples(plt, 3, "Vehicle crash pulse (passenger compartment)")
    ab = cfc_filter(mo.acc_body, float(np.median(np.diff(mo.t))), 60.0) / G
    ax[0].plot(vt, ab[:, 0], color=SERIES[0], label="longitudinal")
    ax[0].plot(vt, ab[:, 1], color=SERIES[1], label="lateral")
    ax[0].set_ylabel("acceleration (g)")
    vb = np.einsum("nji,nj->ni", mo.R, mo.vel) * 3.6
    i0 = int(np.argmin(np.abs(mo.t)))
    ax[1].plot(vt, vb[:, 0] - vb[i0, 0], color=SERIES[0], label="longitudinal")
    ax[1].plot(vt, vb[:, 1] - vb[i0, 1], color=SERIES[1], label="lateral")
    ax[1].set_ylabel("velocity change (km/h)")
    ax[2].plot(vt, mo.crush_subject, color=SERIES[0], label="subject crush")
    ax[2].plot(vt, mo.crush_partner, color=SERIES[1], label="partner crush")
    intr = np.maximum.reduce([mo.intrusion_left, mo.intrusion_right, mo.intrusion_rear])
    if intr.max() > 0:
        ax[2].plot(vt, intr, color=SERIES[2], label="intrusion at the dog")
    ax[2].set_ylabel("deformation (m)")
    for a in ax:
        a.set_xlim(win)
    figs.append(("Vehicle crash pulse", _finish(fig, ax, out / "fig_vehicle_pulse.png")))

    # 2. restraint loads
    names = C.get("tendon_names", [])
    fig, ax = _small_multiples(plt, 3, "Restraint and harness loads")
    k = 0
    for j, nm in enumerate(names):
        if nm in ("neck_loop", "front_strap") or nm.startswith("cargo_"):
            continue
        ax[0].plot(tm, C["tendon_tension"][:, j] / 1e3, color=SERIES[k], label=nm.replace("_", " "))
        k += 1
    if "ring_force" in C and C["ring_force"].max() > 0:
        ax[0].plot(tm, C["ring_force"] / 1e3, color=SERIES[k], label="ring resultant")
    ax[0].set_ylabel("tension (kN)")
    k = 0
    for j, nm in enumerate(names):
        if nm in ("neck_loop", "front_strap"):
            ax[1].plot(tm, C["tendon_tension"][:, j] / 1e3, color=SERIES[k], label=nm.replace("_", " "))
            k += 1
    hf = C["harness_force"]
    if hf.size:
        ax[1].plot(tm, np.linalg.norm(hf[:, :3], axis=1) / 1e3, color=SERIES[k], label="harness frame into thorax")
    ax[1].set_ylabel("body loads (kN)")
    ax[2].plot(tm, C["harness_q"][:, 0] * 1e3, color=SERIES[0], label="shift along body")
    ax[2].plot(tm, np.degrees(C["harness_q"][:, 3]), color=SERIES[1], label="rotation (deg)")
    ax[2].set_ylabel("harness migration (mm / deg)")
    figs.append(("Restraint and harness loads", _finish(fig, ax, out / "fig_restraint_loads.png")))

    # 3. accelerations
    fig, ax = _small_multiples(plt, 1, "Dog accelerations (resultant, filtered)", height=2.8)
    for k, (nm, lab, cfc) in enumerate((("head_acc", "head (CFC 1000)", 1000.0), ("chest_acc", "thorax (CFC 180)", 180.0),
                                        ("pelvis_acc", "pelvis (CFC 180)", 180.0))):
        ax[0].plot(tm, np.linalg.norm(cfc_filter(C[nm], dt, cfc), axis=1) / G, color=SERIES[k], label=lab)
    ax[0].set_ylabel("acceleration (g)")
    figs.append(("Dog accelerations", _finish(fig, ax, out / "fig_accelerations.png")))

    # 4. neck & spine
    fig, ax = _small_multiples(plt, 3, "Neck and spine loads")
    ax[0].plot(tm, -cfc_filter(C["upper_neck_force"][:, 2], dt, 1000.0), color=SERIES[0], label="upper neck")
    ax[0].plot(tm, -cfc_filter(C["lower_neck_force"][:, 2], dt, 1000.0), color=SERIES[1], label="lower neck")
    ax[0].set_ylabel("axial force (N, + tension)")
    ax[1].plot(tm, cfc_filter(C["upper_neck_torque"][:, 1], dt, 600.0), color=SERIES[0], label="upper neck")
    ax[1].plot(tm, cfc_filter(C["lower_neck_torque"][:, 1], dt, 600.0), color=SERIES[1], label="lower neck")
    ax[1].set_ylabel("flexion moment (N m)")
    ax[2].plot(tm, cfc_filter(C["thoracolumbar_force"][:, 0], dt, 600.0), color=SERIES[0], label="thoracolumbar")
    ax[2].plot(tm, cfc_filter(C["lumbosacral_force"][:, 0], dt, 600.0), color=SERIES[1], label="lumbosacral")
    ax[2].set_ylabel("axial force (N, - compression)")
    figs.append(("Neck and spine loads", _finish(fig, ax, out / "fig_neck_spine.png")))

    # 5. kinematics
    fig, ax = _small_multiples(plt, 2, "Dog kinematics in the vehicle")
    hp = C["head_pos"] - C["head_pos"][0]
    for k, lab in enumerate(("forward (x)", "lateral (y)", "vertical (z)")):
        ax[0].plot(tm, hp[:, k], color=SERIES[k], label=lab)
    ax[0].set_ylabel("head excursion (m)")
    ax[1].plot(tm, C["torso_rot_angle"], color=SERIES[0])
    ax[1].set_ylabel("torso rotation (deg)")
    figs.append(("Dog kinematics", _finish(fig, ax, out / "fig_kinematics.png")))

    # 6. contacts (one series -> one colour)
    peaks = res.metrics.get("contact_peaks_N", {})
    if peaks:
        items = list(peaks.items())[:10][::-1]
        fig, a = plt.subplots(figsize=(8.0, 0.36 * len(items) + 1.0))
        fig.suptitle("Peak contact forces between the dog and the interior", x=0.01, ha="left", fontsize=12,
                     fontweight="bold", color=INK)
        y = np.arange(len(items))
        vals = [v / 1e3 for _, v in items]
        a.barh(y, vals, height=0.55, color=SERIES[0])
        a.set_yticks(y, [k for k, _ in items], color=INK2, fontsize=8)
        for yi, v in zip(y, vals):
            a.text(v, yi, f"  {v:.2f} kN", va="center", fontsize=8, color=INK2)
        a.set_xlabel("peak normal force (kN)")
        a.grid(axis="y", visible=False)
        a.set_xlim(0, max(vals) * 1.25)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        p = out / "fig_contacts.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        figs.append(("Interior contacts", p))

    # 7. cargo & seat
    if res.scene.cargo or C["seat_back_angle"].max() > 0.01 or C["door_left"].max() > 0 or C["door_right"].max() > 0:
        fig, ax = _small_multiples(plt, 3, "Trunk / cargo, seat back and intrusion")
        for j, c in enumerate(res.scene.cargo[:4]):
            d0 = C["cargo_pos"][0, j]
            ax[0].plot(tm, np.linalg.norm(C["cargo_pos"][:, j] - d0, axis=1), color=SERIES[j], label=c["name"])
        ax[0].set_ylabel("cargo displacement (m)")
        ax[1].plot(tm, C["cargo_seatback_force"] / 1e3, color=SERIES[0], label="cargo on seat back")
        ax[1].plot(tm, C["dog_cargo_force"] / 1e3, color=SERIES[1], label="cargo on dog")
        ax[1].set_ylabel("force (kN)")
        ax[2].plot(tm, np.degrees(C["seat_back_angle"]), color=SERIES[0], label="seat back (+ forward, deg)")
        ax[2].plot(tm, (C["door_left"] + C["door_right"] + C["rear_wall"]) * 100, color=SERIES[1],
                   label="door / tailgate intrusion (cm)")
        ax[2].set_ylabel("deg / cm")
        figs.append(("Cargo, seat back and intrusion", _finish(fig, ax, out / "fig_cargo_seat.png")))

    # 8. criteria
    crit = res.metrics.get("criteria", [])
    if crit:
        rows = crit[::-1]
        fig, a = plt.subplots(figsize=(8.0, 0.30 * len(rows) + 1.2))
        fig.suptitle("Criteria: value relative to the reference limit", x=0.01, ha="left", fontsize=12,
                     fontweight="bold", color=INK)
        y = np.arange(len(rows))
        r = [min(c["ratio"], 3.0) for c in rows]
        a.barh(y, r, height=0.55, color=[STATUS[c["status"]] for c in rows])
        a.axvline(1.0, color=INK2, linewidth=1.0)
        a.set_yticks(y, [c["label"][:58] for c in rows], fontsize=7, color=INK2)
        for yi, c in zip(y, rows):
            a.text(min(c["ratio"], 3.0), yi, f"  {c['ratio']:.2f}  {c['status']}", va="center", fontsize=7, color=INK2)
        a.set_xlim(0, 3.6)
        a.set_xlabel("ratio to limit (1.0 = limit; capped at 3)")
        a.grid(axis="y", visible=False)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        p = out / "fig_criteria.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)
        figs.append(("Criteria overview", p))
    return figs


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, list) and v and dataclasses.is_dataclass(v[0]):
        return "; ".join(", ".join(f"{f.name}={_fmt(getattr(x, f.name))}" for f in dataclasses.fields(x)) for x in v)
    return html.escape(str(v))


def _flatten(obj, prefix=""):
    rows = []
    for f in dataclasses.fields(obj):
        v = getattr(obj, f.name)
        if dataclasses.is_dataclass(v):
            rows += _flatten(v, prefix + f.name + ".")
        else:
            rows.append((prefix + f.name, v, f.metadata.get("unit", ""), f.metadata.get("help", "")))
    return rows


def _img(path) -> str:
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return f"data:image/png;base64,{data}"


def write_html(res, out_dir, figures, keyframes=None, videos=()) -> Path:
    cfg: ScenarioConfig = res.cfg
    mt = res.metrics
    out = Path(out_dir)
    overall = mt.get("summary", {}).get("overall", "n/a")
    ov_status = {"PASS": "good", "MARGINAL": "marginal", "FAIL": "poor"}.get(overall, "acceptable")
    mass, height = dog_mass_and_height(cfg.rear_passenger)
    info = res.info
    pk = mt.get("peaks", {})
    tiles = [
        ("Vehicle delta-V", f"{pk.get('vehicle_delta_v_kph', 0):.1f} km/h"),
        ("Vehicle peak", f"{pk.get('vehicle_peak_g', 0):.1f} g"),
        ("Tether / belt peak", f"{pk.get('tether_or_belt_N', 0) / 1e3:.2f} kN"),
        ("Head excursion", f"{pk.get('head_excursion_m', 0) * 100:.0f} cm"),
        ("Head peak", f"{pk.get('head_resultant_g', 0):.0f} g"),
        ("Torso rotation", f"{pk.get('torso_rotation_deg', 0):.0f}°"),
    ]
    crit_rows = []
    for c in mt.get("criteria", []):
        st = c["status"]
        lim = "" if c["limit"] is None else f"{c['limit']:.4g}"
        crit_rows.append(
            f"<tr><td>{html.escape(c['label'])}</td><td class='num'>{c['value']:.4g} {html.escape(c['unit'])}</td>"
            f"<td class='num'>{lim}</td><td class='num'>{c['ratio']:.2f}</td>"
            f"<td><span class='pill {st}'>{STATUS_ICON[st]} {st}</span></td><td class='num'>{html.escape(c['basis'])}</td>"
            f"<td class='muted'>{html.escape(c.get('note', ''))}</td></tr>")
    flags = "".join(f"<li>{html.escape(f)}</li>" for f in mt.get("flags", [])) or "<li>None</li>"
    ev = "".join(f"<tr><td>{html.escape(k.replace('_', ' '))}</td><td class='num'>"
                 f"{'' if v is None else f'{float(v) * 1e3:.1f} ms'}</td></tr>" for k, v in res.events.items())
    risk = "".join(f"<tr><td>{html.escape(k)}</td><td class='num'>{v * 100:.1f} %</td></tr>"
                   for k, v in mt.get("risk", {}).items())
    params = []
    for key, title, desc in CATEGORIES:
        sec = getattr(cfg, key)
        rows = "".join(f"<tr><td title='{html.escape(h)}'>{html.escape(n)}</td><td>{_fmt(v)}</td>"
                       f"<td class='muted'>{html.escape(u)}</td></tr>" for n, v, u, h in _flatten(sec))
        params.append(f"<details><summary><b>{title}</b> <span class='muted'>&mdash; {desc}</span></summary>"
                      f"<table class='params'>{rows}</table></details>")
    fig_html = "".join(f"<figure><img alt='{html.escape(t)}' src='{_img(p)}'><figcaption>{html.escape(t)}</figcaption></figure>"
                       for t, p in figures)
    kf = (f"<figure><img alt='key frames' src='{_img(keyframes)}'><figcaption>Key frames (side camera)</figcaption>"
          f"</figure>") if keyframes else ""
    vids = "".join(f"<li><a href='{html.escape(Path(v).name)}'>{html.escape(Path(v).name)}</a></li>" for v in videos)
    notes = "".join(f"<li>{html.escape(n)}</li>" for n in info.get("notes", []))
    scen = (f"{cfg.collision.impact_direction} impact at {cfg.collision.impact_speed_kph:.0f} km/h into "
            f"{cfg.collision.partner.type.replace('_', ' ')}" if cfg.collision.mode == "crash" else
            f"sled test, {cfg.collision.sled.shape} pulse, {cfg.collision.sled.delta_v_kph:.0f} km/h")
    dog = (f"{mass:.1f} kg dog ({cfg.rear_passenger.breed_size}, withers {height:.2f} m), "
           f"{cfg.rear_passenger.posture.replace('_', ' ')} on {cfg.rear_passenger.seating_position.replace('_', ' ')}, "
           f"{'restrained via ' + str(info.get('attachment')).replace('_', ' ') if cfg.rear_passenger.seat_belt_used else 'not restrained'}")
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Harness Crash Report</title>
<style>
:root {{ color-scheme: light; --surface:#fcfcfb; --page:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --line:#e1e0d9; --good:#0ca30c; --acceptable:#fab219; --marginal:#ec835a; --poor:#d03b3b; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ color-scheme: dark; --surface:#1a1a19;
  --page:#0d0d0d; --ink:#f0efec; --ink2:#c3c2b7; --line:#2c2c2a; }} }}
:root[data-theme="dark"] {{ color-scheme: dark; --surface:#1a1a19; --page:#0d0d0d; --ink:#f0efec; --ink2:#c3c2b7;
  --line:#2c2c2a; }}
body {{ margin:0; background:var(--page); color:var(--ink); font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:1080px; margin:0 auto; padding:24px 16px 64px; }}
h1 {{ font-size:22px; margin:0 0 4px; }} h2 {{ font-size:17px; margin:32px 0 8px; }}
.muted {{ color:var(--muted); }} .sub {{ color:var(--ink2); margin:0 0 16px; }}
.card {{ background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:16px; }}
.hero {{ display:flex; gap:16px; align-items:center; flex-wrap:wrap; }}
.hero .verdict {{ font-size:44px; font-weight:650; }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; margin-top:12px; }}
.tile {{ background:var(--surface); border:1px solid var(--line); border-radius:8px; padding:10px 12px; }}
.tile .l {{ color:var(--ink2); font-size:12px; }} .tile .v {{ font-size:20px; font-weight:600; }}
table {{ border-collapse:collapse; width:100%; font-size:13px; }}
td, th {{ border-bottom:1px solid var(--line); padding:5px 8px; text-align:left; vertical-align:top; }}
th {{ color:var(--ink2); font-weight:600; }} td.num {{ font-variant-numeric:tabular-nums; white-space:nowrap; }}
.scroll {{ overflow-x:auto; }}
.pill {{ display:inline-block; padding:1px 8px; border-radius:10px; font-size:12px; font-weight:600; color:#0b0b0b;
  white-space:nowrap; }}
.pill.good {{ background:var(--good); color:#fff; }} .pill.acceptable {{ background:var(--acceptable); }}
.pill.marginal {{ background:var(--marginal); }} .pill.poor {{ background:var(--poor); color:#fff; }}
figure {{ margin:16px 0; background:#fcfcfb; border:1px solid var(--line); border-radius:10px; padding:8px; }}
figure img {{ width:100%; height:auto; display:block; }} figcaption {{ color:#52514e; font-size:12px; padding:4px; }}
details {{ background:var(--surface); border:1px solid var(--line); border-radius:8px; padding:8px 12px; margin:6px 0; }}
table.params td:first-child {{ font-family:ui-monospace,monospace; font-size:12px; white-space:nowrap; }}
.warn {{ border-left:4px solid var(--marginal); padding-left:12px; }}
</style></head><body><main>
<h1>Rear-seat harness crash simulation &mdash; {html.escape(cfg.name)}</h1>
<p class="sub">{html.escape(scen)}. {html.escape(dog)}. Harness: {html.escape(cfg.harness.name)} ({html.escape(cfg.harness.style)}).</p>
<div class="card hero"><div><div class="muted">Overall result</div>
<div class="verdict"><span class="pill {ov_status}" style="font-size:28px;padding:4px 16px">{STATUS_ICON[ov_status]} {overall}</span></div></div>
<div><div class="muted">Worst criterion</div><div><b>{html.escape(mt.get('summary', {}).get('worst_criterion', ''))}</b>
 &mdash; {mt.get('summary', {}).get('worst_ratio', 0):.2f} &times; limit</div>
<div class="muted">Dog remained on its seat: <b>{'yes' if mt.get('remained_on_seat') else 'no'}</b></div>
<div class="muted">Injury index (mean worst ratio per body region, lower is better): <b>{mt.get('summary', {}).get('injury_index', 0):.2f}</b></div></div></div>
<div class="tiles">{''.join(f"<div class='tile'><div class='l'>{l}</div><div class='v'>{v}</div></div>" for l, v in tiles)}</div>
<h2>Findings</h2><div class="card warn"><ul>{flags}</ul></div>
<h2>Criteria</h2><div class="card scroll"><table><tr><th>Criterion</th><th>Value</th><th>Limit</th><th>Ratio</th><th>Status</th><th>Basis</th><th>Note</th></tr>
{''.join(crit_rows)}</table>
<p class="muted">Status bands: ratio &lt; 0.6 good, &lt; 0.85 acceptable, &lt; 1.0 marginal, &ge; 1.0 poor.
"scaled-human" values transfer human injury reference values to the dog by geometric similitude
(&lambda; = {mt.get('scaling', {}).get('lambda_body', 1):.3f}); there are no validated canine injury criteria, so treat them as indicative and use
the simulator primarily to compare harness designs and configurations.</p></div>
<h2>Indicative injury risk (scaled human AIS3+ curves)</h2><div class="card"><table>{risk}</table></div>
{kf}
{fig_html}
<h2>Events</h2><div class="card"><table>{ev}</table></div>
{'<h2>Videos</h2><div class="card"><ul>' + vids + '</ul></div>' if vids else ''}
<h2>Scenario parameters</h2>{''.join(params)}
<details><summary><b>Simulation</b> and <b>assessment</b> settings</summary><table class="params">
{''.join(f"<tr><td>{html.escape(n)}</td><td>{_fmt(v)}</td><td class='muted'>{html.escape(u)}</td></tr>" for n, v, u, h in _flatten(cfg.simulation) + _flatten(cfg.assessment))}
</table></details>
<h2>Model notes</h2><div class="card"><ul>
<li>Road friction {road_friction(cfg.environment):.2f}; speed at the start of the simulation {info.get('initial_speed_kph', 0):.1f} km/h.</li>
<li>Structural engagement of the struck face {info.get('engagement', 0):.2f}; peak interface force {info.get('peak_interface_force_kN', 0):.0f} kN; OLC {info.get('olc_g', 0):.1f} g.</li>
<li>Dog model scaled to {mass:.1f} kg (length scale {info.get('length_scale', 1):.3f}); wall time {info.get('wall_time_s', 0):.1f} s.</li>
{notes}</ul></div>
</main></body></html>"""
    path = out / "report.html"
    path.write_text(doc)
    return path


def build_report(res, out_dir, keyframes=True, videos=()) -> Path:
    out = Path(out_dir)
    figs = make_figures(res, out)
    kf = None
    if keyframes:
        try:
            from .render import render_keyframes
            t0, t1 = max(float(res.states_t[0]), -0.1), float(res.states_t[-1])
            times = [round(x * 1e3) for x in np.linspace(t0, t1 - 1e-3, 8)]
            kf = render_keyframes(res, out / "keyframes.png", camera="side", times_ms=times)
        except Exception as exc:  # rendering needs OpenGL; the report works without it
            res.info.setdefault("notes", []).append(f"Key frames not rendered ({type(exc).__name__}: {exc}).")
    return write_html(res, out, figs, kf, videos)
