"""Regenerate the column's figures and headline numbers from the sample clip.

Runs the exact pipeline from the column on wave.mp4 with a fixed RNG seed,
so every run (and every reader) gets the same figures and the same numbers.

    uv run python make_figures.py
"""

import json

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from brian2 import (
    NeuronGroup,
    SpikeGeneratorGroup,
    SpikeMonitor,
    Synapses,
    mV,
    ms,
    run,
    second,
    seed,
    us,
)
from eventify import video_to_event_stream

SENSOR, C_THRESH, BIN_S = (128, 128), 0.15, 0.1
CLIP = "wave.mp4"

# --- Step 1: clip -> events -------------------------------------------------
events = np.concatenate(
    list(video_to_event_stream(CLIP, c_thresh=C_THRESH, sensor_size=SENSOR))
)
t_end = events["t"].max() / 1e6

# --- Step 2: events -> SNN (identical to the column's Listing 2) ------------
seed(11)  # fixed connectivity: same figure for every reader

idx = events["y"].astype(int) * 128 + events["x"].astype(int)
pixels = SpikeGeneratorGroup(128 * 128, idx, events["t"] * us)

eqs = "dv/dt = (v_rest - v) / tau_m : volt"
integrator = NeuronGroup(
    N=8, model=eqs,
    threshold="v > v_th", reset="v = v_rest", method="exact",
    namespace={"v_rest": -70 * mV, "v_th": -58 * mV, "tau_m": 10 * ms},
)
integrator.v = -70 * mV
S = Synapses(pixels, integrator, on_pre="v_post += 0.5*mV")
S.connect(p=0.02)

# --- Step 3: run -------------------------------------------------------------
M = SpikeMonitor(integrator)
run(events["t"].max() * us)

# --- Stats -------------------------------------------------------------------
bins = np.arange(0, t_end + BIN_S, BIN_S)
event_hist, _ = np.histogram(events["t"] / 1e6, bins=bins)
int_hist, _ = np.histogram(np.asarray(M.t / second), bins=bins)
centers = 0.5 * (bins[:-1] + bins[1:])
r = np.corrcoef(event_hist, int_hist)[0, 1]

stats = {
    "clip": CLIP,
    "clip_seconds": round(t_end, 2),
    "events": int(len(events)),
    "integrator_spikes": int(M.num_spikes),
    "compression": round(len(events) / max(M.num_spikes, 1)),
    "pearson_r": round(float(r), 2),
}
print(json.dumps(stats, indent=2))

# --- Figure 2: sensor rate vs integrator rate --------------------------------
fig, axes = plt.subplots(2, 1, figsize=(9, 4), sharex=True)
axes[0].bar(centers, event_hist, width=BIN_S, color="0.3", align="center")
axes[0].set_ylabel("events per 100 ms")
axes[0].set_title("What the camera saw")
axes[1].bar(centers, int_hist, width=BIN_S, color="C0", align="center")
axes[1].set_ylabel("spikes per 100 ms")
axes[1].set_xlabel("time (s)")
axes[1].set_title("What the network said")
fig.suptitle(f"Input events vs. output spikes  —  r = {r:.2f}", y=1.02)
plt.tight_layout()
for ext in ("png", "pdf"):
    plt.savefig(f"sensor_vs_integrator.{ext}", dpi=160, bbox_inches="tight")
plt.close(fig)

# --- Figure 1: pipeline architecture, drawn from the same run ----------------
cap = cv2.VideoCapture(CLIP)
cap.set(cv2.CAP_PROP_POS_FRAMES, 60)
frame = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY)
cap.release()

# Events from one 33 ms window around the same frame.
t_lo, t_hi = 60 / 30 * 1e6, 61 / 30 * 1e6
win = events[(events["t"] >= t_lo) & (events["t"] < t_hi)]

fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2))
fig.subplots_adjust(left=0.045, right=0.99, top=0.80, bottom=0.14,
                    wspace=0.45)
axes[0].imshow(frame, cmap="gray")
axes[0].set_title("camera frame\n(dense, 30 fps)")
on, off = win[win["p"] == 1], win[win["p"] == 0]
axes[1].scatter(on["x"], on["y"], s=1, c="C0", label="ON")
axes[1].scatter(off["x"], off["y"], s=1, c="0.6", label="OFF")
axes[1].set_xlim(0, 128)
axes[1].set_ylim(128, 0)
axes[1].set_aspect("equal")
axes[1].legend(loc="lower right", fontsize=7, markerscale=4)
axes[1].set_title("events, one 33 ms window\n(sparse: only what changed)")
axes[2].scatter(np.asarray(M.t / second), np.asarray(M.i), s=30, c="C0",
                marker="|", linewidths=1.2)
axes[2].set_ylim(-0.5, 7.5)
axes[2].set_yticks(range(8))
axes[2].set_ylabel("neuron")
axes[2].set_xlabel("time (s)")
axes[2].set_title("output spikes\n(8 neurons)")
for ax in axes[:2]:
    ax.set_xticks([])
    ax.set_yticks([])
p0, p1, p2 = (ax.get_position() for ax in axes)
for a, b, label in ((p0, p1, "eventify-dvs\nchanges \u2192 events"),
                    (p1, p2, "16,384 pixels\n\u2192 8 neurons")):
    xm = (a.x1 + b.x0) / 2
    fig.text(xm, 0.46, "→", fontsize=17, ha="center", va="center")
    fig.text(xm, 0.54, label, fontsize=7, ha="center", va="bottom")
for ext in ("png", "pdf"):
    plt.savefig(f"architecture.{ext}", dpi=160, bbox_inches="tight")
plt.close(fig)


# --- Figure: how one spiking neuron decides (idealized explainer) ------------
# Hand-picked input times demonstrate both regimes: spread-out spikes leak
# away; a tight burst crosses the threshold and fires the neuron.
in_spikes = [0.07, 0.19, 0.32, 0.57, 0.585, 0.60, 0.615, 0.88]
REST, TH, JUMP, TAU, DT = 0.0, 1.0, 0.40, 0.05, 0.001

tt = np.arange(0, 1.0, DT)
v = np.zeros_like(tt)
fires = []
vi = REST
spk = iter(sorted(in_spikes))
nxt = next(spk, None)
for i, ti in enumerate(tt):
    vi += -(vi - REST) / TAU * DT
    while nxt is not None and nxt <= ti:
        vi += JUMP
        nxt = next(spk, None)
    if vi >= TH:
        fires.append(ti)
        v[i] = TH
        vi = REST
    else:
        v[i] = vi

fig, ax = plt.subplots(figsize=(9, 3.4))
ax.set_xlim(0, 1.0)
ax.set_ylim(-0.28, 2.05)
ax.axis("off")

# Input spike row.
ax.text(-0.015, 1.45, "input\nspikes", ha="right", va="center", fontsize=9)
for s in in_spikes:
    ax.plot([s, s], [1.32, 1.58], color="0.3", lw=2)

# Membrane potential, threshold, rest.
ax.plot(tt, v, color="0.15", lw=1.6)
ax.axhline(TH, ls="--", color="C3", lw=1, xmax=0.97)
ax.axhline(REST, ls=":", color="0.55", lw=1, xmax=0.97)
ax.text(1.0, TH, "threshold", ha="left", va="center", fontsize=8, color="C3")
ax.text(1.0, REST, "rest", ha="left", va="center", fontsize=8, color="0.55")
ax.text(-0.015, 0.5, "membrane\npotential $v$", ha="right", va="center",
        fontsize=9)

# Output spike row.
ax.text(-0.015, 1.86, "output", ha="right", va="center", fontsize=9,
        color="C0")
for f in fires:
    ax.plot([f, f], [1.74, 2.0], color="C0", lw=2.5)

# Annotations for the two regimes.
ax.annotate("too far apart:\nthe leak wins", xy=(0.26, 0.30),
            xytext=(0.16, 0.78), fontsize=8.5, ha="center", color="0.3",
            arrowprops=dict(arrowstyle="->", color="0.5", lw=1))
ax.annotate("close together:\nthey pile up", xy=(0.605, 0.82),
            xytext=(0.48, 1.05), fontsize=8.5, ha="center", color="0.3",
            arrowprops=dict(arrowstyle="->", color="0.5", lw=1))
if fires:
    ax.annotate("fires once,\nthen resets", xy=(fires[0] + 0.004, 0.35),
                xytext=(0.8, 0.75), fontsize=8.5, ha="center", color="C0",
                arrowprops=dict(arrowstyle="->", color="C0", lw=1))
ax.annotate("", xy=(0.35, -0.2), xytext=(0.05, -0.2),
            arrowprops=dict(arrowstyle="->", color="0.4", lw=1))
ax.text(0.2, -0.13, "time", ha="center", fontsize=8, color="0.4")

plt.tight_layout()
for ext in ("png", "pdf"):
    plt.savefig(f"neuron_schematic.{ext}", dpi=160, bbox_inches="tight")
plt.close(fig)

print("wrote sensor_vs_integrator, architecture, and neuron_schematic figures")
