# Hello, Spikes

Companion code for the XRDS *Hello World* column
**"Hello, Spikes: A First Spiking Neural Network for Edge Intelligence"** (Fall 2026).

Everything the column walks through lives in a single notebook,
[`hello_spikes.ipynb`](hello_spikes.ipynb):

1. Turn a short video clip into an event stream with `eventify-dvs`.
2. Wire the events into a small Brian2 SNN (LIF integrator + alarm neuron).
3. Run it and watch the alarm neuron fire when something in the frame moves.

## Run

```bash
uv sync
uv run jupyter lab hello_spikes.ipynb
```

Drop any short `.mp4` next to the notebook as `wave.mp4` and run
top-to-bottom. Total wall time on a laptop: ~15 seconds after imports.

## Dependencies

- [Brian2](https://brian2.readthedocs.io/) — spiking-network simulator
- [eventify-dvs](https://pypi.org/project/eventify-dvs/) — video → simulated event-camera stream
- matplotlib, numpy, jupyter

## Column

Draft of the column itself lives at
[`Arpan-206/XRDS-SNN-WriteUp`](https://github.com/Arpan-206/XRDS-SNN-WriteUp).
