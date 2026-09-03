# No Frames Attached

Companion code for the XRDS *Hello World* column
**"No Frames Attached: A Spiking Neural Network for Your Webcam"** (Fall 2026).

Everything the column walks through lives in a single notebook,
[`hello_spikes.ipynb`](hello_spikes.ipynb):

1. Turn a five-second webcam capture into an event stream with `eventify-dvs`.
2. Wire the events into a small Brian2 SNN (an 8-neuron LIF integrator).
3. Run it and watch the integrator's spike rate track the motion in front
   of the camera.

## Run

```bash
uv sync
uv run jupyter lab hello_spikes.ipynb
```

Run top-to-bottom and wave at your webcam when Step 1 asks you to.
Total wall time on a laptop: ~15 seconds after imports.

## Reproduce the column's figures

The exact figures and numbers printed in the column come from the bundled
sample clip (`wave.mp4`, hands waving at the camera) processed with a
fixed RNG seed:

```bash
uv run python make_figures.py
```

This writes `architecture.{png,pdf}` and `sensor_vs_integrator.{png,pdf}`
and prints the headline stats (253,085 events → 201 integrator spikes,
~1,300× compression, r = 0.81).

`wave.mp4` is derived (cropped, resized, re-encoded) from
[HandWaveExample.webm](https://commons.wikimedia.org/wiki/File:HandWaveExample.webm)
by NMu11er, Wikimedia Commons, licensed
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
The derived clip is likewise CC BY-SA 4.0.

## Dependencies

- [Brian2](https://brian2.readthedocs.io/) — spiking-network simulator
- [eventify-dvs](https://pypi.org/project/eventify-dvs/) — video → simulated event-camera stream
- matplotlib, numpy, opencv, jupyter

`eventify-dvs` is a deliberately small stand-in for heavier DVS
simulators. Anything that yields `(x, y, t, p)` events can replace it:
[v2e](https://github.com/SensorsINI/v2e),
[ESIM](https://github.com/uzh-rpg/rpg_esim), or a real event camera via
`dv-processing` / `metavision-sdk`. Only Step 1 of the notebook changes;
the SNN consumes the same tuples either way.

## Column

Draft of the column itself lives at
[`Arpan-206/XRDS-SNN-WriteUp`](https://github.com/Arpan-206/XRDS-SNN-WriteUp).

## License

The code (notebook and `make_figures.py`) is released under the
[MIT License](LICENSE). The sample clip `wave.mp4` is
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
(attribution above).
