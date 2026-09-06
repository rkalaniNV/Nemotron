# Guidebook assets

Generated figures for the [Supervised Fine-Tuning Guidebook](../README.md).

The PNG files are publication assets derived from the reviewed ablation tables.
Update the source evidence and regenerate the corresponding figure together;
do not hand-edit plotted values inside an image.

`make_figures.py` regenerates every figure from the values transcribed at the top
of each function, so a reader can check a chart against the source report without
running anything:

```bash
python3 make_figures.py
```

Scores are Nemotron-3-Nano-30B-A3B on MILU, GSM8K-Indic, IndicIFEval and IndiVibe,
reasoning-on, each arm at its selected checkpoint and anchored to its own
initialisation. Every value was recomputed from the raw evaluation artefacts and
cross-checked against the harness's own `results.yml`.
