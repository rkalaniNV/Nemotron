# Guidebook assets

Generated figures for the [Supervised Fine-Tuning Guidebook](../README.md).

The PNG files are publication assets derived from the reviewed ablation tables.
Update the source evidence and regenerate the corresponding figure together;
do not hand-edit plotted values inside an image.

Scores are Nemotron-3-Nano-30B-A3B on MILU, GSM8K-Indic, IndicIFEval and IndiVibe,
reasoning-on, each arm at its selected checkpoint and anchored to its own
initialisation. Every value was recomputed from the raw evaluation artefacts and
cross-checked against the harness's own `results.yml`.
