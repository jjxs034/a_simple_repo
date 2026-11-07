# Licence of the InsPLAD-derived files

The following files are derived from InsPLAD-det and are licensed under the licence of InsPLAD-det:

- `data/insplad/` (question sets, cross-fitting fold groups, candidate lists, per-question outputs and their summary);
- `adapters/insplad/` (the four LoRA adapters trained on the InsPLAD questions).

Licence: Creative Commons Attribution Non Commercial 3.0 Unported
(https://creativecommons.org/licenses/by-nc/3.0/legalcode).

Source: Vieira-e-Silva, André Luiz; Felix, Heitor; Simões, Francisco; Teichrieb, Veronica. InsPLAD: A Dataset and
Benchmark for Power Line Asset Inspection in UAV Images. Mendeley Data, 2023, version 1.
https://doi.org/10.17632/5n3fjgvfyz.1

Changes made: no image is redistributed. The questions and their answers were generated from the InsPLAD-det
annotations by fixed rules (`powerbrain/bench.py`, class `sphere` dropped); the candidate lists come from detectors
trained on the InsPLAD-det training partition; the per-question outputs and the adapters come from the models described
in the paper. Image paths are written with the placeholder prefix `INSPLAD_ROOT/`.
