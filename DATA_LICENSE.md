# Licences of the data files

## InsPLAD-derived files

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

## ED and TD per-question records

The files in `data/edtd/` (`questions.csv.gz`, `outputs.csv.gz`, `outputs_rules.csv.gz` and `runs.json`) are licensed
under the Creative Commons Attribution-NonCommercial 4.0 International licence (CC BY-NC 4.0,
https://creativecommons.org/licenses/by-nc/4.0/legalcode).

Attribution: the authors of "PowerBrain: A Detector-Grounded Multimodal Large Language Model for Efficient Power Line
Inspection on Edge Devices" (names withheld during double-blind review).

The licence covers only these de-identified records: keyed hashes of question and image identifiers, dataset, question
form, condition, system, seed, correctness and the three coarse derived fields. The underlying ED and TD data (images,
annotations, questions, reference and model answers, file names) are provided under an agreement with the company,
are not released, and are not covered by this licence.
