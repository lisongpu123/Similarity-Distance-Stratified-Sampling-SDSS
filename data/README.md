# Data availability

The raw electronic medical records used in the study contain sensitive clinical information and are **not included** in this public release.

This repository includes:

- SDSS implementation code;
- archived, de-identified experiment summary files generated from the original profile-grid experiments;
- final result provenance tables reconstructed from those archived summaries.

To run the training scripts on private data, place your de-identified Train/Test1/Test2 files in this directory and update the YAML files under `configs/`. The expected label column accepts either 0/1 or Chinese labels such as `正样本` and `负样本`.
