# Model and training code included

This release contains reviewer-visible code for the complete computational workflow, while excluding private patient-level data.

## 1. Core SDSS method implementation

- `sdsskg/similarity.py`: builds text/feature representations and computes cosine distance to the positive centroid.
- `sdsskg/sampling.py`: implements random, all-easy, all-hard, and SDSS distance-stratified negative sampling.

## 2. Prediction model and training pipeline

- `sdsskg/model.py`: MLP prediction backbone (`SDSSKGClassifier`) with optional KG-feature input.
- `sdsskg/train_eval.py`: PyTorch data loaders, training, probability prediction, threshold selection, and evaluation metrics.
- `sdsskg/run_experiment.py`: single SDSS + MLP experiment runner using a YAML configuration.
- `scripts/run_sdss_mlp_training.py`: lightweight reviewer-visible entry point that calls the training runner.
- `sdsskg/run_experiment_clinical_ablation.py` and `sdsskg/run_experiment_new_sampling.py`: reference runners for ablation and alternative sampling comparisons.
- `sdsskg/run_asdss_profile_suite.py`: profile-grid runner used to generate candidate SDSS profiles.
- `sdsskg/reselect_sdss32_final_profile_v3.py`: reconstructs the final SDSS profile selection from archived profile-grid summaries.

## 3. Optional KG/DCKG feature interface

- `sdsskg/kge.py`: optional entity-embedding aggregation interface for KG/DCKG features.
- `docs/DCKG_CONSTRUCTION.md`: pseudocode and implementation notes for KG/DCKG construction and integration with the MLP.

The archived manuscript tables can be reproduced from the included profile-grid outputs without accessing patient-level records. Full retraining requires private de-identified clinical data and local YAML path configuration.
