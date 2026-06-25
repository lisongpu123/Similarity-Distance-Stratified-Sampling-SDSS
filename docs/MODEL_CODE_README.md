# Model and Training Code Included

This release contains two layers of code:

1. **Core SDSS method implementation**
   - `sdsskg/similarity.py`: positive-centroid cosine distance computation.
   - `sdsskg/sampling.py`: SDSS3.2 distance-strata partitioning and hard/middle/easy profile sampling.

2. **Prediction model / training pipeline**
   - `sdsskg/model.py`: MLP prediction backbone (`SDSSKGClassifier`).
   - `sdsskg/train_eval.py`: PyTorch training, threshold selection, and Test1/Test2 evaluation.
   - `sdsskg/run_experiment_clinical_ablation.py`: reference runner for the SDSS distance-only experiments.
   - `sdsskg/run_experiment.py`: reference single-experiment runner.
   - `sdsskg/reselect_sdss32_final_profile_v3.py`: reconstructs final SDSS3.2-Final manuscript tables from archived profile-grid summaries.

The private EMR data are not included.  The archived profile-grid outputs are included so the manuscript tables can be audited without exposing patient-level records.
