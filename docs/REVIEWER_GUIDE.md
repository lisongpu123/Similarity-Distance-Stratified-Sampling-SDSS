# Reviewer guide

## 1. Verify the final reported table

```bash
python scripts/build_final_sdss32_from_archived_grid.py --root results/archived_profile_grid --out results/final_sdss32_provenance
python scripts/check_final_table.py
```

These commands reconstruct the final SDSS table from archived profile-grid files and do not retrain models.

## 2. Inspect the SDSS implementation

Core files:

- `sdsskg/similarity.py`: text embedding and cosine distance to the positive centroid.
- `sdsskg/sampling.py`: distance-stratified hard/middle/easy negative sampling.
- `sdsskg/model.py`: MLP prediction model used after SDSS sampling.
- `sdsskg/train_eval.py`: training, threshold selection, and evaluation utilities.
- `sdsskg/kge.py`: optional KG/DCKG feature encoder interface.
- `docs/DCKG_CONSTRUCTION.md`: pseudocode for KG/DCKG feature construction and MLP integration.

Historical class and file names that contain `sdss32` are retained only for traceability to the archived experiment snapshot. The final manuscript method is reported as **SDSS**.

## 3. Run the toy SDSS demo

```bash
python examples/demo_sdss_sampling.py
```

This demo uses synthetic records and only illustrates the SDSS sampling mechanism.

## 4. Rerun with private de-identified data

The repository does not include raw EMR data. To rerun model training, place de-identified Train/Test1/Test2 files under `data/`, edit YAML paths in `configs/`, and run, for example:

```bash
pip install -r requirements-full.txt
python scripts/run_sdss_mlp_training.py --config configs/dn_two_external_wokg_sdss32.yaml
```

Because the archived manuscript results were generated from a fixed historical experiment snapshot, a current-code rerun may show small numerical differences. The archived profile-grid outputs are therefore included for transparent result provenance.
