# Data access and privacy

The raw electronic medical record data used in this study are not publicly distributed in this repository because they contain sensitive patient-level clinical information and are subject to institutional ethics approval and hospital data-governance restrictions.

Qualified researchers may request access to de-identified data from the corresponding author. Access will be considered after approval by the relevant ethics committee and data-governance authority and may require a data use agreement.

To facilitate evaluation of the computational workflow without exposing patient-level records, this repository provides:

- configuration templates under `configs/`;
- synthetic/toy examples under `examples/`;
- code for SDSS sampling, MLP training, and optional KG/DCKG feature integration under `sdsskg/`;
- result-reconstruction scripts under `scripts/`; and
- archived result files under `results/archived_profile_grid/` for reproducing the manuscript tables.

For private reruns, place local de-identified Train/Test1/Test2 files under `data/` or update the YAML paths to point to their secure local locations. Patient identifiers and original hospital EMR exports should not be committed to this repository.
