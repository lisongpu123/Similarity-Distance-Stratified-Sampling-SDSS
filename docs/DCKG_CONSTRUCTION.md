# DCKG / KG feature construction notes

This document provides reviewer-visible pseudocode for the optional diabetes-complication knowledge-graph (DCKG) feature branch and its integration with the MLP classifier. The public repository does not include private clinical entities or hospital-derived patient records.

## Scope

The final SDSS table in the manuscript can be reconstructed from the archived profile-grid outputs without KG files. The KG/DCKG interface is included for code transparency and for private reruns when local entity dictionaries and embeddings are available.

## Inputs

- De-identified clinical records after patient identifiers have been removed.
- Entity columns extracted from symptom text, examination text, diagnosis descriptions, or structured clinical fields.
- An entity dictionary `entity2id.json` mapping normalized clinical entities to integer IDs.
- An entity embedding matrix `entity_emb.npy` aligned with the dictionary.

## Pseudocode

```text
For each de-identified clinical record:
    1. Remove patient identifiers and columns that should not be used as predictors.
    2. Normalize clinical entities, synonyms, and abbreviations.
    3. Map each entity to its integer ID using entity2id.json.
    4. Look up entity embeddings from entity_emb.npy.
    5. Aggregate entity embeddings for the record, usually by mean pooling.
    6. Concatenate the KG/DCKG feature vector with the non-KG feature vector.
    7. Feed the concatenated representation into the MLP classifier.
```

## Implementation locations

- `sdsskg/kge.py`: entity-list to KG-feature aggregation.
- `sdsskg/model.py`: `SDSSKGClassifier`, which accepts the base feature vector and an optional KG feature vector.
- `sdsskg/run_experiment.py`: end-to-end training runner with the `model.use_kg` flag in the YAML configuration.

## Configuration

A YAML configuration can enable or disable the KG branch:

```yaml
model:
  use_kg: false        # set true when local KG files are available
  kg_dim_fallback: 64

kge:
  entity_emb_path: artifacts/entity_emb.npy
  entity2id_json: artifacts/entity2id.json
  pad_token: "<PAD>"
  agg: mean
```

When `use_kg: false`, the final SDSS pipeline uses only the non-KG clinical/text feature representation and the SDSS-selected negative samples. When `use_kg: true`, local KG/DCKG embeddings are aggregated and concatenated before MLP prediction.
