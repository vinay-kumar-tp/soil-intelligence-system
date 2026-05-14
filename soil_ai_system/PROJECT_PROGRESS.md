# Project Progress

## Phase Tracker

Phase: 1 - Data Engineering & Pipeline Build
Status: In progress

## Tasks

### Phase 0 completed
- 0.1 through 0.13: ALL DONE

### Phase 1 Tasks
- 1.1 `validator.py`: Define strict input data schemas using Pydantic. (done)
- 1.2 `cleaner.py`: Handle missing values, outliers, and normalizations. (done)
- 1.3 `encoder.py`: Categorical encoding and label mapping. (done)
- 1.4 `scaler.py`: Standardization and MinMax scaling. (done)
- 1.5 `feature_engineer.py` / `feature_store.py`: Domain-specific indices generation. (done)
- 1.6 Phase 1 Tests: Write and pass Unit Tests for preprocessing. (updated)
- 1.7 `dataset_merger.py`: Merge, validate, and export processed dataset. (done)
- 1.8 Notebooks: Raw inspection + preprocessing validation. (done)

## Issues

- None open.

## Resolved

- Phase 0 foundation complete, committed, and pushed successfully.

## Next Steps

- Run notebook `notebooks/01_dataset_inspection.ipynb` after raw CSVs are placed.
- Run preprocessing pipeline to generate processed dataset + reports.
- Run notebook `notebooks/02_preprocessing_validation.ipynb`.
