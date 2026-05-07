# NeuroRehab-BCI Project Plan

## Objective
Develop an end-to-end pipeline for Motor Imagery (MI) classification using EEG signals and Deep Learning.

## Phases

### Phase 1: Environment & Data Acquisition
- [ ] Initialize folder structure and `requirements.txt`.
- [ ] Script to download/load BCI Competition IV-2a dataset using `mne.datasets`.
- [ ] Exploratory Data Analysis (EDA) and visualization.

### Phase 2: Preprocessing (src/preprocess.py)
- [ ] Band-pass Filtering (8-30 Hz).
- [ ] Artifact Removal using ICA.
- [ ] Epoching and labeling.
- [ ] Signal normalization/scaling.

### Phase 3: Model Architecture (models/arch/)
- [ ] Implement EEGNet (Temporal-Spatial Convolution).
- [ ] Implement TS-CNN (Alternative architecture).
- [ ] Hyperparameter tuning and validation strategy (Cross-validation).

### Phase 4: Training & Evaluation (src/train.py)
- [ ] PyTorch Dataset and DataLoader implementation.
- [ ] Training loop with logging (Loss, Accuracy, F1-Score).
- [ ] Model serialization (saving weights).

### Phase 5: Real-Time Inference (api/)
- [ ] FastAPI setup.
- [ ] Real-time windowing helper.
- [ ] Prediction endpoint.

### Phase 6: Documentation & Finalization
- [ ] Final report and performance metrics.
- [ ] Setup instructions.

## Documentation Strategy
- `doc/PROGRESS.md`: Log of changes and completed tasks.
- `doc/ARCH.md`: Technical details of the model and preprocessing pipeline.
- `doc/TODO.md`: Current tasks for the next session.
