# Project Progress Log

## [2026-05-07] - Initial Setup
- Initialized folder structure and environment.
- Created project roadmap in `doc/PLAN.md`.

## [2026-05-07] - Data & Preprocessing
- Implemented `src/dataset.py` using `moabb` to fetch BCI IV 2a dataset.
- Implemented `src/preprocess.py` with 8-30Hz bandpass filtering and ICA.

## [2026-05-07] - Model & Training
- Implemented EEGNet in `models/arch/cnn_ts.py`.
- Developed `src/train.py` with PyTorch Dataset and training loop.
- Verified pipeline with Subject 1 (Acc increased from 25% to ~56% in 10 epochs).

## [2026-05-07] - Integration & Testing
- Refined FastAPI inference in `api/main.py` and `api/utils.py`.
- Created `api/test_client.py` for end-to-end verification.
- Verified real-time inference pipeline with sample data from Subject 1.
- Ensured consistent preprocessing between training and inference.

## [2026-05-07] - Real-Time Simulation
- Implemented `api/lsl_simulator.py` to play back recorded EEG data over LSL.
- Implemented `api/lsl_receiver.py` to bridge LSL streams to the Inference API.
- Enabled full hardware-free testing of the real-time processing loop.
