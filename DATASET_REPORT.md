# EEG Dataset Inspection Report

> Licensing note: the official [PhysioNet EDF release](https://physionet.org/content/eegmat/1.0.0/) is marked ODC-BY 1.0. The Kaggle page for this report's converted CSV distribution lists the license as unknown. Keep raw CSVs out of public Git until their redistribution terms are confirmed.

## 1. Dataset Identity

- **Dataset Name:** PhysioNet EEG During Mental Arithmetic Tasks (CSV Distribution)
- **Dataset Source:** PhysioNet (`doi:10.13026/C2JQ1P`) / Zyma et al. (2018); converted CSV copy distributed on Kaggle as the "Complete EEG Dataset" (`amananandrai/complete-eeg-dataset`, license currently listed as unknown)
- **Dataset Version:** 1.0.0
- **Recording Origin:** Department of Human and Animal Physiology, Taras Shevchenko National University of Kyiv
- **Task:** Intensive mental arithmetic task (serial subtraction of a 2-digit number from a 4-digit minuend, e.g., $3141 - 13 - 13 \dots$) performed silently for 60+ seconds
- **Experimental Condition:** Active cognitive computation state (corresponding to task session `_2` from the original protocol)
- **Hardware Platform:** NeuroCom Professional 23-channel EEG system (XAI-MEDICA), $500\text{ Hz}$ acquisition
- **File Format:** Comma-Separated Values (`.csv`), headerless, IEEE 754 float text representations

---

## 2. Local Directory Structure

The dataset resides locally in `./archive/`. A recursive scan reveals no subdirectories and exactly 36 CSV files:

```
archive/
├── s00.csv    (4,431,745 bytes)
├── s01.csv    (4,427,919 bytes)
├── s02.csv    (4,433,297 bytes)
├── s03.csv    (4,453,015 bytes)
├── s04.csv    (4,426,745 bytes)
├── s05.csv    (4,430,693 bytes)
├── s06.csv    (4,463,855 bytes)
├── s07.csv    (4,439,460 bytes)
├── s08.csv    (4,457,917 bytes)
├── s09.csv    (4,455,457 bytes)
├── s10.csv    (4,449,821 bytes)
├── s11.csv    (4,473,484 bytes)
├── s12.csv    (4,430,708 bytes)
├── s13.csv    (4,451,384 bytes)
├── s14.csv    (4,431,943 bytes)
├── s15.csv    (4,429,166 bytes)
├── s16.csv    (4,459,102 bytes)
├── s17.csv    (4,437,582 bytes)
├── s18.csv    (4,474,862 bytes)
├── s19.csv    (4,453,906 bytes)
├── s20.csv    (4,441,976 bytes)
├── s21.csv    (4,455,502 bytes)
├── s22.csv    (4,436,676 bytes)
├── s23.csv    (4,438,058 bytes)
├── s24.csv    (4,452,582 bytes)
├── s25.csv    (4,422,422 bytes)
├── s26.csv    (4,452,553 bytes)
├── s27.csv    (4,459,897 bytes)
├── s28.csv    (4,442,011 bytes)
├── s29.csv    (4,432,469 bytes)
├── s30.csv    (4,442,453 bytes)
├── s31.csv    (4,446,514 bytes)
├── s32.csv    (4,437,571 bytes)
├── s33.csv    (4,452,535 bytes)
├── s34.csv    (4,439,730 bytes)
└── s35.csv    (4,452,152 bytes)
```

- **Total Directories:** 0 subdirectories inside `./archive/`
- **Total Files:** 36 CSV files
- **Total Disk Size:** $160,011,043\text{ bytes}$ ($\approx 152.60\text{ MB}$)
- **Naming Pattern:** `s{subject_id:02d}.csv` where `subject_id` ranges from `00` to `35`

---

## 3. Recording Statistics

| Metric | Value |
| :--- | :--- |
| **Total Number of Recordings** | 36 recordings (1 recording per subject) |
| **Total Number of Subjects** | 36 distinct subjects |
| **Channels per Recording** | 19 channels |
| **Samples per Channel per Recording** | 31,000 samples |
| **Duration per Recording** | 62.00 seconds |
| **Total Samples per Channel (Dataset-Wide)** | $1,116,000\text{ samples}$ |
| **Total Data Points (All Channels)** | $21,204,000\text{ floats}$ |
| **Total Cumulative EEG Duration** | $2,232.0\text{ seconds}$ ($37.20\text{ minutes}$) |
| **Average Recording Duration** | 62.00 seconds |
| **Min Recording Duration** | 62.00 seconds |
| **Max Recording Duration** | 62.00 seconds |
| **Disk Size** | $152.60\text{ MB}$ |

---

## 4. Subject Information

- **Subject Identifiers:** Formatted as `s00`, `s01`, `s02`, $\dots$, `s35` (36 subjects total).
- **Subject-to-Recording Mapping:** Strict 1-to-1 correspondence:
  - `s00` $\to$ `archive/s00.csv`
  - `s01` $\to$ `archive/s01.csv`
  - $\dots$
  - `s35` $\to$ `archive/s35.csv`
- **Subject Identity Retention:** Each file is isolated by subject. This allows leak-free subject-wise cross-validation and splitting (e.g., 28 subjects for training, 4 subjects for validation, 4 subjects for test).
- **Cohort Demographics (from study documentation):** 36 healthy young adults (9 males, 27 females), aged 18–26 years ($M = 21.5, SD = 1.56$), right-handed, normal or corrected-to-normal vision, no history of neurological or psychiatric disorders.

---

## 5. EEG Channels

The CSV files contain exactly 19 channels per recording. These 19 channels correspond to the standardized International 10-20 system:

| Column Index | Channel Name | Anatomical Region | Type |
| :--- | :--- | :--- | :--- |
| Col 0 | **Fp1** | Left Prefrontal | Scalp EEG |
| Col 1 | **Fp2** | Right Prefrontal | Scalp EEG |
| Col 2 | **F3** | Left Frontal | Scalp EEG |
| Col 3 | **F4** | Right Frontal | Scalp EEG |
| Col 4 | **F7** | Left Antero-temporal | Scalp EEG |
| Col 5 | **F8** | Right Antero-temporal | Scalp EEG |
| Col 6 | **T3** | Left Mid-temporal | Scalp EEG |
| Col 7 | **T4** | Right Mid-temporal | Scalp EEG |
| Col 8 | **C3** | Left Central (Motor/Sensory) | Scalp EEG |
| Col 9 | **C4** | Right Central (Motor/Sensory) | Scalp EEG |
| Col 10 | **T5** | Left Postero-temporal | Scalp EEG |
| Col 11 | **T6** | Right Postero-temporal | Scalp EEG |
| Col 12 | **P3** | Left Parietal | Scalp EEG |
| Col 13 | **P4** | Right Parietal | Scalp EEG |
| Col 14 | **O1** | Left Occipital | Scalp EEG |
| Col 15 | **O2** | Right Occipital | Scalp EEG |
| Col 16 | **Fz** | Midline Frontal | Scalp EEG |
| Col 17 | **Cz** | Midline Central (Vertex) | Scalp EEG |
| Col 18 | **Pz** | Midline Parietal | Scalp EEG |

- **Total Channels in CSV:** 19
- **EEG Channels:** 19 (100% scalp EEG)
- **Non-EEG Channels in CSV:** 0 (Original hardware channels for ECG, reference $A_1-A_2$, and annotations were excluded during CSV export)
- **Signal Units:** Microvolts ($\mu\text{V}$)

---

## 6. Sampling Frequency

- **Sampling Frequency:** $500.0\text{ Hz}$
- **Sampling Interval ($\Delta t$):** $0.002\text{ s} = 2.0\text{ ms}$
- **Consistency:** 100% uniform across all 36 recordings and all channels.

---

## 7. Recording Duration

- **Duration per File:** $62.00\text{ seconds}$
- **Duration Formula:** $\text{Duration} = \frac{31,000\text{ samples}}{500\text{ samples/sec}} = 62.0\text{ s}$
- **Total Dataset EEG Time:** $2,232.0\text{ s} = 37.20\text{ minutes}$
- **Uniformity:** Identical across all 36 files.

---

## 8. Labels / Classes

- **Embedded Labels:** **None.** The CSV files in `./archive/` contain only 19 continuous numerical columns representing EEG potentials. No column encodes class labels or trial markers.
- **Header Status:** Files are headerless.
- **External Dataset Class Definitions:**
  - In the original PhysioNet source study, subjects were stratified based on arithmetic calculation rate:
    - *Good counters* (Group G, 24 subjects): Count quality = 1
    - *Bad counters* (Group B, 12 subjects): Count quality = 0
  - In `./archive/`, this metadata is not bundled in the directory.
- **Digital Twin Modeling Implications:**
  - The dataset is immediately ready for **self-supervised next-step / future-window prediction** (modeling the continuous state transition function $s_{t+1} = f(s_t)$ of the neural digital twin).
  - Subject identification (36-class biometric fingerprinting) is also natively available using the subject IDs.

---

## 9. Temporal / Sequential Information

- **Continuity:** The EEG data in each file represents a continuous, uninterrupted 62-second timeseries.
- **Sequential Ordering:** Rows are ordered chronologically with uniform $\Delta t = 2\text{ ms}$.
- **Windowing Feasibility:** Excellent. A continuous 62-second stream can be partitioned into sequential temporal windows:
  - **1.0-second window (500 samples):** 62 non-overlapping windows per subject ($2,232$ total windows).
  - **2.0-second window (1000 samples):** 31 non-overlapping windows per subject ($1,116$ total windows).
  - **Sliding windows with 50% overlap:** Yields double the window count for sequence training.
- **Timestamps:** Implicit via sampling index ($t_k = k \times 0.002\text{ s}$ for $k \in [0, 30999]$).
- **Event Markers:** None in the CSV files.

---

## 10. Data Quality

Read-only automated inspection across all 36 files produced the following findings:

- **Missing Files:** 0 (all expected subjects `s00`–`s35` exist).
- **Unreadable Files:** 0.
- **Corrupted Records:** 0.
- **NaN Count:** 0 across all $21,204,000$ data points.
- **Infinite Values:** 0 across all $21,204,000$ data points.
- **Shape Consistency:** 100% consistent: exactly `(31000, 19)` for all 36 files.
- **Amplitude Range:** Min: $-582.73\ \mu\text{V}$, Max: $+202.96\ \mu\text{V}$.
- **Mean & Standard Deviation:** Mean values per channel fluctuate around $0.0\ \mu\text{V}$ ($\pm 0.2\ \mu\text{V}$); standard deviations range from $6.3\ \mu\text{V}$ to $17.9\ \mu\text{V}$.
- **Physiological Artifacts:** Occasional excursions below $-100\ \mu\text{V}$ or above $+100\ \mu\text{V}$ in frontal electrodes (`Fp1`, `Fp2`) indicate un-rejected blink artifacts and minor muscle noise, which will need filtering and robust scaling in Phase 2.

---

## 11. Potential Problems

1. **Absence of Channel Names in CSV Files:** The files are headerless. Channel alignment must be explicitly preserved via the 19 standard 10-20 channels determined from the physical acquisition configuration.
2. **No Embedded Classification Labels:** Supervised state classification will require either self-supervised state estimation, subject identity modeling, or ingesting the external PhysioNet `subject-info.csv` if cognitive capacity classification is required.
3. **Recording Length (62 s per subject):** While sufficient for sequential dynamic modeling ($31,000$ timesteps per subject), long-term circadian drift or multi-hour fatigue cannot be studied with this dataset.
4. **Physiological & Environmental Noise:** Baseline drift and blink spikes require bandpass filtering (e.g. 0.5–45 Hz) and notch filtering (50 Hz powerline) in Phase 2.

---

## 12. Suitability for the Quantum Neural Digital Twin

The dataset is **highly suitable** for developing a Quantum Neural Digital Twin:

1. **Clean Structured Format:** Uniform dimensions `(31000, 19)` across all 36 subjects, zero missing values, zero NaNs, zero Infs.
2. **Dense Multichannel Cortex Coverage:** 19 channels span the full cortex, providing spatial correlations suitable for graph or convolutional spatial encoders (e.g., EEGNet).
3. **Continuous Temporal Dynamics:** High temporal resolution ($500\text{ Hz}$, $2\text{ ms}$ interval) allows recurrent/bidirectional modeling (BiLSTM / State Space Models) to simulate brain dynamics.
4. **Quantum Embedding Feasibility:** 19 channels can be projected via spatial dimensionality reduction (PCA, spatial projection, or autoencoder bottleneck) down to 4–8 quantum features, matching typical NISQ quantum simulator constraints for Variational Quantum Circuits (VQCs).
5. **Rigorous Subject Separation:** Strict separation across 36 distinct subject files ensures zero leakage during subject-wise train/validation/test evaluation.
