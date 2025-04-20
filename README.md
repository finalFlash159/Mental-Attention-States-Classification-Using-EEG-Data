# Mental Attention States Classification Using EEG Data

This project implements a mental state classification system using EEG data to detect three distinct cognitive states: focused, unfocused, and drowsy. The system utilizes both time-domain and frequency-domain approaches for feature extraction and classification.

## Project Overview

The project implements multiple approaches to classify mental states:
- Time-domain analysis using 1D CNN-LSTM hybrid model
- Frequency-domain analysis using traditional ML approaches (Random Forest and SVM)
- Comprehensive signal processing pipeline including ICA for artifact removal

## Dataset Description

The data was collected from 5 participants over 7 experimental sessions, with each session lasting 45-55 minutes. Participants operated a simulated train under controlled conditions:

- First 10 minutes: Focused state - actively controlling the simulation
- Next 10 minutes: Unfocused state - stopped monitoring but remained awake
- Final 10 minutes: Drowsy state - allowed to relax and doze

### Data Collection Details
- 14 EEG channels: AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4
- Sampling rate: 128 Hz
- Channel locations cover frontal, temporal, parietal, and occipital regions
- Includes gyroscope data (GYROX, GYROY) for motion artifact detection

## Methodology

### Signal Processing Pipeline

1. Preprocessing
   - Bandpass filtering (0.3-30 Hz) to remove DC offset and high-frequency noise
   - ICA (Independent Component Analysis) for artifact removal:
     * Identifies and removes eye blinks, muscle artifacts, and cardiac signals
     * Uses FastICA algorithm with 14 components
     * Manual component selection based on topographic maps and time series
   - Motion artifact detection using gyroscope data

2. Feature Engineering
   
   a. Time Domain Approach:
   - Raw signal windows (30 seconds with 25% overlap)
   - Minimal preprocessing to preserve temporal patterns
   - Direct input to CNN-LSTM model

   b. Frequency Domain Approach:
   - Power spectral density estimation using Welch's method
   - Feature extraction from standard frequency bands:
     * Delta (0.5-4 Hz): Deep sleep indicators
     * Theta (4-8 Hz): Drowsiness and meditation
     * Alpha (8-13 Hz): Relaxed wakefulness
     * Beta (13-30 Hz): Active thinking and focus
   - Statistical features:
     * Mean power in each band
     * Peak frequency
     * Spectral entropy
     * Band power ratios

### Classification Models

1. CNN-LSTM Hybrid (Time Domain):
   - 3 Conv1D layers with max pooling
   - 2 LSTM layers (128, 64 units)
   - Dropout layers for regularization
   - Softmax output for 3-class classification

2. Random Forest (Frequency Domain):
   - 1000 trees
   - Maximum depth: 10
   - Feature importance-based selection
   - Class weight balancing

3. SVM (Frequency Domain):
   - RBF kernel
   - Grid search for hyperparameter optimization
   - Feature selection using Random Forest importance scores

## Results

Performance metrics across different models:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|---------|-----------|
| Random Forest | 0.81 | 0.81 | 0.81 | 0.81 |
| CNN-LSTM | 0.74 | 0.73 | 0.74 | 0.73 |
| SVM (features selected) | 0.73 | 0.74 | 0.73 | 0.73 |

### Model Characteristics
- CNN-LSTM: Best at detecting state transitions and temporal patterns
- Random Forest: Excellent performance with frequency domain features
- SVM: Strong baseline performance, improved with feature selection

## Key Findings

1. Frequency Domain Analysis:
   - Most discriminative features found in alpha and beta bands
   - Spectral power ratios highly effective for state discrimination
   - ICA crucial for improving signal quality

2. Time Domain Analysis:
   - CNN-LSTM effectively learns temporal dependencies
   - Direct signal processing reduces information loss
   - More robust to individual variations

3. Classification Performance:
   - Highest accuracy in distinguishing focused vs. drowsy states
   - Most challenging: differentiating unfocused from other states
   - Random Forest shows best overall performance

## Future Work

1. Real-time Implementation:
   - Optimize processing pipeline for online analysis
   - Develop streaming data handling
   - Reduce classification latency

2. Model Improvements:
   - Investigate deep learning architectures for frequency domain
   - Develop hybrid feature extraction approaches
   - Implement attention mechanisms for temporal modeling

3. Applications:
   - Integration with attention monitoring systems
   - Driver drowsiness detection
   - Workplace safety monitoring

## Contributors

- Vo Minh Thinh
- Nguyen Truong Thinh
- Tran Binh Phuong
- Nguyen Hong Son

