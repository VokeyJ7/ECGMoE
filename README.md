# ECG Arrhythmia Mixture-of-Experts (MoE) Classifier

## Overview

This project investigates a **Mixture-of-Experts (MoE) neural network for ECG arrhythmia classification**.

My original approach used a **1D CNN + MLP classifier**. Because the ECG dataset is significantly imbalanced, I developed a GPU-accelerated MoE architecture to investigate whether multiple expert networks can improve classification of underrepresented arrhythmia classes.

The model contains a shared CNN feature extractor, five expert networks, and a learned router that determines how strongly each expert contributes to the final prediction.

Current best performance:

- **Peak Training Accuracy:** 99.901%
- **Peak Testing Accuracy:** 99.957%
- **Minimum Average Testing Loss:** 0.0015
- **Lowest Recorded 100-Batch Training Loss:** 0.0006

---

## Motivation

The primary research question is:

> Can a Mixture-of-Experts architecture improve ECG arrhythmia classification on an imbalanced dataset by allowing expert networks to learn complementary or specialized representations?

The architecture contains five experts for a five-class arrhythmia problem. However, the experts are **not explicitly assigned to specific arrhythmia classes**. Their specialization is learned automatically through backpropagation.

---

## Architecture

```text
                    ECG Signal
                        |
                        v
                CNN Feature Extractor
                        |
                        v
                  Latent Vector
                        |
             +----------+----------+
             |                     |
             v                     v
           Router              5 Experts
             |                E1 E2 E3 E4 E5
             |                     |
             +----------+----------+
                        |
                        v
               Weighted Combination
                        |
                        v
                   Classifier
                        |
                        v
              Arrhythmia Prediction

### CNN Feature Extractor

The shared feature extractor uses 1D convolutions to learn morphological and temporal ECG features.

```text
ECG -> Conv1D -> SiLU -> Conv1D -> Adaptive Pooling -> Flatten
```

The resulting latent vector is sent to both the router and expert networks.

### Router

The router produces five logits that are converted into normalized routing weights using softmax.

Example:

```text
[0.10, 0.05, 0.65, 0.15, 0.05]
```

These weights determine how strongly each expert contributes to the prediction.

### Expert Networks

All five MLP experts receive the same latent ECG representation and independently transform it.

The experts are not hard-coded to specific arrhythmias. Their behavior is learned during training.

### Weighted Expert Combination

The expert outputs are combined using the router weights:

```text
MoE Output =
    w1 * Expert1(h)
  + w2 * Expert2(h)
  + w3 * Expert3(h)
  + w4 * Expert4(h)
  + w5 * Expert5(h)
```

This implementation uses **soft routing**, meaning every expert can contribute to a prediction.

### Classification

The weighted expert representation is passed to a final linear classifier that produces logits for the five arrhythmia classes.

The entire architecture is trained end-to-end using backpropagation.

---

## Results

| Metric | Result |
| --- | --- |
| Peak Training Accuracy | **99.901%** |
| Peak Testing Accuracy | **99.957%** |
| Minimum Average Testing Loss | **0.0015** |
| Lowest 100-Batch Training Loss | **0.0006** |

The model achieved strong aggregate classification performance. However, because the dataset is imbalanced, overall accuracy alone is not sufficient to determine whether the MoE architecture benefits minority classes.

---

## Research Comparison

The primary thesis comparison is:

```text
Traditional CNN-MLP
        vs.
Mixture-of-Experts CNN
```

Evaluation will include:

- Accuracy
- Training and testing loss
- Per-class precision
- Per-class recall
- Per-class F1 score
- Macro and weighted F1
- Confusion matrix
- Minority-class performance
- Router weight distributions
- Expert utilization

---

## Router and Expert Analysis

Router outputs will be analyzed by true arrhythmia class to determine whether expert specialization emerges.

Possible behaviors include:

- **Class specialization:** experts become associated with particular arrhythmias
- **Feature specialization:** experts focus on different ECG characteristics
- **Distributed specialization:** multiple experts collaborate across classes

The analysis will also check for **expert collapse**, where the router relies too heavily on only one or two experts.

---

## Research Hypothesis

The central hypothesis is that distributing representation learning across multiple expert networks may improve classification of an imbalanced ECG dataset compared with relying on a single shared classifier.

The MoE architecture does not inherently solve class imbalance. Instead, this project investigates whether **learned expert specialization provides measurable benefits for heterogeneous and underrepresented arrhythmia patterns**.

---

## Current Status

The GPU-accelerated MoE model has been implemented and trained in PyTorch.

Current architecture:

- Shared 1D CNN feature extractor
- Adaptive average pooling
- Learned softmax router
- Five MLP expert networks
- Soft weighted expert aggregation
- Five-class classification head

The project is currently in the **evaluation and analysis stage**.

---

## Next Steps

- Generate per-class precision, recall, and F1 scores
- Construct confusion matrices
- Analyze minority-class performance
- Measure router weights by arrhythmia class
- Evaluate expert utilization
- Check for expert collapse
- Compare results against the original CNN-MLP baseline

The broader goal is to determine whether the MoE architecture provides meaningful advantages beyond high overall accuracy, particularly for imbalanced ECG classification.
