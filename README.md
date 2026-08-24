# ECG Arrhythmia Mixture-of-Experts (MoE) Classifier

## Overview

This project explores a **Mixture-of-Experts (MoE) neural network for ECG arrhythmia classification**.

My initial approach used a **hybrid 1D Convolutional Neural Network (CNN) and Multilayer Perceptron (MLP)** classifier. The CNN extracted features from ECG signals, while the MLP used those learned features to predict the arrhythmia class.

However, the ECG dataset contains significant **class imbalance**, meaning some arrhythmia classes have substantially more training examples than others. A single classifier trained across all classes may consequently become biased toward the dominant classes.

To investigate an alternative approach, I developed a **Mixture-of-Experts architecture**.

Rather than asking one model to learn how to classify every type of ECG signal, the MoE provides **five smaller expert networks** that can learn different representations of the data. A learned **router network** determines how strongly each expert should contribute to the final prediction.

## Motivation

The primary research idea is:

> Can a Mixture-of-Experts architecture help address challenges associated with an imbalanced ECG dataset by allowing multiple expert networks to specialize in different arrhythmia patterns?

The architecture contains five experts corresponding to the five arrhythmia classes in the classification problem. Importantly, the experts are **not explicitly assigned an arrhythmia class during training**. Instead, specialization is learned automatically through backpropagation.

This makes it possible to investigate whether individual experts naturally become more important for particular types of ECG signals.

## Architecture

The model follows the general pipeline:

```text
                    ECG Signal
                        |
                        v
              +-------------------+
              |  CNN Feature      |
              |  Extractor        |
              +-------------------+
                        |
                        v
                Latent Vector
                        |
             +----------+----------+
             |                     |
             v                     v
        +----------+        +-------------+
        |  Router  |        |   Experts   |
        +----------+        +-------------+
             |              /  /  |  \  \
             |             E1 E2  E3  E4 E5
             |              \  \  |  /  /
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
```

### 1. CNN Feature Extractor

The shared feature extractor uses **1D convolutions** to identify useful morphological and temporal features within the ECG signal.

The extracted feature maps are passed through adaptive average pooling and flattened into a compact **latent feature vector**.

```text
Raw ECG
   |
Conv1D
   |
SiLU
   |
Conv1D
   |
Adaptive Average Pooling
   |
Flatten
   |
Latent Representation
```

This latent representation is supplied to both the router and the five expert networks.

### 2. Router

The router is a small neural network that determines how much each expert should contribute for a particular ECG.

The router first generates five logits:

```text
Latent Vector
     |
Linear Layer
     |
[logit1, logit2, logit3, logit4, logit5]
```

Softmax converts these logits into normalized routing weights:

```text
[0.10, 0.05, 0.65, 0.15, 0.05]
```

The weights sum to 1 and represent the relative contribution of each expert for that sample.

### 3. Expert Networks

Five independent MLP expert networks receive the same latent ECG representation.

```text
                    Latent Vector
                   /   /  |  \   \
                  /   /   |   \   \
                E1   E2   E3   E4   E5
```

Each expert independently transforms the latent representation.

Although the architecture contains the same number of experts as arrhythmia classes, the experts are not hard-coded to particular classes. The goal is to allow **learned specialization** to emerge during training.

### 4. Weighted Expert Combination

All five experts are evaluated and their outputs are combined according to the router probabilities.

For five experts:

```text
MoE Output =
    w1 * Expert1(h)
  + w2 * Expert2(h)
  + w3 * Expert3(h)
  + w4 * Expert4(h)
  + w5 * Expert5(h)
```

where `h` is the latent ECG representation and `w1...w5` are the routing weights.

This implementation therefore uses **soft routing**: every expert can contribute to a prediction, but the router determines the magnitude of each contribution.

### 5. Classification

The weighted combination of the expert outputs forms a new MoE representation.

```text
Weighted Expert Representation
              |
              v
       Linear Classifier
              |
              v
          5 Logits
              |
              v
    Arrhythmia Prediction
```

The complete network can be trained end-to-end through backpropagation.

## Research Direction

The original CNN-MLP architecture provides a useful baseline for determining whether the MoE architecture offers measurable benefits.

The planned comparison includes:

* Classification accuracy
* Per-class precision and recall
* Macro and weighted F1 scores
* Confusion matrices
* Performance on minority arrhythmia classes
* Router weight distributions
* Expert utilization by arrhythmia class

The router outputs are particularly useful for evaluating whether **expert specialization actually emerges**.

For example, if one expert consistently receives higher routing weights for a particular arrhythmia class while another expert is favored for a different class, this may provide evidence that the MoE has learned meaningful specialization.

Conversely, if nearly all samples are routed primarily toward the same expert, this may indicate **expert collapse** and motivate additional routing or load-balancing strategies.

## Research Hypothesis

The central hypothesis of this experiment is that distributing representation learning across multiple specialized experts may improve classification of an imbalanced ECG dataset compared with relying on a single shared CNN-MLP classifier.

The MoE architecture does not inherently guarantee a solution to class imbalance. Instead, this project investigates whether **learned expert specialization provides a useful mechanism for handling heterogeneous and unevenly represented arrhythmia patterns**.

## Current Status

The initial MoE architecture has been implemented in PyTorch and consists of:

* Shared 1D CNN feature extractor
* Adaptive average pooling
* Latent ECG representation
* Learned softmax router
* Five independent MLP experts
* Soft weighted expert aggregation
* Final classification head

The next stage is to train the model, evaluate its classification performance, analyze router behavior, and compare the results against the original CNN-MLP baseline.
