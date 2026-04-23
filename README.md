# Siamese Network with Triplet Loss for Person Re-Identification using PyTorch

This project implements a **Siamese Neural Network** trained with **Triplet Loss** to learn discriminative feature embeddings for **person re-identification**. The model maps visually similar inputs closer in embedding space while pushing dissimilar samples farther apart, enabling robust similarity-based matching.

## Overview

Siamese Networks are designed to learn **relative similarity** rather than direct classification. Instead of predicting class labels, the network learns an embedding function such that:

- Similar inputs → closer in feature space  
- Dissimilar inputs → farther apart  

This project leverages **Triplet Loss**, which operates on:

- **Anchor (A)** — reference image  
- **Positive (P)** — same identity as anchor  
- **Negative (N)** — different identity  

The objective is to enforce:

$$
d(A, P) + \text{margin} < d(A, N)
$$

where:

$$
d(x, y) = ‖f(x) - f(y)‖
$$

This formulation is widely used in:

- Face recognition  
- Signature verification  
- Person re-identification  

## Objectives

- Learn embedding representations using **Triplet Loss**
- Implement a complete **Siamese training pipeline in PyTorch**
- Construct and utilize **Anchor–Positive–Negative (APN)** triplets
- Train a model for **distance-based identity matching**

## Methodology

### 1. Data Preparation

The dataset is structured to generate **triplets dynamically** during training. Each training sample consists of:

- An anchor image  
- A positive image (same class as anchor)  
- A negative image (different class)  

Key aspects:

- Class-wise indexing for efficient sampling  
- Randomized selection for variability  
- Balanced exposure to positive and negative samples  

### 2. Triplet Dataset Construction

A custom dataset class:

- Groups images by identity  
- Samples valid triplets \((A, P, N)\)  
- Applies transformations (resize, normalization, tensor conversion)  

This enables learning of **intra-class compactness** and **inter-class separation**.

### 3. Model Architecture

The Siamese Network consists of:

- A shared **convolutional backbone**
- Fully connected layers projecting to a **low-dimensional embedding space**

Key properties:

- **Weight sharing** across all inputs  
- Produces **fixed-length embeddings**  
- Optimized for **distance-based comparison**  

### 4. Embedding Learning

Each input is passed through the same network:

$$
f(A), \quad f(P), \quad f(N)
$$

The objective:

- Minimize: $$‖f(A) − f(P)‖$$

- Maximize: $$‖f(A) − f(N)‖$$

This transforms the problem into **metric learning**.

### 5. Triplet Loss Function

The loss function is defined as:

$$
\mathcal{L} = \max \left( \|f(A) - f(P)\|^2 - \|f(A) - f(N)\|^2 + \text{margin},\ 0 \right)
$$

Where:

- **margin** enforces separation  
- Loss becomes zero when constraint is satisfied  

This encourages:

- Compact clusters for same identities  
- Clear separation for different identities  

### 6. Training Pipeline

The training process includes:

- Forward pass of triplets  
- Embedding computation  
- Distance calculation  
- Loss evaluation using Triplet Loss  
- Backpropagation and optimization  

Important aspects:

- Batch-wise triplet sampling  
- Stable optimization with margin tuning  
- Efficient computation using GPU (if available)  

### 7. Model Behavior After Training

After training, the model:

- Maps images to embedding vectors  
- Enables similarity comparison using:
  - Euclidean distance  
  - Cosine similarity  

This supports:

- Identity matching  
- Image retrieval  
- Similarity ranking  

## Project Workflow

```
Raw Image Dataset
        ↓
Class-wise Organization
        ↓
Triplet Sampling (Anchor, Positive, Negative)
        ↓
DataLoader (Batch Processing)
        ↓
Siamese Network (Shared Weights)
        ↓
Embedding Generation
        ↓
Triplet Loss Computation
        ↓
Backpropagation & Optimization
        ↓
Trained Embedding Space
        ↓
Similarity-Based Inference
```

## Key Characteristics

- Learns a distance metric instead of explicit class boundaries
- Shared-weight architecture ensures consistent feature extraction
- Triplet-based supervision improves generalization to unseen identities
- Decouples training identities from inference identities

## Applications

- Person re-identification  
- Face verification  
- Image similarity search  
- Duplicate detection  
- Signature verification  

## Technologies Used

- **PyTorch** — model development and training  
- **Torchvision** — preprocessing and transforms  
- **NumPy / PIL** — data handling  
- **Matplotlib** — visualization  

## Summary

This project presents a full implementation of a **Siamese Network trained with Triplet Loss**, focusing on learning meaningful feature embeddings for similarity-based tasks. By shifting from classification to metric learning, the model becomes more flexible and effective for real-world identity matching problems.
