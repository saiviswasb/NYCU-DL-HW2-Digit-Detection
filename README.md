# Digit Detection using Faster R-CNN (ResNet50-V2)

## Overview
This repository contains the implementation for an object detection model designed to identify digits in complex images. The model achieved a **0.40 mAP** on the CodaBench leaderboard.

## Compliance & Architecture
- **Backbone:** `FasterRCNN_ResNet50_FPN_V2`
- **Pre-trained Weights:** Only the ResNet50 backbone utilizes pre-trained weights (`ResNet50_Weights.DEFAULT`), in strict compliance with the assignment rules.
- **Detection Head:** The encoder, decoder, and RPN are initialized randomly and trained completely from scratch.
- **Optimizer:** AdamW (`lr=1e-4`, `weight_decay=1e-4`) with a Cosine Annealing Learning Rate Scheduler.

## Leaderboard Result
![Leaderboard Screenshot](leaderboard.png)

## How to Run
1. Ensure the dataset (`train/`, `test/`, and `train.json`) is located in your working directory.
2. The code is designed to run efficiently on a GPU using Automatic Mixed Precision (`torch.amp.autocast`).
3. Execute the training script to train the model for 6 epochs.
4. The script will automatically output a `pred.json` file for evaluation.
