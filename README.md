# Digit Detection using Faster R-CNN (ResNet50-V2)

## Overview
This repository contains the implementation for an object detection model designed to identify digits in complex images. The model achieved a **0.40 mAP** on the CodaBench leaderboard.

## Compliance & Architecture
- **Backbone:** `FasterRCNN_ResNet50_FPN_V2`
- **Pre-trained Weights:** Only the ResNet50 backbone utilizes pre-trained weights (`ResNet50_Weights.DEFAULT`), in strict compliance with the assignment rules.
- **Detection Head:** The encoder, decoder, and RPN are initialized randomly and trained completely from scratch.
- **Optimizer:** AdamW (`lr=1e-4`, `weight_decay=1e-4`) with a Cosine Annealing Learning Rate Scheduler.

## Training Convergence
To ensure the randomly initialized detection heads converged properly within the hardware constraints, the model was trained for 6 epochs using Automatic Mixed Precision. 

<img width="704" height="475" alt="image" src="https://github.com/user-attachments/assets/442ff6ae-5a7b-4b28-8f83-a932ec9444d9" />


## Additional Experiment: Inference Filter Tuning
To optimize the CodaBench mAP score, a post-processing ablation study was conducted. By bypassing the default 0.50 confidence filter and lowering the threshold to 0.05, the model safely captured hidden and overlapping digits (maximizing the Recall metric) without introducing excessive noise.

<img width="1189" height="278" alt="image" src="https://github.com/user-attachments/assets/6034c7e3-d2b8-4be3-b006-a76a627ecb88" />


## Leaderboard Result
<img width="1437" height="52" alt="image" src="https://github.com/user-attachments/assets/e95bb804-14f4-4e35-a295-13bc68cb698a" />


## How to Run
1. Ensure the dataset (`train/`, `test/`, and `train.json`) is located in a `./dataset/` directory.
2. Install the required dependencies (PyTorch, Torchvision).
3. Execute `train.py` to train the model.
4. The script will automatically output a `pred.json` file in the `./output/` directory for evaluation.
