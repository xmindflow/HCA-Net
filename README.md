# HCA-Net: Hierarchical Context Attention Network for Intervertebral Disc Semantic Labeling

[![arXiv](https://img.shields.io/badge/arXiv-2311.12486-b31b1b.svg)](https://arxiv.org/pdf/2311.12486.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Accurate and automated segmentation of intervertebral discs (IVDs) in medical images is crucial for assessing spine-related disorders, such as osteoporosis, vertebral fractures, or IVD herniation. We present HCA-Net, a novel contextual attention network architecture for semantic labeling of IVDs, with a special focus on exploiting prior geometric information. Our approach excels at processing features across different scales and effectively consolidating them to capture the intricate spatial relationships within the spinal cord. To achieve this, HCA-Net models IVD labeling as a pose estimation problem, aiming to minimize the discrepancy between each predicted IVD location and its corresponding actual joint location. In addition, we introduce a skeletal loss term to reinforce the model's geometric dependence on the spine. This loss function is designed to constrain the model's predictions to a range that matches the general structure of the human vertebral skeleton. As a result, the network learns to reduce the occurrence of false predictions and adaptively improves the accuracy of IVD location estimation. Through extensive experimental evaluation on multi-center spine datasets, our approach consistently outperforms previous state-of-the-art methods on both MRI T1w and T2w modalities.

<div align="center" float="left">
  <img width="400" alt="HCA-Net" src="https://github.com/xmindflow/HCA-Net/assets/6207884/49c9e0e8-d80d-4c15-9686-e1f0ae4d0092">
  <br>
  Caption: Structure of the proposed HCA-Net method for IVD semantic labeling
</div>

## Key Features
- **Hierarchical Context Attention**: Exploits contextual information for accurate IVD segmentation.
- **Pose Estimation Approach**: Models IVD labeling as a pose estimation problem.
- **Skeletal Loss**: Reinforces geometric dependence on the spine structure, reducing false predictions.
- **Multi-Modality Support**: Extensively evaluated on MRI T1w and T2w modalities.

## Getting Started

### Prerequisites
- Linux Machine
- Python 3.8 or earlier version
- PyTorch
- Other dependencies on `requirements.txt`

### Installation

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/yourusername/HCA-Net.git
cd HCA-Net
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```

### Dataset
You can follow [this link](https://github.com/rezazad68/Deep-Intervertebral-Disc-Labeling) to prepare the dataset and use it or simply download the prepared dataset from [this link](https://mega.nz/folder/ABUjSSqT#KE7L1Km3NHTAuil7SoHDeg) (should be less than 160 MB) and continue with the rest of the steps.

### Usage
You can find the list of available arguments in `src/main.py` for modification.

**Example commands for using HCA-Net:**

```bash
# Train
python src/main.py -mode "train" --name "v01" --modality "t1" --epoch 500 --lr 0.00025 --train-batch 3

# Test
python src/main.py -mode "test"  --name "v01" --modality "t1"
```

> Note: _To achieve the best result, consider all possible disk points per channel (final `out` block) then use [this link](https://github.com/rezazad68/intervertebral-lookonce) for post-processing to obtain the final skeleton for evaluation._
>

## Experimental Results
<p align="center">
  <img width="900" alt="image" src="https://github.com/xmindflow/HCA-Net/assets/6207884/1195d13c-4f63-4b58-b644-c3c5d654d07e">
  <br>
  <img width="900" alt="image" src="https://github.com/xmindflow/HCA-Net/assets/6207884/f82cd0d2-02b1-4bb1-bb07-0d615af7f5eb">
</p>

A notable illustration of intervertebral disc semantic detection and labeling in the test dataset is shown in the T1w (first row) and T2w MRI modalities (second row). In the first column, the network input is showcased, the second column displays the ground truth, and the third and final column exhibits the outcome of the last `out` block.

<hr>

<p align="center">
  <img width="900" alt="image" src="https://github.com/xmindflow/HCA-Net/assets/6207884/b2063058-876b-46dc-9ba5-c6c7bf91e492">
  <br>
<!-- Intervertebral disc semantic labeling on the spine generic public dataset. DTT indicates the Distance To the Target -->
</p>

<hr>

<p align="center">
  <img width="700" alt="image" src="https://github.com/xmindflow/HCA-Net/assets/6207884/d0511948-bfeb-44ed-983e-6c13543e3063">
  <br>
  Comparison of results on T1w (a-b) and T2w (c-d) MRI modalities between the proposed HCA-Net (b and d) and the pose estimation method [13] (a and c). Green dots denote ground truth.
</p>

## Citation
If you find HCA-Net useful for your work, please consider citing us:

```bibtex
@misc{bozorgpour2023hcanet,
      title={HCA-Net: Hierarchical Context Attention Network for Intervertebral Disc Semantic Labeling}, 
      author={Afshin Bozorgpour and Bobby Azad and Reza Azad and Yury Velichko and Ulas Bagci and Dorit Merhof},
      year={2023},
      eprint={2311.12486},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## References
- Deep Intervertebral Disc Labeling [https://github.com/rezazad68/Deep-Intervertebral-Disc-Labeling]
- Intervertebral Lookonce [https://github.com/rezazad68/intervertebral-lookonce]
