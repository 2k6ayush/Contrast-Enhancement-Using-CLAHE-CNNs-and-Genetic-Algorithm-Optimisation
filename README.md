# 🖼️ Contrast Enhancement Using CLAHE, CNNs and Genetic Algorithm Optimisation

**Subject:** Fundamentals of Image Processing  
**Project Type:** Image Enhancement using Traditional Image Processing, Deep Learning and Evolutionary Optimisation  
**Team Size:** 3 Members  
**Final Evaluation:** 24 November 2026

---

## 📌 1. Project Overview

Image enhancement is an important task in digital image processing. Images captured in poor lighting conditions often suffer from low contrast, poor visibility, loss of details and noise.

Traditional enhancement techniques such as Histogram Equalization and Contrast Limited Adaptive Histogram Equalization (CLAHE) can improve image contrast. However, manually selecting their parameters may not always produce optimal results.

This project proposes a hybrid image enhancement system that combines:

1. **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Improves local contrast and reveals hidden details.
2. **CNN (Convolutional Neural Network):** Learns image enhancement patterns from paired low-light and normal-light images.
3. **Genetic Algorithm (GA):** Automatically optimises CLAHE parameters to obtain better enhancement results.

The proposed system aims to improve image contrast, brightness and visibility while preserving important image details and avoiding excessive noise amplification.

The performance of the proposed approach will be compared against traditional image processing techniques and CNN-based enhancement methods.

---

## 🎯 2. Project Objectives

- Improve the contrast and visibility of low-light images.
- Implement traditional image enhancement techniques such as Histogram Equalization and CLAHE.
- Develop a CNN-based image enhancement model.
- Implement a Genetic Algorithm to automatically optimise CLAHE parameters.
- Develop a hybrid enhancement model combining GA-optimised CLAHE and CNN refinement.
- Perform hyperparameter tuning to identify suitable parameter combinations.
- Conduct ablation studies to understand the contribution of individual components.
- Compare traditional, deep learning and proposed enhancement approaches.
- Evaluate image quality using suitable quantitative metrics.
- Develop a user-friendly application for uploading and enhancing images.

---

## 🧠 3. Proposed Methodology

The project consists of three major approaches.

### 3.1 Traditional Image Processing

Traditional image enhancement techniques will be implemented using OpenCV.

Methods:

- Histogram Equalization (HE)
- Contrast Limited Adaptive Histogram Equalization (CLAHE)

CLAHE divides an image into smaller regions and enhances local contrast independently. A clip limit prevents excessive contrast amplification.

The traditional approach acts as the baseline for comparison.

### 3.2 CNN-Based Image Enhancement

A Convolutional Neural Network will be developed using PyTorch.

The CNN learns enhancement patterns using paired low-light and normal-light images from the LOL dataset.

**Proposed CNN workflow:**

1. Input a low-light image.
2. Extract image features using convolution layers.
3. Apply activation functions.
4. Learn brightness and contrast enhancement patterns.
5. Reconstruct the enhanced image.
6. Compare the output against the normal-light ground-truth image.

Initial training configuration:

| Parameter | Initial Value |
|---|---|
| Framework | PyTorch |
| Input Size | 256 × 256 |
| Batch Size | 8–16 |
| Epochs | 30–50 |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | L1 + SSIM Loss |

These values are initial experimental settings and may be adjusted during hyperparameter tuning.

### 3.3 Proposed Hybrid Model

The proposed approach combines Genetic Algorithm optimisation, CLAHE and CNN refinement.

**Processing Pipeline:**

```text
             LOW-LIGHT IMAGE
                    |
                    v
             IMAGE PREPROCESSING
                    |
                    v
             GENETIC ALGORITHM
                    |
                    v
        OPTIMISE CLAHE PARAMETERS
                    |
                    v
             OPTIMISED CLAHE
                    |
                    v
               CNN REFINEMENT
                    |
                    v
           FINAL ENHANCED IMAGE
                    |
                    v
             QUALITY EVALUATION
```

The Genetic Algorithm searches for suitable CLAHE parameters, including:

- Clip Limit
- Tile Grid Size

The optimised CLAHE output is passed to the trained CNN for further refinement.

The final output is evaluated using image quality metrics.

**Note:** The GA fitness function will consider image quality, contrast improvement and detail preservation while penalising excessive noise. For paired images, reference-based metrics can be used. For images without ground truth, appropriate no-reference metrics will be considered.

---

## 🧬 4. Genetic Algorithm Optimisation

A Genetic Algorithm is an evolutionary optimisation technique inspired by natural selection.

It searches for better parameter combinations through generations.

### GA Workflow

1. **Population Initialisation:** Generate random CLAHE parameter combinations.
2. **Fitness Evaluation:** Evaluate the quality of the enhanced images.
3. **Selection:** Select better-performing parameter combinations.
4. **Crossover:** Combine selected parameter combinations.
5. **Mutation:** Randomly modify some parameters to maintain diversity.
6. **Replacement:** Form a new population.
7. **Termination:** Stop when the maximum number of generations is reached or a stopping condition is satisfied.

### Initial GA Parameters

| Parameter | Initial Range |
|---|---|
| CLAHE Clip Limit | 1–8 |
| Tile Grid Size | 4–16 |
| Population Size | 20–50 |
| Generations | 20–100 |
| Crossover Rate | 0.7–0.9 |
| Mutation Rate | 0.01–0.1 |

The final parameter combination will be selected based on experimental results.

---

## 📂 5. Dataset

Two publicly available datasets will be considered.

### 5.1 LOL Dataset (Primary Dataset)

**LOL – Low-Light Image Enhancement Dataset**

- Contains paired low-light and normal-light images.
- Suitable for supervised CNN training.
- Supports reference-based image quality evaluation.
- Used for training, validation and testing.

Dataset Link:  
https://daooshee.github.io/BMVC2018website/

### 5.2 BSD500 Dataset (Additional Dataset)

**BSD500 – Berkeley Segmentation Dataset and Benchmark**

- Contains natural images.
- Can be used for additional testing and generalisation experiments.
- Can be used to evaluate enhancement on artificially darkened images.

Dataset Link:  
https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html

**Dataset Strategy:**

LOL will be the primary dataset because it provides paired low-light and normal-light images.

BSD500 will be considered for additional testing. If artificial low-light images are generated from BSD500, the darkening procedure will be documented.

The dataset will be divided into training, validation and testing subsets where applicable. Test images will not be used for model training or hyperparameter optimisation.

---

## 🧪 6. Mandatory Comparative Experiments

All six experiments specified in the project guidelines will be implemented.

| Experiment | Implementation | Purpose |
|---|---|---|
| Experiment 1 | Traditional Image Processing | Establish baseline performance |
| Experiment 2 | CNN-Based Enhancement | Evaluate learning-based enhancement |
| Experiment 3 | Proposed Hybrid Model | Demonstrate GA + CLAHE + CNN |
| Experiment 4 | Hyperparameter Tuning | Identify suitable parameter combinations |
| Experiment 5 | Ablation Study | Evaluate individual component contributions |
| Experiment 6 | Performance Comparison | Compare image quality and computational performance |

### Experiment 1: Traditional Image Processing

Implement:

- Histogram Equalization
- CLAHE

Record the enhanced images and their evaluation metrics.

### Experiment 2: CNN-Based Enhancement

- Develop a CNN architecture.
- Train using the LOL dataset.
- Evaluate using validation and testing images.
- Record training loss and image quality metrics.

### Experiment 3: Proposed Hybrid Enhancement

Implement:

- Genetic Algorithm optimisation.
- CLAHE with optimised parameters.
- CNN-based refinement.

Compare the proposed output against the traditional and CNN-based approaches.

### Experiment 4: Hyperparameter Tuning

Experiment with different combinations of:

- CLAHE clip limit.
- Tile grid size.
- GA population size.
- Number of generations.
- Crossover rate.
- Mutation rate.
- CNN learning rate.

Record the results of every experiment.

### Experiment 5: Ablation Study

Evaluate the contribution of each component.

| Configuration | CLAHE | GA | CNN |
|---|---|---|---|
| A | Yes | No | No |
| B | No | No | Yes |
| C | Yes | Yes | No |
| D – Full Proposed Model | Yes | Yes | Yes |

Compare the outputs to determine how each component contributes to the final enhancement quality.

### Experiment 6: Performance Comparison

Compare all three major approaches based on:

- Image quality.
- Computational time.
- Model size.
- Noise reduction.
- Detail preservation.

---

## 📊 7. Evaluation Metrics

The following metrics will be considered for evaluating image enhancement performance.

| Metric | Description | Preferred Result |
|---|---|---|
| PSNR | Measures reconstruction quality against ground truth | Higher |
| SSIM | Measures structural similarity | Higher |
| Entropy | Measures image information content | Context-dependent |
| NIQE | Measures perceptual quality without reference images | Lower |
| Contrast Improvement Index | Measures contrast enhancement | Context-dependent |
| Edge Preservation | Measures preservation of image details and edges | Higher |
| Execution Time | Time required to enhance an image | Lower |
| Model Size | Storage required for the trained model | Lower where practical |

**Important:** PSNR and SSIM require reference images. NIQE is a no-reference metric. Entropy and contrast improvement should not be maximised blindly because excessive contrast and noise can increase these measurements without improving visual quality.

### Comparative Results Table

The final report will include the following format:

| Evaluation Metric | Traditional Method | ML/DL Method | Proposed Method |
|---|---|---|---|
| PSNR | Actual Result | Actual Result | Actual Result |
| SSIM | Actual Result | Actual Result | Actual Result |
| Entropy | Actual Result | Actual Result | Actual Result |
| NIQE | Actual Result | Actual Result | Actual Result |
| Contrast Improvement | Actual Result | Actual Result | Actual Result |
| Edge Preservation | Actual Result | Actual Result | Actual Result |
| Execution Time (ms/image) | Actual Result | Actual Result | Actual Result |
| Model Size (MB) | Actual Result | Actual Result | Actual Result |

All values will be filled using actual experimental measurements.

---

## 👥 8. Team Members and Responsibilities

The project is divided among three members to support parallel development.

### Member 1 – Traditional Image Processing and Evaluation

**Responsibilities:**

- Download and organise the dataset.
- Implement Histogram Equalization.
- Implement CLAHE.
- Implement PSNR and SSIM.
- Implement entropy and NIQE evaluation.
- Implement additional image quality metrics.
- Generate traditional enhancement outputs.
- Generate visual comparisons.

**Assigned Files:**

```text
traditional/
    __init__.py
    traditional.py
    metrics.py

dataset.py
visualization.py
```

### Member 2 – CNN and Deep Learning

**Responsibilities:**

- Design the CNN architecture.
- Prepare the LOL dataset for training.
- Implement the training pipeline.
- Implement loss functions.
- Train and validate the CNN.
- Save trained model weights.
- Implement CNN inference.
- Evaluate CNN enhancement results.

**Assigned Files:**

```text
cnn/
    __init__.py
    model.py
    train.py
    inference.py
    losses.py
```

### Member 3 – Genetic Algorithm and Integration

**Responsibilities:**

- Implement Genetic Algorithm optimisation.
- Design the fitness function.
- Implement selection, crossover and mutation.
- Optimise CLAHE parameters.
- Develop the hybrid enhancement pipeline.
- Integrate traditional and CNN modules.
- Develop the Streamlit application.
- Implement comparative experiments.
- Generate final comparison graphs and tables.

**Assigned Files:**

```text
optimization/
    __init__.py
    genetic_algorithm.py
    hybrid.py

app.py

evaluation/
    experiments.py
    comparison.py
```

### Parallel Development Strategy

Each team member will work on separate modules and separate GitHub branches.

The agreed function interfaces will allow modules to be integrated without modifying each other's internal implementations.

---

## 🔗 9. Common Function Interfaces

All team members must follow common input and output conventions.

**Image Format:**

- NumPy arrays.
- RGB colour format at module boundaries.
- Image dimensions: H × W × 3.
- Pixel range: 0–255.

OpenCV-specific BGR conversions should be handled internally.

### Member 1 Interfaces

```python
def apply_histogram_equalization(image):
    pass

def apply_clahe(image, clip_limit=2.0, grid_size=8):
    pass

def calculate_psnr(enhanced, ground_truth):
    pass

def calculate_ssim(enhanced, ground_truth):
    pass

def calculate_entropy(image):
    pass

def calculate_niqe(image):
    pass
```

### Member 2 Interfaces

```python
def build_model():
    pass

def enhance_with_cnn(image, model):
    pass
```

### Member 3 Interfaces

```python
def optimize_clahe(image, ground_truth=None):
    pass

def enhance_hybrid(image, model):
    pass
```

These are agreed function signatures. Actual implementations will be developed independently.

The function contracts, return types and image formats must remain consistent throughout development.

---

## 🏗️ 10. Project Folder Structure

The GitHub repository will follow this structure:

```text
Image-Enhancement-Project/
│
├── app.py
│
├── traditional/
│   ├── __init__.py
│   ├── traditional.py
│   └── metrics.py
│
├── cnn/
│   ├── __init__.py
│   ├── model.py
│   ├── train.py
│   ├── inference.py
│   └── losses.py
│
├── optimization/
│   ├── __init__.py
│   ├── genetic_algorithm.py
│   └── hybrid.py
│
├── evaluation/
│   ├── experiments.py
│   ├── comparison.py
│   └── visualization.py
│
├── dataset/
│   └── README.md
│
├── models/
│   └── README.md
│
├── results/
│   ├── metrics.csv
│   └── figures/
│
├── requirements.txt
├── .gitignore
└── README.md
```

### Folder Description

| Folder/File | Purpose |
|---|---|
| traditional/ | Traditional image enhancement methods |
| cnn/ | CNN architecture, training and inference |
| optimization/ | Genetic Algorithm and hybrid enhancement |
| evaluation/ | Experiments, comparisons and visualisation |
| dataset/ | Dataset instructions and organisation |
| models/ | Trained model storage instructions |
| results/ | Experimental results and graphs |
| app.py | Streamlit application |
| requirements.txt | Python dependencies |
| README.md | Project documentation |

Large datasets and model weights will be stored separately from GitHub source code, using an appropriate shared storage location.

---

## 🌿 11. GitHub Collaboration Strategy

The project will use Git and GitHub for parallel development.

### Branch Structure

```text
main
│
├── member-1-traditional
│
├── member-2-cnn
│
└── member-3-ga-app
```

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL

cd Image-Enhancement-Project
```

### Step 2: Create Individual Branches

**Member 1:**

```bash
git switch -c member-1-traditional
```

**Member 2:**

```bash
git switch -c member-2-cnn
```

**Member 3:**

```bash
git switch -c member-3-ga-app
```

### Step 3: Commit Changes

```bash
git add .

git commit -m "Implemented assigned module"

git push -u origin YOUR_BRANCH_NAME
```

### Step 4: Create Pull Requests

1. Open the GitHub repository.
2. Select the individual branch.
3. Click New Pull Request.
4. Select `main` as the destination.
5. Review the changed files.
6. Merge after testing.

### Rules to Minimise Merge Conflicts

- Each member should modify only their assigned files.
- Avoid modifying shared files simultaneously.
- Do not commit datasets or large model weights.
- Follow the agreed function interfaces.
- Use Pull Requests for merging.
- Pull the latest main branch before starting new work.
- Coordinate changes to requirements.txt and README.md.

This workflow is designed to minimise merge conflicts and enable independent development.

---

## 💻 12. Technology Stack

| Category | Technology |
|---|---|
| Programming Language | Python |
| Image Processing | OpenCV |
| Numerical Computing | NumPy |
| Deep Learning | PyTorch |
| Genetic Algorithm | DEAP / NumPy |
| Evaluation Metrics | scikit-image / NIQE implementation |
| Frontend | Streamlit |
| Data Handling | Pandas |
| Visualisation | Matplotlib |
| Version Control | Git and GitHub |
| Dataset | LOL and BSD500 |

---

## ⚙️ 13. Installation and Setup

### Prerequisites

- Python 3.10 or compatible version.
- Git.
- Visual Studio Code or another Python IDE.
- Internet connection for downloading datasets and dependencies.
- NVIDIA GPU is optional but recommended for faster CNN training.

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL

cd Image-Enhancement-Project
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 3: Activate the Environment

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

**Linux / Ubuntu:**

```bash
source .venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install numpy opencv-python torch torchvision

pip install scikit-image matplotlib pandas streamlit pillow

pip install deap
```

### Step 5: Run the Application

```bash
streamlit run app.py
```

The application will open in your default web browser.

**Note:** Install the appropriate PyTorch build for your hardware if GPU acceleration is required. The exact dependency versions will be finalised in requirements.txt after testing.

---

## 🖥️ 14. Proposed Application Features

A Streamlit-based application will be developed to demonstrate the project.

### Feature 1: Image Upload

- Upload low-light images.
- Support JPG, JPEG and PNG formats.
- Display the original image.

### Feature 2: Enhancement Methods

Allow users to select:

- Histogram Equalization.
- Traditional CLAHE.
- CNN Enhancement.
- GA-Optimised CLAHE.
- Proposed Hybrid Enhancement.

### Feature 3: Image Comparison

Display:

- Original image.
- Traditional enhancement output.
- CNN enhancement output.
- Proposed hybrid output.

### Feature 4: Genetic Algorithm Visualisation

Display:

- Selected clip limit.
- Selected tile grid size.
- Fitness across generations.
- Best parameter combination.

### Feature 5: Evaluation Dashboard

Display:

- PSNR.
- SSIM.
- Entropy.
- NIQE.
- Contrast improvement.
- Edge preservation.
- Execution time.

### Feature 6: Download Results

Allow users to download the enhanced image.

### Feature 7: Performance Comparison

Display comparison graphs and tables for all three approaches.

---

## 📅 15. Project Implementation Timeline

| Date | Task |
|---|---|
| 9–10 November 2026 | Repository setup, dataset download and environment configuration |
| 11 November 2026 | Abstract submission |
| 12–15 November 2026 | Independent development of all three modules |
| 16–18 November 2026 | Integration and application development |
| 19–21 November 2026 | Hyperparameter tuning, ablation and performance experiments |
| 22–23 November 2026 | Report preparation and presentation |
| 24 November 2026 | Final project evaluation |

---

## 📋 16. Development Checklist

### Initial Setup

- [ ] Create GitHub repository.
- [ ] Add all three team members.
- [ ] Create project folder structure.
- [ ] Create individual Git branches.
- [ ] Set up Python virtual environment.
- [ ] Download LOL dataset.
- [ ] Finalise common function interfaces.

### Member 1 – Traditional Image Processing

- [ ] Implement Histogram Equalization.
- [ ] Implement CLAHE.
- [ ] Implement PSNR.
- [ ] Implement SSIM.
- [ ] Implement entropy.
- [ ] Implement NIQE.
- [ ] Generate traditional enhancement outputs.
- [ ] Test metric calculations.

### Member 2 – CNN

- [ ] Prepare LOL dataset.
- [ ] Implement CNN architecture.
- [ ] Implement training pipeline.
- [ ] Implement loss functions.
- [ ] Train CNN.
- [ ] Validate CNN.
- [ ] Save model weights.
- [ ] Implement inference.
- [ ] Generate CNN enhancement outputs.

### Member 3 – Genetic Algorithm and Integration

- [ ] Implement GA population initialisation.
- [ ] Implement fitness function.
- [ ] Implement selection.
- [ ] Implement crossover.
- [ ] Implement mutation.
- [ ] Optimise CLAHE parameters.
- [ ] Implement hybrid enhancement.
- [ ] Develop Streamlit application.
- [ ] Integrate all modules.
- [ ] Generate comparison graphs.

### Final Experiments

- [ ] Complete Experiment 1 – Traditional Image Processing.
- [ ] Complete Experiment 2 – CNN Enhancement.
- [ ] Complete Experiment 3 – Proposed Hybrid Model.
- [ ] Complete Experiment 4 – Hyperparameter Tuning.
- [ ] Complete Experiment 5 – Ablation Study.
- [ ] Complete Experiment 6 – Performance Comparison.
- [ ] Record actual evaluation metrics.
- [ ] Prepare visual comparisons.
- [ ] Prepare final report.
- [ ] Prepare presentation.
- [ ] Test the complete application.

---

## 📈 17. Expected Outcomes

The expected outcomes of the project are:

1. A working traditional image enhancement system using Histogram Equalization and CLAHE.
2. A trained CNN model capable of enhancing low-light images.
3. A Genetic Algorithm capable of automatically selecting CLAHE parameters.
4. A hybrid image enhancement system combining GA-optimised CLAHE and CNN refinement.
5. A comparative analysis of traditional, deep learning and proposed methods.
6. A hyperparameter tuning report.
7. An ablation study demonstrating the contribution of each component.
8. A Streamlit application for interactive image enhancement.
9. Quantitative and visual evaluation of the enhancement results.

The proposed model is expected to improve image enhancement performance, but its actual effectiveness will be established through experiments rather than assumed in advance.

---

## 📦 18. Final Submission Requirements

The following deliverables will be prepared:

- Project abstract (maximum 5–7 slides).
- Complete source code.
- Dataset and dataset documentation.
- Traditional image enhancement implementation.
- CNN architecture and trained model.
- Genetic Algorithm implementation.
- Proposed hybrid enhancement implementation.
- Results of all six mandatory experiments.
- Comparative results table.
- Hyperparameter tuning results.
- Ablation study results.
- Visual comparisons and error maps where applicable.
- Complete project report.
- Final presentation.
- Working project demonstration.

---

## 👨‍💻 19. Team Information

| Role | Team Member | Responsibilities |
|---|---|---|
| Member 1 | Name to be added | Traditional Image Processing and Evaluation |
| Member 2 | Name to be added | CNN and Deep Learning |
| Member 3 | Name to be added | Genetic Algorithm and Integration |

**Subject:** Fundamentals of Image Processing

**Project Title:** Contrast Enhancement Using CLAHE, CNNs and Genetic Algorithm Optimisation

---

## 📚 20. References

1. LOL Dataset – Low-Light Image Enhancement:  
   https://daooshee.github.io/BMVC2018website/

2. BSD500 Dataset – Berkeley Segmentation Dataset:  
   https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html

3. OpenCV Documentation:  
   https://docs.opencv.org/

4. PyTorch Documentation:  
   https://pytorch.org/docs/stable/index.html

5. Scikit-image Documentation:  
   https://scikit-image.org/docs/stable/

6. Streamlit Documentation:  
   https://docs.streamlit.io/

---

## ⭐ Project Summary

This project aims to develop an intelligent image enhancement system that combines traditional image processing, deep learning and evolutionary optimisation.

By integrating CLAHE, CNNs and Genetic Algorithms, the proposed system attempts to achieve improved contrast, better visibility and preservation of image details.

The project will be evaluated through six mandatory comparative experiments to understand the performance, effectiveness and contribution of each implemented technique.

**Built with Python, OpenCV, PyTorch, Genetic Algorithms and Streamlit.**

---
**Fundamentals of Image Processing | 2026**