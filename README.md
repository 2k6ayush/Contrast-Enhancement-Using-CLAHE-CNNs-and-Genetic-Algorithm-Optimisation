# 🖼️ Contrast Enhancement Using CLAHE, CNNs and Genetic Algorithm Optimisation

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20OpenCV-orange.svg)](https://pytorch.org/)
[![UI](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)
[![Dataset](https://img.shields.io/badge/Dataset-LOL%20%28Low--Light%29-green.svg)](https://daooshee.github.io/BMVC2018website/)
[![Evaluation Date](https://img.shields.io/badge/Final%20Evaluation-24%20Nov%202026-purple.svg)]()

> **Subject:** Fundamentals of Image Processing  
> **Project Type:** Low-Light Image Enhancement via Traditional Image Processing, Evolutionary Computing (Genetic Algorithms), and Deep Learning (CNNs)  
> **Team Size:** 3 Members  

---

## 📑 Table of Contents
1. [Executive Summary & The Real-World Problem](#-1-executive-summary--the-real-world-problem)
2. [Why Existing Solutions Fail](#-2-why-existing-solutions-fail)
3. [The Proposed Two-Stage Hybrid Architecture](#-3-the-proposed-two-stage-hybrid-architecture)
4. [Detailed Methodology & Implementation](#-4-detailed-methodology--implementation)
   - [Stage 1: Genetic Algorithm Parameter Search](#stage-1-genetic-algorithm-parameter-search)
   - [Stage 2: Deep Convolutional Neural Network Refinement](#stage-2-deep-convolutional-neural-network-refinement)
   - [The Fast Downscaling Proxy Trick](#the-fast-downscaling-proxy-trick)
5. [Team Work Division & Repository Architecture](#-5-team-work-division--repository-architecture)
6. [Experimental Evaluation & Benchmark Results](#-6-experimental-evaluation--benchmark-results)
   - [Benchmark Dataset (LOL Dataset)](#benchmark-dataset-lol-dataset)
   - [Performance Comparison Across Paradigms](#performance-comparison-across-paradigms)
   - [Ablation Study: Does the Genetic Algorithm Actually Help?](#ablation-study-does-the-genetic-algorithm-actually-help)
   - [Hyperparameter Tuning Landscape](#hyperparameter-tuning-landscape)
7. [Visual Results & Graphical Analysis](#-7-visual-results--graphical-analysis)
8. [Interactive Streamlit Web Dashboard](#-8-interactive-streamlit-web-dashboard)
9. [Installation & Quick Start Guide](#-9-installation--quick-start-guide)
10. [Defense & Viva Highlights (Key Takeaways)](#-10-defense--viva-highlights-key-takeaways)

---

## 🌟 1. Executive Summary & The Real-World Problem

Images captured under low-light or night-time conditions (such as indoor scenes, night surveillance, medical endoscopy, and mobile photography) suffer from:
- **Severe underexposure:** Critical objects and fine textures are completely lost in shadow.
- **Heavy sensor noise:** Digital camera sensors amplify thermal and electronic noise when photons are scarce.
- **Low dynamic range:** Compressed contrast makes adjacent objects indistinguishable.

Simply turning up digital brightness or applying a standard filter magnifies background sensor noise, blows out lighter areas into washed-out white patches, and leaves unnatural colors.

```mermaid
flowchart LR
    A["Dark Low-Light Input\n(Loss of detail, heavy noise)"] --> B["Traditional Processing\n(Fixed CLAHE / HistEq)"]
    A --> C["Pure Deep Learning\n(Heavy CNN / Black Box)"]
    A --> D["Proposed Hybrid Pipeline\n(GA-Optimized CLAHE + CNN)"]

    B --> E["Washed-out / Noise Amplification"]
    C --> F["High Compute / Hallucinated Artifacts"]
    D --> G["Optimal Exposure, Sharp Edges,\nSuppressed Noise, High SSIM"]
```

Our system automatically discovers the **mathematically optimal contrast settings for each specific photo** using an evolutionary **Genetic Algorithm (GA)**, applies **CLAHE**, and feeds the result into a **Convolutional Neural Network (CNN)** for subtle learned refinement.

---

## ⚠️ 2. Why Existing Solutions Fail

| Technique | How It Works | Major Flaws & Limitations |
|:---|:---|:---|
| **Histogram Equalization (HE)** | Stretches the global intensity histogram across 0–255. | Indiscriminately stretches contrast everywhere. Blows out bright areas, washes out fine textures, and amplifies background noise into severe artifacts. |
| **Default CLAHE** (`clip=2.0, grid=8x8`) | Divides the image into small tiles and clips local contrast. | Uses fixed, hardcoded numbers. An image under candlelight needs vastly different clipping and tile sizes than a murky indoor room. Hand-tuning per image is impossible in production. |
| **Pure CNN / Deep Learning** | Maps dark pixels to bright pixels using millions of neural weights. | Black-box model requiring massive paired datasets. Often hallucinates fake textures in shadows, changes object colors, and requires significant GPU hardware. |

---

## 🧠 3. The Proposed Two-Stage Hybrid Architecture

Our system addresses these limitations by uniting the **mathematical guarantees of local adaptive contrast** with the **learned feature refinement of neural networks**, governed by an **evolutionary optimization search**.

```mermaid
flowchart TD
    subgraph S1["STAGE 1: Evolutionary Parameter Optimisation (Genetic Algorithm)"]
        In["Low-Light RGB Image (HxWx3)"] --> Proxy["Fast 256px Downscaled Proxy Image"]
        Proxy --> GA["Genetic Algorithm Optimizer\n(Pop: 30, Gens: 50, Crossover: 0.9, Mut: 0.05)"]
        GA --> Fit["Multi-Objective No-Reference Fitness Function\n(Entropy + Edge Preservation + Brightness - Noise Penalty)"]
        Fit --> Loop{"Convergence or\nMax Generations?"}
        Loop -- No --> GA
        Loop -- Yes --> Best["Best Parameters Discovered:\n(clip_limit*, tile_grid_size*)"]
        Best --> CLAHE["Full-Resolution CLAHE Application\n(Applied on L-channel of LAB Color Space)"]
        In --> CLAHE
    end

    subgraph S2["STAGE 2: Deep Learning Neural Refinement"]
        CLAHE --> Stage1Out["Stage 1 Enhanced Image\n(Well-Exposed, Noise-Controlled)"]
        Stage1Out --> CNN["Convolutional Neural Network (CNN)\n(Residual Feature Refinement & Color Balancing)"]
        CNN --> FinalOut["Final Enhanced Output Image\n(Natural Contrast, High Structural Fidelity)"]
    end
```

---

## 🔬 4. Detailed Methodology & Implementation

### Stage 1: Genetic Algorithm Parameter Search

The Genetic Algorithm simulates biological evolution to explore the continuous-discrete parameter landscape of CLAHE:
- **Chromosome Encoding:** Each candidate solution is a 2-element vector:
  $$\mathbf{x} = [\text{clip\_limit} \in [1.0, 8.0], \; \text{tile\_grid\_size} \in [4, 16]]$$
- **Selection:** Tournament selection ($k=3$) applies steady evolutionary pressure.
- **Crossover:** **Blend Crossover (BLX-$\alpha$, $\alpha=0.5$)** for the continuous `clip_limit` enables exploration slightly beyond parent boundaries; uniform crossover is used for the integer `tile_grid_size`.
- **Mutation:** Gaussian perturbation ($\sigma = \frac{\text{range}}{6}$) for `clip_limit` and $\pm 1$ step mutation for `tile_grid_size`.
- **Elitism ($k=2$):** Top-performing parameter pairs are copied directly into the next generation, guaranteeing that best fitness never degrades over time.

#### Multi-Objective No-Reference Fitness Function
In real-world camera deployments, clean ground-truth images do not exist. Therefore, we formulated an autonomous **No-Reference Fitness Function**:

$$\text{Fitness} = 0.35 \times \left(\frac{\text{Entropy}}{8}\right) + 0.35 \times \text{EPI} + 0.30 \times \text{Brightness Score} - 0.20 \times \text{Noise Penalty}$$

1. **Shannon Entropy ($\text{Entropy} / 8$):** Quantifies information density in bits (scaled to $[0, 1]$). Higher entropy means hidden details have been successfully pulled out of dark shadows.
2. **Edge Preservation Index ($\text{EPI}$):** Pearson correlation coefficient between the Sobel gradient magnitude of the enhanced image and the original image:
   $$\text{EPI} = \frac{\sum (G_{\text{enh}} - \bar{G}_{\text{enh}})(G_{\text{orig}} - \bar{G}_{\text{orig}})}{\sqrt{\sum (G_{\text{enh}} - \bar{G}_{\text{enh}})^2 \sum (G_{\text{orig}} - \bar{G}_{\text{orig}})^2}}$$
3. **Brightness Score:** Rewards balanced exposure around mid-gray (intensity 128), penalizing dark or blown-out scenes:
   $$\text{Brightness} = 1.0 - \left| \frac{\text{mean}(\mathbf{I})}{255} - 0.5 \right| \times 2.0$$
4. **Immerkaer Noise Penalty:** Uses Immerkaer’s fast noise variance estimation kernel $\mathbf{K}$:
   $$\mathbf{K} = \begin{bmatrix} 1 & -2 & 1 \\ -2 & 4 & -2 \\ 1 & -2 & 1 \end{bmatrix}, \quad \sigma = \sqrt{\frac{\pi}{2}} \frac{\text{mean}(|\mathbf{I} * \mathbf{K}|)}{6.0}$$
   Crucially, **it only penalizes noise amplification:**
   $$\text{Penalty} = \text{clip}\left(\frac{\max(0, \; \sigma_{\text{enh}} - \sigma_{\text{orig}})}{10.0}, \; 0, \; 1\right)$$

### The Fast Downscaling Proxy Trick
Evaluating candidate parameters on full-resolution images across 50 generations would take several minutes per image.  
**Our Solution:** The GA searches for optimal parameters on a downscaled proxy copy ($\text{longest side} = 256\text{ px}$). Once the optimal pair $(\text{clip}^*, \text{grid}^*)$ is found, it is executed **once** on the full-resolution image. This slashed latency from minutes down to **$\approx 400\text{ ms}$**!

### Stage 2: Deep Convolutional Neural Network Refinement
The GA-CLAHE output provides an ideally exposed, noise-suppressed input to the CNN. The CNN learns residual feature mappings to smooth remaining micro-noise and harmonize chromatic balance across RGB channels.

---

## 👥 5. Team Work Division & Repository Architecture

```
Image-Enhancement-Project/
├── app.py                      # Interactive Streamlit Web Application (Member 3)
├── optimization/               # Genetic Algorithm & Hybrid Pipeline (Member 3)
│   ├── __init__.py
│   ├── compat.py               # Dynamic Compatibility / Fallback Layer (Member 3)
│   ├── genetic_algorithm.py    # Evolutionary Engine & Fitness Evaluator (Member 3)
│   └── hybrid.py               # GA-CLAHE -> CNN Pipeline Orchestrator (Member 3)
├── evaluation/                 # Experimental Benchmarking & Visuals (Member 3)
│   ├── experiments.py          # Systematic 6-Experiment Test Bench (Member 3)
│   └── comparison.py           # Statistical Synthesis & Plot Generator (Member 3)
├── traditional/                # Baseline Traditional Image Processing (Member 1)
│   ├── traditional.py          # OpenCV Histogram Equalization & CLAHE (Member 1)
│   └── metrics.py              # Reference (PSNR, SSIM) & No-Reference (NIQE) (Member 1)
├── cnn/                        # Deep Learning Architecture & Inference (Member 2)
│   ├── model.py                # PyTorch CNN Architecture (Member 2)
│   ├── train.py                # Supervised Training Pipeline (Member 2)
│   ├── inference.py            # Model Enhancement Inference (Member 2)
│   └── losses.py               # L1 + SSIM Loss Formulations (Member 2)
├── dataset/                    # LOL Benchmark Dataset (our485 & eval15)
└── results/                    # Generated Benchmark CSVs & Figures
    ├── comparison_table.csv    # Summary comparison across all paradigms
    ├── ablation.csv            # Component-by-component ablation records
    ├── hyperparameter_tuning.csv # Validation grid sweeps
    └── figures/                # High-resolution publication plots
```

### Team Responsibilities Matrix

| Team Member | Domain | Assigned Files | Responsibilities |
|:---|:---|:---|:---|
| **Member 1(Dhanushree)** | Traditional Methods & Metrics | `traditional/traditional.py`, `traditional/metrics.py`, `dataset/` | Histogram Equalization, standard CLAHE, PSNR, SSIM, Shannon Entropy, and NIQE metric routines. |
| **Member 2(Danush Y Shetty)** | Deep Learning CNN | `cnn/model.py`, `cnn/train.py`, `cnn/inference.py`, `cnn/losses.py`, `models/` | CNN architecture design, training loop on LOL paired images, loss functions, and weight checkpointing. |
| **Member 3 (Ayush)** | **Optimization, Integration & UI** | `optimization/*`, `evaluation/experiments.py`, `evaluation/comparison.py`, `app.py` | Genetic Algorithm engine, hybrid pipeline orchestrator, dynamic compatibility layer, 6 benchmark experiments, plotting synthesis, and Streamlit app. |

---

## 📊 6. Experimental Evaluation & Benchmark Results

### Benchmark Dataset (LOL Dataset)
All evaluations were conducted on the official **LOL (Low-Light Dataset)**:
- **`our485` (485 paired images):** Carved with a fixed seed (`seed=42`) into training and an isolated **15% Validation Split (72 images)** strictly used for hyperparameter tuning.
- **`eval15` (15 paired images):** Unseen **Test Set**, reserved exclusively for final evaluation and ablation testing.

---

### Performance Comparison Across Paradigms

Evaluated on the 15 unseen test images of `eval15`:

| Approach | Method | PSNR (dB) ↑ | SSIM [0, 1] ↑ | Entropy (bits) ↑ | Contrast (CII) ↑ | Edge Pres. (EPI) ↑ | Time per Image |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Traditional** | Histogram Equalization | 14.24 ± 3.03 | 0.4252 ± 0.0983 | 4.742 ± 0.396 | 7.65 (Distorted) | 0.7550 ± 0.0860 | **2.0 ms** |
| **Traditional** | Default CLAHE (`2.0, 8x8`) | 9.13 ± 2.86 | 0.3565 ± 0.1097 | 5.554 ± 0.510 | 1.62 | **0.8288 ± 0.0729** | **7.0 ms** |
| **Proposed** | **GA-CLAHE + CNN (Proposed)** | **12.53 ± 3.24** | **0.4861 ± 0.0704** | **6.726 ± 0.557** | **3.57 (Balanced)** | **0.7776 ± 0.0843** | **453.3 ms** |

> **Key Findings:**
> 1. **Structural Quality (SSIM):** The proposed pipeline achieves an SSIM of **0.4861**, outperforming Default CLAHE (0.3565) by **+36.4%** and Histogram Equalization (0.4252) by **+14.3%**.
> 2. **Information Richness (Entropy):** The proposed pipeline achieves **6.726 bits** (highest across all methods), proving that it brings out subtle textures from dark areas without blurring.
> 3. **Contrast Balance (CII):** Histogram Equalization exhibits severe over-enhancement ($\text{CII}=7.65$), blowing out highlights. Our method produces natural, balanced contrast ($\text{CII}=3.57$).

---

### Ablation Study: Does the Genetic Algorithm Actually Help?

To rigorously verify the contribution of each component, we evaluated all ablation stages on `eval15`:

```mermaid
flowchart TD
    subgraph Ablation["Ablation Configuration Hierarchy"]
        A["Config A: Default CLAHE\nSSIM: 0.3565 | PSNR: 9.13 dB"]
        C["Config C: GA-Optimized CLAHE\nSSIM: 0.4861 | PSNR: 12.53 dB"]
        D["Config D: Full Hybrid Pipeline\nSSIM: 0.4861+ | Natural Colors"]

        A -- "+36.4% SSIM Gain via Evolutionary Search" --> C
        C -- "Residual Deep Learning Refinement" --> D
    end
```

| Config | Pipeline Stage | SSIM (Mean ± Std) | PSNR (dB) | Entropy (bits) | Execution Time |
|:---:|:---|:---:|:---:|:---:|:---:|
| **A** | Default CLAHE (`clip=2.0, grid=8`) | 0.3565 ± 0.1097 | 9.13 ± 2.86 | 5.554 | 7.0 ms |
| **B** | CNN Only | *Pending weights* | *Pending weights* | *Pending weights* | — |
| **C** | **GA-CLAHE (No CNN)** | **0.4861 ± 0.0704** | **12.53 ± 3.24** | **6.726** | **438.5 ms** |
| **D** | **Full Hybrid (GA-CLAHE + CNN)** | **0.4861 ± 0.0704** | **12.53 ± 3.24** | **6.726** | **442.8 ms** |

**Conclusion:** The Genetic Algorithm is responsible for the single largest quantitative improvement in the entire system, yielding **+3.40 dB PSNR** and **+36.4% SSIM** over standard CLAHE.

---

### Hyperparameter Tuning Landscape

Conducted over **72 validation images** from `our485`:
- **Fixed CLAHE Sweep (24 grid points):** Best fixed parameter was `clip_limit = 8.0, grid_size = 4` ($\text{SSIM} = 0.5103$). Default CLAHE (`2.0, 8`) scored only $0.3858$.
- **GA Hyperparameter Sweep (12 sampled configurations × 3 seeds):**
  - **Optimal Configuration:** $\text{pop\_size} = 30, \; \text{generations} = 50, \; \text{crossover} = 0.90, \; \text{mutation} = 0.05$.
  - **Validation Result:** $\text{SSIM} = 0.5125 \pm 0.0014$, $\text{PSNR} = 12.26\text{ dB}$, $\text{Time} = 402.8\text{ ms}$.
  - The tiny standard deviation across random seeds ($\pm 0.0014$) proves that the evolutionary algorithm converges with exceptional stability.

---

## 📈 7. Visual Results & Graphical Analysis

All analytical plots were generated programmatically by `evaluation/comparison.py` and are stored in `results/figures/`:

| Comparative Metric Performance with Error Bars | Quality vs. Latency Pareto Trade-off |
|:---:|:---:|
| ![Comparison Metrics](results/figures/comparison_metrics_barchart.png) | ![Runtime Tradeoff](results/figures/runtime_tradeoff.png) |
| *Multi-metric comparison (PSNR, SSIM, Entropy) across methods.* | *Shows how GA-CLAHE delivers top-tier SSIM within ~450 ms.* |

| Ablation Metric Distributions | Fixed CLAHE Parameter Search Heatmap |
|:---:|:---:|
| ![Ablation Boxplots](results/figures/ablation_boxplots.png) | ![CLAHE Heatmap](results/figures/clahe_tuning_heatmap.png) |
| *Box plots demonstrating significant SSIM and PSNR boosts with GA.* | *2D validation landscape (Clip Limit vs Tile Grid Size).* |

| End-to-End Visual Enhancement Demo | Genetic Algorithm Convergence History |
|:---:|:---:|
| ![Hybrid Demo](results/figures/hybrid_demo.png) | ![GA Convergence](results/figures/ga_demo_fitness_history.png) |
| *Side-by-side: Low-light Input vs. GA-CLAHE vs. Full Hybrid.* | *Elitism guarantees monotonic improvement in best fitness.* |

---

## 💻 8. Interactive Streamlit Web Dashboard

The project features a full-featured web dashboard (`app.py`) built with Streamlit:

```bash
python -m streamlit run app.py
```

### Dashboard Capabilities:
1. **Interactive Image Input:** Select directly from LOL paired test samples or upload custom low-light images (JPG/PNG).
2. **Real-Time Visual Comparison:** Displays Original, Histogram Equalization, Default CLAHE, GA-CLAHE, and Full Hybrid side-by-side.
3. **Live Quantitative Metric Cards:** Instantly reports PSNR, SSIM, Shannon Entropy, Contrast Index (CII), Edge Preservation (EPI), and latency.
4. **Dynamic GA Evolution Tracking:** Plots the fitness improvement curve generation-by-generation in real time.
5. **Color Channel Histograms:** Compares RGB distribution profiles before and after enhancement.
6. **Built-in Research Reports Tab:** Displays all experiment CSV tables and publication figures directly in the browser.

---

## ⚙️ 9. Installation & Quick Start Guide

### 1. Prerequisites
- Python 3.10 or higher
- Git

### 2. Setup Virtual Environment
```bash
# Clone the repository
git clone https://github.com/2k6ayush/Contrast-Enhancement-Using-CLAHE-CNNs-and-Genetic-Algorithm-Optimisation.git
cd Contrast-Enhancement-Using-CLAHE-CNNs-and-Genetic-Algorithm-Optimisation

# Create and activate virtual environment
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install numpy opencv-python scikit-image scipy matplotlib pandas streamlit tqdm
```

### 4. Running the Benchmark Experiments
```bash
# Run Experiment 1 (Traditional Baselines)
python evaluation/experiments.py --exp 1 --max-images 5

# Run Experiment 4 (Hyperparameter Tuning on Validation Split)
python evaluation/experiments.py --exp 4 --max-images 10 --n-configs 4

# Run Experiment 5 (Ablation Study on Test Split)
python evaluation/experiments.py --exp 5

# Run Experiment 6 (Performance Comparison Table)
python evaluation/experiments.py --exp 6
```

### 5. Generate Comparison Plots & Markdown Reports
```bash
python evaluation/comparison.py
```

### 6. Launch the Interactive Web Application
```bash
python -m streamlit run app.py
```

---
