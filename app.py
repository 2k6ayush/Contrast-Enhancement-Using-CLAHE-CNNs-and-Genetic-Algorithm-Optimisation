"""
app.py — Streamlit Interactive Demo Application
===============================================
Member 3 owns this file.

PROJECT:
    Contrast Enhancement Using CLAHE, CNNs, and Genetic Algorithm Optimisation
    (Low-Light Image Enhancement)

FEATURES:
    1. Interactive side-by-side comparison across all paradigms:
       - Low-light Input
       - Histogram Equalization
       - Default CLAHE (clip=2.0, grid=8)
       - GA-Optimized CLAHE (Proposed Stage 1)
       - Full Hybrid Pipeline (Proposed Stage 1 + 2)
       - Normal-light Ground Truth (if available)
    2. Real-time metric calculation (PSNR, SSIM, Entropy, CII, EPI, Time).
    3. Live GA convergence tracking (fitness history curve).
    4. RGB intensity histogram distributions.
    5. Benchmark viewer: displays generated project figures and experiment CSVs.
    6. System status bar: reports real vs stub fallback modules.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optimization.compat import (
    apply_histogram_equalization,
    apply_clahe,
    calculate_psnr,
    calculate_ssim,
    calculate_entropy,
    calculate_edge_preservation,
    build_model,
    enhance_with_cnn,
    using_stubs,
)
from optimization.hybrid import enhance_hybrid, enhance_ga_clahe
from optimization.genetic_algorithm import optimize_clahe

# ===========================================================================
# STREAMLIT PAGE CONFIG & STYLING
# ===========================================================================

st.set_page_config(
    page_title="Low-Light Image Enhancement Demo",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #64748B;
        text-transform: uppercase;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ===========================================================================
# HELPER FUNCTIONS
# ===========================================================================

def load_rgb_image(file_or_path) -> np.ndarray:
    """Load an image as an RGB uint8 HxWx3 numpy array."""
    if isinstance(file_or_path, (str, Path)):
        img = cv2.imread(str(file_or_path))
        if img is None:
            raise ValueError(f"Could not load image from {file_or_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        # Streamlit UploadedFile
        file_bytes = np.asarray(bytearray(file_or_path.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode uploaded image.")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def compute_all_metrics(
    enhanced: np.ndarray,
    original: np.ndarray,
    gt: Optional[np.ndarray],
    time_ms: float,
) -> dict:
    """Compute performance metrics for an enhanced image."""
    metrics = {
        "time_ms": time_ms,
        "entropy": calculate_entropy(enhanced),
        "contrast_index": float(enhanced.std()) / (float(original.std()) + 1e-8),
    }

    if gt is not None:
        metrics["psnr"] = calculate_psnr(enhanced, gt)
        metrics["ssim"] = calculate_ssim(enhanced, gt)
        metrics["epi"] = calculate_edge_preservation(enhanced, gt)
    else:
        metrics["psnr"] = None
        metrics["ssim"] = None
        metrics["epi"] = calculate_edge_preservation(enhanced, original)

    return metrics


def plot_rgb_histograms(images_dict: dict[str, np.ndarray]) -> plt.Figure:
    """Generate color histogram plots for comparison."""
    fig, axes = plt.subplots(1, len(images_dict), figsize=(4 * len(images_dict), 2.5), dpi=120)
    if len(images_dict) == 1:
        axes = [axes]

    colors = ("red", "green", "blue")
    for ax, (title, img) in zip(axes, images_dict.items()):
        for i, col in enumerate(colors):
            hist, _ = np.histogram(img[:, :, i], bins=64, range=(0, 256))
            ax.plot(hist, color=col, alpha=0.7, linewidth=1.2)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xlim([0, 64])
        ax.set_xticks([0, 32, 64])
        ax.set_xticklabels(["0", "128", "255"], fontsize=8)
        ax.set_yticks([])
        ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    return fig


# ===========================================================================
# SIDEBAR
# ===========================================================================

st.sidebar.markdown("## ⚙️ Configuration & Controls")

# 1. Dataset Image or Custom Upload
data_source = st.sidebar.radio(
    "Choose Input Image Source:",
    ["LOL Dataset Sample", "Upload Custom Image"],
    index=0,
)

eval_dir = PROJECT_ROOT / "dataset" / "eval15"
low_dir = eval_dir / "low"
high_dir = eval_dir / "high"

selected_image: Optional[np.ndarray] = None
ground_truth_image: Optional[np.ndarray] = None
image_name: str = "custom_image"

if data_source == "LOL Dataset Sample":
    if low_dir.exists():
        sample_files = sorted([
            f.name for f in low_dir.iterdir()
            if f.is_file() and not f.name.startswith(".") and f.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ])
        if sample_files:
            chosen_file = st.sidebar.selectbox("Select Test Image:", sample_files)
            image_name = chosen_file
            selected_image = load_rgb_image(low_dir / chosen_file)

            gt_file = high_dir / chosen_file
            if gt_file.exists():
                ground_truth_image = load_rgb_image(gt_file)
                st.sidebar.success(f"✓ Paired normal-light ground truth loaded ({chosen_file})")
            else:
                st.sidebar.info("Ground truth image not found for this sample.")
        else:
            st.sidebar.warning("No test images found in dataset/eval15/low.")
    else:
        st.sidebar.warning("dataset/eval15 folder not found.")
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload Low-Light Image (PNG/JPG):",
        type=["png", "jpg", "jpeg"],
    )
    if uploaded_file is not None:
        selected_image = load_rgb_image(uploaded_file)
        image_name = uploaded_file.name

    gt_uploaded = st.sidebar.file_uploader(
        "Upload Optional Ground Truth (Reference):",
        type=["png", "jpg", "jpeg"],
    )
    if gt_uploaded is not None:
        ground_truth_image = load_rgb_image(gt_uploaded)
        st.sidebar.success("✓ Reference ground truth loaded.")

# 2. GA Hyperparameter Controls
with st.sidebar.expander("🧬 Genetic Algorithm Settings", expanded=False):
    ga_pop_size = st.slider("Population Size", min_value=10, max_value=50, value=30, step=5)
    ga_generations = st.slider("Generations", min_value=10, max_value=50, value=30, step=5)
    ga_crossover = st.slider("Crossover Rate (BLX-α)", min_value=0.5, max_value=0.95, value=0.8, step=0.05)
    ga_mutation = st.slider("Mutation Rate", min_value=0.01, max_value=0.20, value=0.05, step=0.01)
    ga_seed = st.number_input("Random Seed", value=42, step=1)
    ga_work_size = st.select_slider("Downscale Work Size (for GA speed)", options=[128, 256, 384, 512], value=256)

# 3. Model Settings
with st.sidebar.expander("🧠 CNN Model Settings", expanded=False):
    model_path_input = st.text_input("Model Weights Path (.pth / .pt):", value="")
    st.caption("Leave empty if teammate's CNN weights are not yet available.")


# ===========================================================================
# MAIN INTERFACE
# ===========================================================================

st.markdown('<div class="main-header">Contrast Enhancement using CLAHE, CNNs & GA</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Hybrid Low-Light Image Enhancement Pipeline Demonstration | Member 3: GA & System Integration</div>', unsafe_allow_html=True)

# Status bar: Module integration check
stubs_status = using_stubs()
active_stubs = [k for k, v in stubs_status.items() if v]

if active_stubs:
    st.info(
        f"ℹ️ **Integration Status:** Running in Standalone / Compatibility Mode ({len(active_stubs)} fallback stubs active). "
        "Real teammate implementations will automatically engage when merged."
    )
else:
    st.success("✅ **Integration Status:** All teammate modules (Traditional & CNN) are fully merged and active.")

# Navigation Tabs
tab_demo, tab_reports, tab_arch = st.tabs([
    "🚀 Live Enhancement & Comparison",
    "📊 Experiment Benchmark Reports",
    "📘 Project Architecture & Guide",
])

# ---------------------------------------------------------------------------
# TAB 1: LIVE ENHANCEMENT DEMO
# ---------------------------------------------------------------------------
with tab_demo:
    if selected_image is None:
        st.warning("👈 Please select or upload a low-light image from the sidebar to begin.")
    else:
        st.markdown(f"#### Input Image: `{image_name}` (Resolution: {selected_image.shape[1]}x{selected_image.shape[0]})")

        run_btn = st.button("✨ Run Enhancement Pipeline", type="primary", use_container_width=True)

        if run_btn:
            with st.spinner("Processing through traditional, GA-optimized CLAHE, and hybrid stages..."):
                results_data = {}

                # 1. Histogram Equalization
                t0 = time.perf_counter()
                he_img = apply_histogram_equalization(selected_image)
                t_he = (time.perf_counter() - t0) * 1000.0
                results_data["HE"] = (he_img, compute_all_metrics(he_img, selected_image, ground_truth_image, t_he))

                # 2. Default CLAHE
                t0 = time.perf_counter()
                clahe_def = apply_clahe(selected_image, clip_limit=2.0, grid_size=8)
                t_def = (time.perf_counter() - t0) * 1000.0
                results_data["Default CLAHE"] = (clahe_def, compute_all_metrics(clahe_def, selected_image, ground_truth_image, t_def))

                # 3. GA-CLAHE (Proposed Stage 1)
                t0 = time.perf_counter()
                ga_res = optimize_clahe(
                    selected_image,
                    ground_truth=None,  # Real-world no-reference deployment
                    pop_size=ga_pop_size,
                    generations=ga_generations,
                    crossover_rate=ga_crossover,
                    mutation_rate=ga_mutation,
                    seed=int(ga_seed),
                    work_size=ga_work_size,
                )
                ga_img = ga_res["enhanced"]
                t_ga = (time.perf_counter() - t0) * 1000.0
                results_data["GA-CLAHE"] = (ga_img, compute_all_metrics(ga_img, selected_image, ground_truth_image, t_ga))

                # 4. Full Hybrid
                # Try loading model if path given
                loaded_model = None
                if model_path_input and Path(model_path_input).exists():
                    try:
                        import torch
                        loaded_model = build_model()
                        loaded_model.load_state_dict(torch.load(model_path_input, map_location="cpu"))
                        loaded_model.eval()
                    except Exception:
                        loaded_model = None

                hybrid_res = enhance_hybrid(
                    selected_image,
                    model=loaded_model,
                    ground_truth=None,
                    pop_size=ga_pop_size,
                    generations=ga_generations,
                    crossover_rate=ga_crossover,
                    mutation_rate=ga_mutation,
                    seed=int(ga_seed),
                    work_size=ga_work_size,
                )
                hybrid_img = hybrid_res["output"]
                results_data["Hybrid (Proposed)"] = (
                    hybrid_img,
                    compute_all_metrics(hybrid_img, selected_image, ground_truth_image, hybrid_res["time_ms"]),
                )

                # Store in session state
                st.session_state["results"] = results_data
                st.session_state["ga_result"] = ga_res

        # If results exist, display them
        if "results" in st.session_state:
            res = st.session_state["results"]
            ga_res = st.session_state["ga_result"]

            # Visual Grid
            st.markdown("### 🖼️ Visual Comparison")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(selected_image, caption="1. Original Low-Light Input", use_container_width=True)
                st.image(res["HE"][0], caption="2. Histogram Equalization", use_container_width=True)
            with col2:
                st.image(res["Default CLAHE"][0], caption="3. Default CLAHE (clip=2.0, grid=8)", use_container_width=True)
                st.image(
                    res["GA-CLAHE"][0],
                    caption=f"4. GA-Optimized CLAHE (clip={ga_res['clip_limit']:.2f}, grid={ga_res['tile_grid_size']})",
                    use_container_width=True,
                )
            with col3:
                st.image(res["Hybrid (Proposed)"][0], caption="5. Full Hybrid (GA-CLAHE + CNN)", use_container_width=True)
                if ground_truth_image is not None:
                    st.image(ground_truth_image, caption="6. Normal-Light Ground Truth", use_container_width=True)
                else:
                    st.info("No Ground Truth available for reference comparison.")

            # Metrics Table
            st.markdown("### 📈 Quantitative Performance Metrics")
            table_rows = []
            for name, (_, m) in res.items():
                row = {
                    "Method": name,
                    "Execution Time (ms)": f"{m['time_ms']:.1f}",
                    "Entropy (bits)": f"{m['entropy']:.3f}",
                    "Contrast (CII)": f"{m['contrast_index']:.2f}",
                }
                if m["psnr"] is not None:
                    row["PSNR (dB)"] = f"{m['psnr']:.2f}"
                    row["SSIM"] = f"{m['ssim']:.4f}"
                    row["Edge Pres. (EPI)"] = f"{m['epi']:.4f}"
                else:
                    row["Edge Pres. (vs Input)"] = f"{m['epi']:.4f}"
                table_rows.append(row)

            df_metrics = pd.DataFrame(table_rows)
            st.dataframe(df_metrics, use_container_width=True, hide_index=True)

            # GA Parameter Details & Convergence
            st.markdown("### 🧬 Genetic Algorithm Convergence Analysis")
            ga_col1, ga_col2 = st.columns([1, 2])

            with ga_col1:
                st.markdown(
                    f"""
                    **Optimal Parameters Discovered:**
                    - **Clip Limit:** `{ga_res['clip_limit']:.4f}`
                    - **Tile Grid Size:** `{ga_res['tile_grid_size']} x {ga_res['tile_grid_size']}`
                    - **Generations Run:** `{ga_res['generations_run']}`
                    - **Best Fitness Score:** `{ga_res['best_fitness']:.4f}`
                    - **Optimization Time:** `{ga_res['time_sec']:.2f} s`
                    """
                )

            with ga_col2:
                best_history = ga_res["history"]["best"]
                mean_history = ga_res["history"]["mean"]
                fig_ga, ax_ga = plt.subplots(figsize=(6, 2.5), dpi=120)
                gens = range(1, len(best_history) + 1)
                ax_ga.plot(gens, best_history, "b-o", markersize=3, label="Best Fitness (Elites)")
                ax_ga.plot(gens, mean_history, "r--s", markersize=2, label="Mean Population Fitness")
                ax_ga.set_xlabel("Generation", fontsize=8)
                ax_ga.set_ylabel("Fitness Score", fontsize=8)
                ax_ga.set_title("GA Evolution Progress", fontsize=9, fontweight="bold")
                ax_ga.legend(fontsize=8)
                ax_ga.grid(True, linestyle="--", alpha=0.4)
                fig_ga.tight_layout()
                st.pyplot(fig_ga)
                plt.close(fig_ga)

            # Histograms
            with st.expander("📊 Color Channel Histogram Distributions", expanded=False):
                hist_imgs = {
                    "Original": selected_image,
                    "HE": res["HE"][0],
                    "Default CLAHE": res["Default CLAHE"][0],
                    "GA-CLAHE": res["GA-CLAHE"][0],
                    "Hybrid": res["Hybrid (Proposed)"][0],
                }
                fig_hist = plot_rgb_histograms(hist_imgs)
                st.pyplot(fig_hist)
                plt.close(fig_hist)


# ---------------------------------------------------------------------------
# TAB 2: BENCHMARK & EVALUATION REPORTS
# ---------------------------------------------------------------------------
with tab_reports:
    st.markdown("### 📊 Project Benchmark & Experimental Results")
    st.write(
        "Quantitative results from all 6 systematic experiments run on the LOL dataset "
        "(Low-Light paired images: `our485` for training/validation, `eval15` for testing)."
    )

    results_dir = PROJECT_ROOT / "results"
    figures_dir = results_dir / "figures"

    rep_tab1, rep_tab2, rep_tab3 = st.tabs([
        "🏆 Overall Performance",
        "🔬 Ablation Study",
        "⚙️ Hyperparameter Sweeps",
    ])

    with rep_tab1:
        cmp_csv = results_dir / "comparison_table.csv"
        if cmp_csv.exists():
            df_cmp = pd.read_csv(cmp_csv)
            st.dataframe(df_cmp, use_container_width=True)

        col_f1, col_f2 = st.columns(2)
        bar_fig = figures_dir / "comparison_metrics_barchart.png"
        tradeoff_fig = figures_dir / "runtime_tradeoff.png"

        with col_f1:
            if bar_fig.exists():
                st.image(str(bar_fig), caption="Performance Metrics with Error Bars", use_container_width=True)
        with col_f2:
            if tradeoff_fig.exists():
                st.image(str(tradeoff_fig), caption="Quality (SSIM) vs. Execution Time Trade-off", use_container_width=True)

    with rep_tab2:
        abl_csv = results_dir / "ablation.csv"
        if abl_csv.exists():
            df_abl = pd.read_csv(abl_csv)
            st.dataframe(
                df_abl.groupby(["config", "config_label"])[["psnr", "ssim", "entropy", "contrast_improvement_index", "edge_preservation", "time_ms"]].mean(),
                use_container_width=True,
            )

        abl_fig = figures_dir / "ablation_boxplots.png"
        if abl_fig.exists():
            st.image(str(abl_fig), caption="Component Distributions across Test Images", use_container_width=True)

    with rep_tab3:
        tune_csv = results_dir / "hyperparameter_tuning.csv"
        if tune_csv.exists():
            df_tune = pd.read_csv(tune_csv)
            st.markdown("#### Hyperparameter Tuning (72 Validation Images)")
            st.dataframe(df_tune.head(15), use_container_width=True)

        heat_fig = figures_dir / "clahe_tuning_heatmap.png"
        if heat_fig.exists():
            st.image(str(heat_fig), caption="Fixed CLAHE Parameter Landscape (Validation SSIM)", use_container_width=True)


# ---------------------------------------------------------------------------
# TAB 3: PROJECT ARCHITECTURE & DOCUMENTATION
# ---------------------------------------------------------------------------
with tab_arch:
    st.markdown("### 📘 System Architecture & Methodology")
    st.markdown(
        """
        #### 1. The Core Problem
        Low-light images suffer from non-uniform dark regions, low dynamic range, and sensor noise.
        Traditional **Histogram Equalization** applies global stretching that frequently blows out bright spots
        and amplifies noise. **Default CLAHE** (`clip=2.0, grid=8`) provides local equalization, but its fixed
        parameters fail across images with different illumination levels.

        #### 2. The Proposed Hybrid Pipeline
        ```
        Low-Light Image
              │
              ▼
        ┌─────────────────────────────────────────────────────────┐
        │  STAGE 1: Genetic Algorithm Parameter Optimisation      │
        │  - Chromosome: [clip_limit ∈ [1, 8], grid_size ∈ [4, 16]]│
        │  - Fitness: Multi-objective (Entropy + Edge + Brightness│
        │             minus Immerkaer Noise Penalty)              │
        │  - Operators: Tournament Selection, BLX-α Crossover,     │
        │               Gaussian / Step Mutation, Elitism         │
        └────────────────────────────┬────────────────────────────┘
                                     │
                           Optimized CLAHE Image
                                     │
                                     ▼
        ┌─────────────────────────────────────────────────────────┐
        │  STAGE 2: Deep Convolutional Neural Network Refinement  │
        │  - Learns residual feature mapping                      │
        │  - Removes residual artifacts and smooths color tones   │
        └────────────────────────────┬────────────────────────────┘
                                     │
                                     ▼
                             Enhanced RGB Image
        ```

        #### 3. Fitness Formulation (No-Reference Mode)
        When no paired ground truth exists, the Genetic Algorithm maximizes:
        $$\\text{Fitness} = 0.35 \\left(\\frac{\\text{Entropy}}{8}\\right) + 0.35 (\\text{EPI}) + 0.30 (\\text{Brightness}) - 0.20 (\\text{Noise Penalty})$$
        Where:
        - **EPI:** Pearson correlation of Sobel edge gradients with the input.
        - **Brightness:** Penalizes mean intensities deviating from mid-gray (128).
        - **Noise Penalty:** Convolved with Immerkaer's Laplacian kernel, penalizing noise *amplification* ($> 0$).
        """
    )
