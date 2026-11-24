"""Streamlit studio for low-light image enhancement."""
from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cnn.inference import enhance_with_cnn, load_checkpoint
from optimization.compat import (
    apply_clahe, apply_histogram_equalization, calculate_edge_preservation,
    calculate_entropy, calculate_psnr, calculate_ssim, using_stubs,
)
from optimization.genetic_algorithm import optimize_clahe

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

st.set_page_config(page_title="Low-Light Image Enhancement", page_icon="◐", layout="wide")
st.markdown("""
<style>
:root{--bg:#F7F5EF;--side:#252720;--text:#23251F;--accent:#C65324;--amber:#E8B44A;--success:#506139;--border:#DEDCD2}
.stApp{background:var(--bg);color:var(--text)}
[data-testid="stHeader"]{background:rgba(247,245,239,.96)}
[data-testid="stSidebar"]{background:var(--side)}
[data-testid="stSidebar"] *{color:#F7F5EF}
[data-testid="stSidebar"] input{color:var(--text)!important;background:#F7F5EF!important}
[data-testid="stSidebar"] [data-baseweb="select"] *{color:var(--text)!important}
h1,h2,h3,h4{color:var(--text);letter-spacing:-.02em} h1{font-size:clamp(2rem,4vw,3rem)!important;line-height:1.05!important}
.subtitle{color:#5B5D54;font-size:1.05rem;margin:-.3rem 0 1.5rem}
.workflow{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.75rem;margin-bottom:1.5rem}
.step{border-top:3px solid var(--amber);padding:.75rem;background:#FCFBF7}.step strong{color:var(--accent);margin-right:.35rem}
.model{display:flex;justify-content:space-between;gap:.5rem;padding:.45rem 0;border-bottom:1px solid #46483F;font-size:.88rem}
.ready{color:#D8E5C8!important;font-weight:700}.missing{color:#F2C876!important;font-weight:700}
.meta{border-left:3px solid var(--amber);padding:.65rem .8rem;background:#FCFBF7;color:#5B5D54}
.note{border:1px solid var(--border);border-radius:5px;padding:.75rem;background:#FCFBF7;color:#5B5D54;margin-bottom:1rem}
.footer{border-top:1px solid var(--border);margin-top:3rem;padding:1rem 0;color:#6A6C63;font-size:.82rem}
.stButton>button,.stDownloadButton>button{border-radius:5px;border-color:var(--accent);font-weight:700;min-height:2.75rem}
.stButton>button[kind="primary"]{background:var(--accent);color:white}
.stButton>button[kind="primary"]:hover:not(:disabled),.stDownloadButton>button:hover{background:#A9441F;color:white;border-color:#A9441F}
button:focus-visible,input:focus-visible,[role="radio"]:focus-visible,[role="tab"]:focus-visible{outline:3px solid var(--amber)!important;outline-offset:2px}
[role="tab"][aria-selected="true"]{color:var(--accent)!important;border-bottom-color:var(--accent)!important}
@media(max-width:760px){.workflow{grid-template-columns:1fr}.step{padding:.55rem .7rem}}
</style>""", unsafe_allow_html=True)


def load_rgb_image(source) -> np.ndarray:
    if isinstance(source, (str, Path)):
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        label = str(source)
    else:
        image = cv2.imdecode(np.frombuffer(source.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        label = getattr(source, "name", "uploaded image")
    if image is None:
        raise ValueError(f"Could not decode image: {label}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def encode_png(image: np.ndarray) -> bytes:
    ok, data = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("Could not encode the selected result")
    return data.tobytes()


def metrics(enhanced, original, reference, elapsed_ms) -> dict:
    values = {
        "time_ms": elapsed_ms,
        "entropy": calculate_entropy(enhanced),
        "contrast": float(enhanced.std()) / (float(original.std()) + 1e-8),
        "edge": calculate_edge_preservation(enhanced, reference if reference is not None else original),
        "psnr": None, "ssim": None,
    }
    if reference is not None:
        values["psnr"] = calculate_psnr(enhanced, reference)
        values["ssim"] = calculate_ssim(enhanced, reference)
    return values


def project_path(value: str) -> Optional[Path]:
    if not value.strip():
        return None
    path = Path(value.strip()).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def _case_insensitive_child(parent: Path, name: str) -> Optional[Path]:
    """Return a child directory without assuming filesystem case semantics."""
    if not parent.is_dir():
        return None
    return next(
        (child for child in parent.iterdir() if child.is_dir() and child.name.casefold() == name.casefold()),
        None,
    )


def discover_dataset_pairs(dataset_root: Optional[Path]) -> tuple[Path | None, list[dict], list[tuple[str, str]]]:
    """Find decodable eval15 low/high pairs and return precise diagnostics.

    Discovery is deliberately uncached so a Streamlit rerun immediately sees
    restored, downloaded, or externally configured dataset files.
    """
    diagnostics: list[tuple[str, str]] = []
    if dataset_root is None:
        return None, [], [("error", "No dataset root is configured.")]
    if not dataset_root.is_dir():
        return dataset_root / "eval15", [], [
            ("error", f"Dataset root does not exist: {dataset_root}")
        ]

    # Accept either the dataset root or eval15 itself as the configured path.
    if dataset_root.name.casefold() == "eval15":
        eval_root = dataset_root
    else:
        eval_root = _case_insensitive_child(dataset_root, "eval15")
        if eval_root is None:
            return dataset_root / "eval15", [], [
                ("error", f"Evaluation folder is missing: {dataset_root / 'eval15'}")
            ]

    low_dir = _case_insensitive_child(eval_root, "low")
    high_dir = _case_insensitive_child(eval_root, "high")
    if low_dir is None:
        diagnostics.append(("error", f"Low-light folder is missing: {eval_root / 'low'}"))
    if high_dir is None:
        diagnostics.append(("error", f"Normal-light folder is missing: {eval_root / 'high'}"))
    if low_dir is None or high_dir is None:
        return eval_root, [], diagnostics

    low_files = sorted(
        (path for path in low_dir.iterdir() if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS),
        key=lambda path: path.name.casefold(),
    )
    high_files = sorted(
        (path for path in high_dir.iterdir() if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS),
        key=lambda path: path.name.casefold(),
    )
    if not low_files:
        diagnostics.append(("warning", f"Low-light folder contains no supported images: {low_dir}"))
    if not high_files:
        diagnostics.append(("warning", f"Normal-light folder contains no supported images: {high_dir}"))
    if not low_files or not high_files:
        return eval_root, [], diagnostics

    high_by_name: dict[str, list[Path]] = {}
    for path in high_files:
        high_by_name.setdefault(path.name.casefold(), []).append(path)

    pairs: list[dict] = []
    missing_partners: list[str] = []
    unreadable: list[str] = []
    ambiguous: list[str] = []
    for low_path in low_files:
        partners = high_by_name.get(low_path.name.casefold(), [])
        if not partners:
            missing_partners.append(low_path.name)
            continue
        if len(partners) > 1:
            ambiguous.append(low_path.name)
            continue
        high_path = partners[0]
        low_check = cv2.imread(str(low_path), cv2.IMREAD_COLOR)
        high_check = cv2.imread(str(high_path), cv2.IMREAD_COLOR)
        if low_check is None or high_check is None:
            unreadable.append(low_path.name if low_check is None else high_path.name)
            continue
        if low_check.shape != high_check.shape:
            diagnostics.append((
                "error",
                f"Pair dimensions differ for {low_path.name}: {low_check.shape[:2]} vs {high_check.shape[:2]}",
            ))
            continue
        pairs.append({"name": low_path.name, "low": low_path, "high": high_path})

    if missing_partners:
        diagnostics.append((
            "warning",
            f"{len(missing_partners)} low-light image(s) have no filename-matched partner: "
            + ", ".join(missing_partners[:5]),
        ))
    if ambiguous:
        diagnostics.append((
            "error",
            f"{len(ambiguous)} low-light image(s) have ambiguous normal-light partners: "
            + ", ".join(ambiguous[:5]),
        ))
    if unreadable:
        diagnostics.append((
            "error",
            f"{len(unreadable)} image(s) could not be decoded: " + ", ".join(unreadable[:5]),
        ))
    if pairs:
        diagnostics.append(("success", f"Found {len(pairs)} valid low/normal-light pair(s)."))
    return eval_root, pairs, diagnostics


@st.cache_resource(show_spinner=False)
def cached_checkpoint(path: str, modified_ns: int, expected_mode: str):
    del modified_ns
    return load_checkpoint(path, "auto", expected_mode)


def path_token(path: Optional[Path]):
    return str(path) if path else "", path.stat().st_mtime_ns if path and path.is_file() else None


def render_legacy_results() -> None:
    st.subheader("Experiment results")
    st.warning("These CSV files and figures predate the newly trained checkpoints and lack checkpoint provenance. They are legacy project artifacts, not measurements from the current models.")
    results, figures = ROOT / "results", ROOT / "results" / "figures"
    traditional, overall, ablation, tuning = st.tabs(
        ["Traditional methods", "Overall performance", "Ablation study", "Parameter sweeps"]
    )
    with traditional:
        st.markdown("**Dhanushree (Member 1) — traditional image processing**")
        st.caption(
            "Imported from Member 1's traditional-method work. These are existing saved "
            "artifacts; opening this tab does not rerun evaluation."
        )
        summary = results / "traditional_summary.csv"
        if summary.is_file():
            st.dataframe(pd.read_csv(summary), width="stretch", hide_index=True)
        else:
            st.info("No traditional-method summary is available.")
        st.caption(
            "The live Enhance workflow now uses Member 1's Histogram Equalisation, CLAHE, "
            "PSNR, SSIM and entropy implementations through the compatibility interface."
        )
        saved_examples = sorted(
            path for path in figures.glob("traditional_*.png")
            if not path.name.startswith("traditional_error_")
        )
        if saved_examples:
            with st.expander("View saved traditional-method comparisons", expanded=False):
                selected_example = st.selectbox(
                    "Saved comparison",
                    saved_examples,
                    format_func=lambda path: path.stem.removeprefix("traditional_") + ".png",
                )
                st.image(
                    str(selected_example),
                    caption="Saved Member 1 comparison artifact (legacy result)",
                    width="stretch",
                )
    with overall:
        source = results / "comparison_table.csv"
        st.dataframe(pd.read_csv(source), width="stretch", hide_index=True) if source.is_file() else st.info("No legacy comparison table is available.")
        a, b = st.columns(2)
        with a:
            figure = figures / "comparison_metrics_barchart.png"
            if figure.is_file(): st.image(str(figure), caption="Legacy metric comparison", width="stretch")
        with b:
            figure = figures / "runtime_tradeoff.png"
            if figure.is_file(): st.image(str(figure), caption="Legacy runtime trade-off", width="stretch")
    with ablation:
        source = results / "ablation.csv"
        if source.is_file():
            frame = pd.read_csv(source)
            cols = ["psnr", "ssim", "entropy", "contrast_improvement_index", "edge_preservation", "time_ms"]
            st.dataframe(frame.groupby(["config", "config_label"])[cols].mean(), width="stretch")
        else: st.info("No legacy ablation table is available.")
        figure = figures / "ablation_boxplots.png"
        if figure.is_file(): st.image(str(figure), caption="Legacy ablation distributions", width="stretch")
    with tuning:
        source = results / "hyperparameter_tuning.csv"
        st.dataframe(pd.read_csv(source).head(15), width="stretch", hide_index=True) if source.is_file() else st.info("No legacy parameter-sweep table is available.")
        figure = figures / "clahe_tuning_heatmap.png"
        if figure.is_file(): st.image(str(figure), caption="Legacy CLAHE parameter landscape", width="stretch")


# Technical controls and diagnostics stay out of the primary workflow.
st.sidebar.markdown("### Studio controls")
st.sidebar.caption("Models and technical configuration")
with st.sidebar.expander("Advanced settings", expanded=False):
    st.markdown("**Genetic algorithm**")
    ga_pop = st.slider("Population", 10, 50, 30, 5)
    ga_generations = st.slider("Generations", 10, 50, 30, 5)
    ga_crossover = st.slider("Crossover rate", .50, .95, .80, .05)
    ga_mutation = st.slider("Mutation rate", .01, .20, .05, .01)
    ga_seed = st.number_input("Random seed", value=42, step=1)
    ga_work_size = st.select_slider("GA work size", [128, 256, 384, 512], value=256)
    st.divider()
    st.markdown("**Dataset**")
    dataset_root_input = st.text_input(
        "Dataset root",
        value=os.environ.get("CLAHE_CNN_DATASET_ROOT", "Dataset"),
        help=(
            "Absolute path or a path relative to app.py. You may provide the dataset root "
            "containing eval15, or the eval15 folder itself. The environment variable "
            "CLAHE_CNN_DATASET_ROOT supplies the default."
        ),
    )
    st.divider()
    st.markdown("**Checkpoint paths**")
    baseline_input = st.text_input("Baseline CNN", "models/baseline/best.pt", help="Raw low-light RGB to enhanced RGB.")
    refinement_input = st.text_input("Refinement CNN", "models/refinement/best.pt", help="GA–CLAHE RGB to refined RGB.")

dataset_root = project_path(dataset_root_input)
baseline_path, refinement_path = project_path(baseline_input), project_path(refinement_input)
baseline_model = refinement_model = None
baseline_meta = refinement_meta = None
model_errors = {}
for label, path, mode in (("Baseline", baseline_path, "baseline"), ("Refinement", refinement_path, "ga_clahe_refinement")):
    if not path or not path.is_file():
        model_errors[label] = f"Checkpoint not found: {path or '(no path supplied)'}"
        continue
    try:
        model, metadata, _ = cached_checkpoint(str(path), path.stat().st_mtime_ns, mode)
        if label == "Baseline": baseline_model, baseline_meta = model, metadata
        else: refinement_model, refinement_meta = model, metadata
    except Exception as exc:
        model_errors[label] = f"{type(exc).__name__}: {exc}"

st.sidebar.markdown(f'''<div class="model"><span>Baseline CNN</span><span class="{'ready' if baseline_model is not None else 'missing'}">{'Loaded' if baseline_model is not None else 'Unavailable'}</span></div><div class="model"><span>Refinement CNN</span><span class="{'ready' if refinement_model is not None else 'missing'}">{'Loaded' if refinement_model is not None else 'Unavailable'}</span></div><div class="model"><span>Traditional methods</span><span class="ready">Dhanushree / Member 1</span></div>''', unsafe_allow_html=True)
for label, error in model_errors.items(): st.sidebar.error(f"{label}: {error}")

ga_settings = {"pop_size": ga_pop, "generations": ga_generations, "crossover_rate": ga_crossover, "mutation_rate": ga_mutation, "seed": int(ga_seed), "work_size": ga_work_size}
recorded_settings = (refinement_meta or {}).get("ga_preprocessing", {}).get("ga_settings", {})
preprocessing_matches = bool(recorded_settings) and all(recorded_settings.get(k) == v for k, v in ga_settings.items())
if refinement_model is not None and not preprocessing_matches:
    st.sidebar.warning("GA settings differ from the refinement checkpoint metadata. Hybrid inference remains available, but preprocessing parity is not preserved.")
fallbacks = [name for name, active in using_stubs().items() if active]
with st.sidebar.expander("Compatibility details"):
    if fallbacks:
        st.caption("Compatibility fallbacks currently active:")
        for name in fallbacks: st.write(f"• {name}")
    else: st.write("No compatibility fallbacks are active.")
    st.caption("NIQE is unavailable because no validated implementation is installed.")

st.title("Low-Light Image Enhancement")
st.markdown('<p class="subtitle">Compare traditional methods, CNN enhancement and GA optimisation.</p>', unsafe_allow_html=True)
st.markdown('<div class="workflow"><div class="step"><strong>1.</strong>Choose an image</div><div class="step"><strong>2.</strong>Enhance image</div><div class="step"><strong>3.</strong>Compare and download</div></div>', unsafe_allow_html=True)
enhance_tab, experiments_tab = st.tabs(["Enhance", "Experiment results"])

with enhance_tab:
    st.subheader("1. Choose an image")
    source_mode = st.radio("Image source", ["Upload image", "Try a dataset sample"], horizontal=True)
    image = reference = None
    image_name, input_error = "image", None
    if source_mode == "Upload image":
        left, right = st.columns(2)
        with left: uploaded = st.file_uploader("Low-light image", ["png", "jpg", "jpeg"])
        with right: reference_upload = st.file_uploader("Paired normal-light reference (optional)", ["png", "jpg", "jpeg"], help="Used for metrics only, never GA optimisation.")
        try:
            if uploaded is not None: image, image_name = load_rgb_image(uploaded), uploaded.name
            if reference_upload is not None: reference = load_rgb_image(reference_upload)
        except Exception as exc: input_error = str(exc)
    else:
        eval_root, sample_pairs, dataset_diagnostics = discover_dataset_pairs(dataset_root)
        st.caption(f"Resolved sample path: {eval_root or '(not configured)'}")
        st.button(
            "Refresh dataset samples",
            key="refresh_dataset_samples",
            help="Dataset discovery is uncached; this reruns the scan immediately.",
        )
        for level, message in dataset_diagnostics:
            getattr(st, level)(message)
        if sample_pairs:
            sample_names = [pair["name"] for pair in sample_pairs]
            image_name = st.selectbox("Dataset sample", sample_names)
            selected_pair = sample_pairs[sample_names.index(image_name)]
            try:
                image = load_rgb_image(selected_pair["low"])
                reference = load_rgb_image(selected_pair["high"])
                st.caption(
                    f"Low-light: {selected_pair['low'].name} · "
                    f"Normal-light reference: {selected_pair['high'].name}"
                )
            except Exception as exc:
                input_error = str(exc)
    if input_error: st.error(f"Image loading failed: {input_error}")
    if image is not None:
        if reference is not None and reference.shape != image.shape:
            input_error = "reference dimensions do not match"
            st.error("The paired reference dimensions do not match the input. Remove or replace it before enhancing.")
        if reference is not None and not input_error:
            low_preview, reference_preview = st.columns(2)
            with low_preview:
                st.image(image, caption=f"Low-light input · {image_name}", width="stretch")
            with reference_preview:
                st.image(reference, caption="Matched normal-light reference", width="stretch")
            st.success("Paired reference ready for PSNR and SSIM.")
        else:
            preview, details = st.columns([2, 1])
            with preview:
                st.image(image, caption="Input preview", width="stretch")
            with details:
                st.caption("No paired reference. PSNR and SSIM will not be shown.")
        st.markdown(
            f'<div class="meta"><strong>{image_name}</strong><br>{image.shape[1]} × {image.shape[0]} pixels<br>RGB, uint8</div>',
            unsafe_allow_html=True,
        )
    else: st.caption("Upload a low-light image to activate enhancement.")

    signature = hashlib.sha256(b"".join([
        image.tobytes() if image is not None else b"",
        reference.tobytes() if reference is not None else b"",
        repr((image_name, str(dataset_root), ga_settings, path_token(baseline_path), path_token(refinement_path))).encode(),
    ])).hexdigest()
    previous = st.session_state.get("result_signature")
    invalidated = previous is not None and previous != signature
    if invalidated:
        for key in ("results", "ga_result", "result_signature", "selected_method"): st.session_state.pop(key, None)
        st.session_state["processing_state"] = "idle"

    st.subheader("2. Enhance image")
    if invalidated: st.info("The image or processing configuration changed. Run enhancement again to refresh results.")
    clicked = st.button("Enhance image", type="primary", width="stretch", disabled=image is None or input_error is not None)
    if clicked and image is not None:
        st.session_state["processing_state"] = "processing"
        try:
            results = {}
            with st.status("Enhancing image…", expanded=True) as status:
                st.write("Applying histogram equalisation and CLAHE")
                start = time.perf_counter(); output = apply_histogram_equalization(image); elapsed = (time.perf_counter()-start)*1000
                results["Histogram equalisation"] = {"image": output, "metrics": metrics(output, image, reference, elapsed)}
                start = time.perf_counter(); clahe = apply_clahe(image, 2.0, 8); clahe_ms = (time.perf_counter()-start)*1000
                results["CLAHE"] = {"image": clahe, "metrics": metrics(clahe, image, reference, clahe_ms)}
                if baseline_model is not None:
                    st.write("Running the baseline CNN")
                    start = time.perf_counter(); output = enhance_with_cnn(image, baseline_model); elapsed = (time.perf_counter()-start)*1000
                    results["CNN"] = {"image": output, "metrics": metrics(output, image, reference, elapsed)}
                st.write("Searching for GA–CLAHE parameters")
                start = time.perf_counter()
                ga_result = optimize_clahe(image, ground_truth=None, pop_size=ga_pop, generations=ga_generations, crossover_rate=ga_crossover, mutation_rate=ga_mutation, seed=int(ga_seed), work_size=ga_work_size)
                ga_image, ga_ms = ga_result["enhanced"], (time.perf_counter()-start)*1000
                results["GA–CLAHE"] = {"image": ga_image, "metrics": metrics(ga_image, image, reference, ga_ms)}
                if refinement_model is not None:
                    st.write("Running CNN refinement")
                    start = time.perf_counter(); fixed = enhance_with_cnn(clahe, refinement_model); elapsed = (time.perf_counter()-start)*1000
                    results["Fixed CLAHE + CNN (experimental input mismatch)"] = {"image": fixed, "metrics": metrics(fixed, image, reference, clahe_ms+elapsed)}
                    start = time.perf_counter(); hybrid = enhance_with_cnn(ga_image, refinement_model); elapsed = (time.perf_counter()-start)*1000
                    results["GA–CLAHE + CNN"] = {"image": hybrid, "metrics": metrics(hybrid, image, reference, ga_ms+elapsed)}
                st.session_state.update(results=results, ga_result=ga_result, result_signature=signature, processing_state="completed", ga_run_count=st.session_state.get("ga_run_count", 0)+1)
                status.update(label="Enhancement completed", state="complete", expanded=False)
        except Exception as exc:
            st.session_state.pop("results", None); st.session_state.pop("ga_result", None)
            st.session_state.update(processing_state="failed", processing_error=f"{type(exc).__name__}: {exc}")

    state = st.session_state.get("processing_state", "idle")
    if state == "idle" and image is not None: st.caption("Ready to enhance.")
    elif state == "completed" and st.session_state.get("result_signature") == signature: st.success("Results are ready. Select a method to compare or download.")
    elif state == "failed": st.error(f"Enhancement failed: {st.session_state.get('processing_error', 'Unknown error')}")

    results = st.session_state.get("results") if st.session_state.get("result_signature") == signature else None
    if results:
        st.subheader("3. Compare and download")
        methods = list(results)
        if st.session_state.get("selected_method") not in methods: st.session_state["selected_method"] = "CNN" if "CNN" in methods else methods[0]
        method = st.selectbox("Enhanced result", methods, key="selected_method")
        chosen = results[method]
        if method.startswith("Fixed CLAHE"):
            st.markdown('<div class="note"><strong>Experimental comparison:</strong> refinement weights trained on GA–CLAHE input are being applied to fixed CLAHE input.</div>', unsafe_allow_html=True)
        elif method == "GA–CLAHE + CNN" and not preprocessing_matches:
            st.warning("Current GA settings differ from the refinement checkpoint metadata; preprocessing parity is not preserved.")
        original_col, result_col = st.columns(2)
        with original_col: st.markdown("#### Original"); st.image(image, caption=image_name, width="stretch")
        with result_col: st.markdown("#### Enhanced"); st.image(chosen["image"], caption=method, width="stretch")
        stem = Path(image_name).stem or "image"
        method_slug = "".join(c.lower() if c.isalnum() else "-" for c in method).strip("-")
        st.download_button("Download selected result", encode_png(chosen["image"]), f"{stem}-{method_slug}.png", "image/png", width="stretch")
        if reference is not None:
            with st.expander("View paired normal-light reference"): st.image(reference, caption="Normal-light reference", width="stretch")
        else: st.info("PSNR and SSIM are unavailable because this image has no genuine paired normal-light reference.")
        if st.toggle("View all methods"):
            columns = st.columns(3)
            for index, (name, result) in enumerate(results.items()):
                with columns[index % 3]: st.image(result["image"], caption=name, width="stretch")
        with st.expander("Detailed metrics and GA details"):
            rows = []
            for name, result in results.items():
                value = result["metrics"]
                row = {"Method":name, "Time (ms)":round(value["time_ms"],1), "Entropy (bits)":round(value["entropy"],3), "Contrast index":round(value["contrast"],3), "Edge preservation":round(value["edge"],4)}
                if reference is not None: row.update({"PSNR (dB)":round(value["psnr"],3), "SSIM":round(value["ssim"],5)})
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            if reference is None: st.caption("PSNR and SSIM are omitted because no paired reference is available.")
            st.caption("NIQE: unavailable — no validated implementation is present.")
            ga_result = st.session_state["ga_result"]
            st.write(f"GA selected clip limit {ga_result['clip_limit']:.4f} and tile grid {ga_result['tile_grid_size']} after {ga_result['generations_run']} generation(s).")

with experiments_tab: render_legacy_results()
st.markdown('<div class="footer">Contrast Enhancement Using CLAHE, CNNs and Genetic Algorithm Optimisation · Team project</div>', unsafe_allow_html=True)
