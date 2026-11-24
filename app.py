"""Streamlit studio for low-light image enhancement."""
from __future__ import annotations

import hashlib
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
    st.markdown("**Checkpoint paths**")
    baseline_input = st.text_input("Baseline CNN", "models/baseline/best.pt", help="Raw low-light RGB to enhanced RGB.")
    refinement_input = st.text_input("Refinement CNN", "models/refinement/best.pt", help="GA–CLAHE RGB to refined RGB.")

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

st.sidebar.markdown(f'''<div class="model"><span>Baseline CNN</span><span class="{'ready' if baseline_model is not None else 'missing'}">{'Loaded' if baseline_model is not None else 'Unavailable'}</span></div><div class="model"><span>Refinement CNN</span><span class="{'ready' if refinement_model is not None else 'missing'}">{'Loaded' if refinement_model is not None else 'Unavailable'}</span></div>''', unsafe_allow_html=True)
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
        low_dir, high_dir = ROOT / "Dataset" / "eval15" / "low", ROOT / "Dataset" / "eval15" / "high"
        samples = sorted(p.name for p in low_dir.iterdir() if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in {".png", ".jpg", ".jpeg"}) if low_dir.is_dir() else []
        if not samples:
            st.info("Dataset samples are unavailable in this checkout. Choose “Upload image” to continue with your own image.")
        else:
            image_name = st.selectbox("Dataset sample", samples)
            try:
                image = load_rgb_image(low_dir / image_name)
                if (high_dir / image_name).is_file(): reference = load_rgb_image(high_dir / image_name)
                else: st.warning("This sample has no paired reference; PSNR and SSIM are unavailable.")
            except Exception as exc: input_error = str(exc)
    if input_error: st.error(f"Image loading failed: {input_error}")
    if image is not None:
        if reference is not None and reference.shape != image.shape:
            input_error = "reference dimensions do not match"
            st.error("The paired reference dimensions do not match the input. Remove or replace it before enhancing.")
        preview, details = st.columns([2, 1])
        with preview: st.image(image, caption="Input preview", width="stretch")
        with details:
            st.markdown(f'<div class="meta"><strong>{image_name}</strong><br>{image.shape[1]} × {image.shape[0]} pixels<br>RGB, uint8</div>', unsafe_allow_html=True)
            if reference is not None and not input_error: st.success("Paired reference ready for PSNR and SSIM.")
            else: st.caption("No paired reference. PSNR and SSIM will not be shown.")
    else: st.caption("Upload a low-light image to activate enhancement.")

    signature = hashlib.sha256(b"".join([
        image.tobytes() if image is not None else b"",
        reference.tobytes() if reference is not None else b"",
        repr((image_name, ga_settings, path_token(baseline_path), path_token(refinement_path))).encode(),
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
