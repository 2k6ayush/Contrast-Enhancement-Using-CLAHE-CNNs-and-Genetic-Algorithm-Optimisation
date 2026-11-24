# Result provenance

The CSVs and figures originally committed directly in this directory were produced before the Member-2 CNN package and checkpoint format were integrated. They do not record a validated trained checkpoint, architecture metadata, preprocessing metadata, or whether a compatibility fallback was active. They are preserved unchanged as **legacy/fallback artifacts** and must not be presented as verified trained-CNN results.

Run `python -m cnn.evaluate` with explicit baseline and refinement checkpoints for current measurements. New evaluation output goes to `results/cnn_evaluation/` and records per-image metrics, aggregates, checkpoint paths and SHA-256 hashes, preprocessing, loading time, inference time, device, and unavailable NIQE status. That generated directory is ignored by Git because its numbers depend on separately trained weights.

`results/final_eval15/` is reserved for measured, provenance-rich final-test CSV/JSON artifacts. It is not present in this integration, so the repository makes no claim that a current-checkpoint final evaluation has been saved. Existing `summary_eval15.csv` and other top-level result files remain labeled as legacy artifacts; they are not relabeled as results from the newly trained checkpoints.
