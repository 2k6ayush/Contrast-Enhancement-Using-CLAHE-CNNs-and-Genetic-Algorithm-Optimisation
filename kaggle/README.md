# Kaggle CNN handoff

Notebook: `kaggle/cnn_training.ipynb`

Upload or attach three items in Kaggle:

1. Import `cnn_training.ipynb` as the notebook.
2. Create/attach a private Kaggle dataset containing `clahe-cnn-source.zip`.
3. Attach the LOL dataset separately with `Dataset/our485/{low,high}` and `Dataset/eval15/{low,high}`.

Enable a GPU accelerator, then edit `SOURCE_ZIP` and `DATASET_ROOT` in the first code cell to match the paths shown under `/kaggle/input`. The cell extracts the exact uncommitted source into `/kaggle/working/clahe-cnn/source` and copies the packaged `splits/our485_split.json` unchanged to `/kaggle/working/clahe-cnn/splits/our485_split.json`.

Run the setup and validation cells first. Training is opt-in:

- Set only `RUN_BASELINE = True` for raw-low-light baseline training. Leave `RUN_REFINEMENT = False`; no GA preprocessing is then called.
- Set only `RUN_REFINEMENT = True` when deliberately preparing and training the reference-free GA-CLAHE refinement model.
- Leave `RUN_FINAL_TEST = False` during model selection.

All checkpoints, logs, cache files, previews, and final-test output are written below `/kaggle/working/clahe-cnn`. Use **Save Version** and download that directory from notebook output. Full training is not started automatically.
