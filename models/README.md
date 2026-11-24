# Model checkpoints

Place locally trained or downloaded Kaggle checkpoints here. Model weights are ignored by Git.

- `models/baseline/best.pt`: raw low-light to normal-light CNN.
- `models/refinement/best.pt`: GA-CLAHE to normal-light refinement CNN.

The Streamlit app validates the checkpoint format and `input_mode`; filenames alone are not trusted.
