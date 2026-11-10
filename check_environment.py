
import numpy as np
import pandas as pd
import torch
import sklearn
import streamlit as st

print("Environment verification started")

print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("PyTorch:", torch.__version__)
print("Scikit-learn:", sklearn.__version__)
print("Streamlit:", st.__version__)

print("CUDA Available:", torch.cuda.is_available())

print("\nAll libraries imported successfully!")
print("Environment setup completed")
