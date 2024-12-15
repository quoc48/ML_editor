import pandas as pd
import spacy
import umap
import numpy as np
from pathlib import Path
import sys
sys.path.append("..")
import warnings
warnings.filterwarnings('ignore')
from ml_editor.data_processing import format_raw_df, get_random_train_test_split, get_vectorized_inputs_and_label, get_split_by_author

data_path = Path('data/writers.csv')
df = pd.read_csv(data_path)
df = format_raw_df(df.copy())