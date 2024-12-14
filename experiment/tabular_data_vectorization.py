import pandas as pd
from pathlib import Path
import sys
sys.path.append("..")
import warnings
warnings.filterwarnings('ignore')

# Add the project root (ML_EDITOR) to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from ml_editor.data_processing import get_vertorized_series, get_normalized_series

df = pd.read_csv(Path('data/writers.csv'))

df["is_question"] = df["PostTypeId"] == 1
tabular_df = df[df["is_question"]][["Tags", "CommentCount", "CreationDate", "Score"]]

tabular_df["NormComment"] = get_normalized_series(tabular_df, "CommentCount")
tabular_df["NormScore"] = get_normalized_series(tabular_df, "Score")

# Convert our date to a pandas datetime
tabular_df["date"] = pd.to_datetime(tabular_df["CreationDate"])

# Extract meaningful features from the datetime object
tabular_df["year"] = tabular_df["date"].dt.year
tabular_df["month"] = tabular_df["date"].dt.month
tabular_df["day"] = tabular_df["date"].dt.day
tabular_df["hour"] = tabular_df["date"].dt.hour

print(tabular_df.head())