import pandas as pd
from pathlib import Path
import sys
sys.path.append("..")
import warnings
warnings.filterwarnings('ignore')

# Add the project root (ML_EDITOR) to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from ml_editor.data_processing import get_vertorized_series

df = pd.read_csv(Path('data/writers.csv'))

df["is_question"] = df["PostTypeId"] == 1
tabular_df = df[df["is_question"]][["Tags", "CommentCount", "CreationDate", "Score"]]
print(tabular_df.head())
 