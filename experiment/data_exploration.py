import json
from tqdm import tqdm
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ElT
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import os

from pathlib import Path
import sys
sys.path.append("..")
import warnings
warnings.filterwarnings('ignore')

print("Current working directory:", os.getcwd())

df = pd.read_csv(Path('data/writers.csv'))

print("Columns in the DataFrame:", df.columns)

# Start by changing types to ake precessing easier
df["AnswerCount"] = df["AnswerCount"].fillna(-1)
df["AnswerCount"] = df["AnswerCount"].astype(int)
df["PostTypeId"] = df["PostTypeId"].astype(int)
df["Id"] = df["Id"].astype(int)
df.set_index("Id", inplace=True, drop=False)

# Add measure of the length of a post
df["full_text"] = df["Title"].str.cat(df["body_text"], sep=" ", na_rep="")
df["text_len"] = df["full_text"].str.len()

# A question is a post of id 1
df["is_question"] = df["PostTypeId"] == 1

# Display columns and counts of non-null entries
print("Displaying DataFrame info:")
df.info()

# Additional checks
print(f"Number of rows: {len(df)}")
print("First 5 rows of the DataFrame:")
print(df.head())

print("Data exploration completed successfully.")