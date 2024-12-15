import pandas as pd
import spacy
import umap
import numpy as np 
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
sys.path.append("..")
import warnings
warnings.filterwarnings('ignore')

# Add the project root (ML_EDITOR) to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from ml_editor.data_processing import format_raw_df, get_split_by_author
from ml_editor.data_visualization import plot_embeddings

data_path = Path('data/writers.csv')
df = pd.read_csv(data_path)
df = format_raw_df(df.copy())

train_author, test_author = get_split_by_author(df[df["is_question"]])
questions = train_author[train_author["is_question"]]
raw_text = questions["body_text"]
# Extract a label to use as a color on our plots. 
# This label does not need to be the same label as the one for the classifier.
sent_labels = questions["AcceptedAnswerId"].notna()

print(sent_labels.value_counts())

# Create an instance of a tfidf vectorizer, 
# We could use CountVectorizer for a non normalized version
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=2**21)

# Fit our vectorizer to questions in our dataset
# Returns an array of vectorized text
bag_of_words = vectorizer.fit_transform(raw_text)

print(bag_of_words.shape)

umap_embedder = umap.UMAP()
umap_bow = umap_embedder.fit_transform(bag_of_words)
plot_embeddings(umap_bow, sent_labels)