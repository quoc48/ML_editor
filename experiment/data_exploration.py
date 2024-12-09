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

df = pd.read_csv(Path('data/writers.csv'))

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

df = df[df["PostTypeId"].isin([1,2])]

# Display columns and counts of non-null entries
print("Displaying DataFrame info:")

questions_with_accepted_answers = df[df["is_question"] & ~(df["AcceptedAnswerId"].isna())]
q_and_a = questions_with_accepted_answers.join(df[["body_text"]], on="AcceptedAnswerId", how="left", rsuffix="_answer")

# Setting this option allows us to display all the data
pd.options.display.max_colwidth = 500
q_and_a[["body_text", "body_text_answer"]][:3]

print(q_and_a[["body_text", "body_text_answer"]][:3])

has_accepted_answer = df[df["is_question"] & ~(df["AcceptedAnswerId"].isna())]
received_answers = df[df["is_question"] & (df["AnswerCount"]!=0)]
no_answers = df[df["is_question"] & (df["AcceptedAnswerId"].isna()) & (df["AnswerCount"]==0)]

print("%s total questions \n %s  received at least one answer \n %s received an accepted answer" % (
    len(df[df["is_question"]]),
    len(received_answers),
    len(has_accepted_answer)))
# df.info()

# Create the figure
fig = plt.figure(figsize=(16,10))
fig.suptitle("Distribution of question scores")
plt.xlabel("Question scores")
plt.ylabel("Number of questions")

# Plot the histogram
df["Score"].hist(bins=200)

# Show the plot
plt.show()