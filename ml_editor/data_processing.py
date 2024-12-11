import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from scipy.sparse import vstack, hstack

def format_raw_df(df):
    """
    Cleanup data and join questions to answers
    :param df: raw DataFrame
    :return: processed DataFrame
    """
    # Fixing types and setting index
    df["PostTypeId"] = df["PostTypeId"].astype(int)
    df["Id"] = df["Id"].astype(int)
    df["AnswerCount"] = df["AnswerCount"].fillna(-1)
    df["AnswerCount"] = df["AnswerCount"].astype(int)
    df["OwnerUserId"].fillna(-1, inplace=True)
    df["OwnerUserId"] = df["OwnerUserId"].astype(int)
    df.set_index("Id", inplace=True, drop=False)

    df["is_question"] = df["PostTypeId"] == 1

    # Filtering out PostTypeIds other than documented ones
    df = df[df["PostTypeId"].isin([1, 2])]

    # Linking questions and answers
    df = df.join(
        df[["Id", "Title", "body_text", "Score", "AcceptedAnswerId"]],
        on="ParentId",
        how="left",
        rsuffix="_question",
    )
    return df

def get_vertorized_series(text_series, vectorizer):
    """
    Vectorizes an input series using a pre-trained vertorizer
    :param text_series: pandas Series of text
    :param vectorizer: pretrained sklearn vectorizer
    :return: array if vectorized feature
    """
    vectors = vectorizer.transform(text_series)
    vectorized_series = [vectors[i] for i in range(vectors.shape[0])]
    return vectorized_series
