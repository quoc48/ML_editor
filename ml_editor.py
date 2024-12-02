import argparse
import logging
import sys

import pyphen
import nltk

pyphen.language_fallback("en_US")

logger = logging.getLogger()
logger.setLevel(logging.INFO)
console_out = logging.StreamHandler(sys.stdout)
console_out.setLevel(logging.DEBUG)

def parse_argument():
    """
    Simple argument parser for the command line
    :return: The text to be edited
    """
    parser = argparse.ArgumentParser(
        description="Receive text to be edited")
    parser.add_argument("text", metavar="input text", type=str)
    args = parser.parse_args()
    return args.text


def clean_input(text):
    """
    Text sanitization function
    :param text: User input text
    :return: Santitized text, without non ascii characters
    """
    # To keep things simple at the start, let's only keep ASCII characters
    return str(text.encode().decode('ascii', errors='ignore'))


def preprocess_input(text):
    """
    Tokenize text that has been sainitized
    :param text: Satinized text
    :return: Text ready to be fed to analysis, by having sentences and 
    words tokenized
    """
    sentences = nltk.sent_tokenize(text)
    tokens = [nltk.word_tokenize(sentence) for sentence in sentences]
    return tokens

def get_suggestions(sentence_list):
    """
    Returns a string containing out suggestions
    :param sentence_list: a list of sentences, each being a list of words
    :return: suggestions to improve the input
    """
    told_said_usage = sum(
        (count_word_usage(tokens, ["told", "said"]) for tokens in sentence_list)
    )
    but_and_usage = sum(
        (count_word_usage(tokens, ["but", "and"]) for tokens in sentence_list)
    )
    wh_adverbs_usage = sum(
        (
            count_word_usage(
                tokens,
                [
                    "when",
                    "where",
                    "why",
                    "whence",
                    "whereby",
                    "wherein",
                    "whereupon",
                ],
            )
            for tokens in sentence_list
        )
    )
    result_str = ""
    adverbs_usage = "Adverb usage: %s told/said, %s but/and, %s wh adverbs" % (
        told_said_usage,
        but_and_usage,
        wh_adverbs_usage
    )
    result_str += adverbs_usage
    average_word_length = compute_total_average_word_length(sentence_list)
    unique_words_fraction = compute_total_unique_words_fraction(sentence_list)

    word_stats = "Average word length %.2f, fraction of unique words %.2f" % (
        average_word_length,
        unique_words_fraction,
    )
    # Using HTML break to later display on a webapp
    result_str += "<br/>"
    result_str += word_stats

    number_of_syllables = count_total_syllables(sentence_list)
    number_of_words = count_total_words(sentence_list)
    number_of_sentences = len(sentence_list)

    syllable_counts = "%d syllables, %d words, %d sentences" % (
        number_of_syllables,
        number_of_words,
        number_of_sentences,
    )
    result_str += "<br/>"
    result_str += syllable_counts

    flesch_score = compute_flesch_reading_ease(
        number_of_syllables, number_of_words, number_of_sentences
    )

    flesch = "%d syllables, %.2f flesch score: %s" % (
        number_of_syllables,
        flesch_score,
        get_reading_level_from_flesch(flesch_score),
    )

    result_str += "<br/>"
    result_str += flesch

    return result_str

def get_recommendations_from_input(txt):
    """
    Cleans, preprocesses, and generates heuristic suggestion for input string
    :param txt: Input text
    :return: Suggestions for a given text input
    """
    processed = clean_input(txt)
    tokenized_sentences = preprocess_input(processed)
    suggestions = get_suggestions(tokenized_sentences)
    return suggestions


if __name__ == "__main__":
    input_text = parse_argument()
    print(get_recommendations_from_input(input_text))