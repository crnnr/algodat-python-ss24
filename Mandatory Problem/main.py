import logging
from faker import Faker
import random
import time
import numpy as np
import pandas as pd
from time import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

searchword = "foobabuschubdidududubfoobabuschubdidududub"
three_word_pattern = "Match Multiword Strings!"

random.seed(0)
Faker.seed(0)

def compute_prefix_function(pattern):
    """
    Compute the prefix function for a given pattern.

    Args:
        pattern (str): The pattern for which to compute the prefix function.

    Returns:
        list: The prefix function values for each position in the pattern.

    """
    m = len(pattern)
    lps = [0] * m
    length = 0
    i = 1
    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
    return lps

def KMP(text, pattern):
    """
    Implements the Knuth-Morris-Pratt (KMP) algorithm to find all occurrences of a pattern in a given text.

    Args:
        text (str): The text to search for occurrences of the pattern.
        pattern (str): The pattern to search for in the text.

    Returns:
        list: A list of indices where the pattern occurs in the text. If the pattern is not found, an empty list is returned.
    """
    prefix = compute_prefix_function(pattern)
    result = []
    j = 0
    for i in range(len(text)):
        while j > 0 and text[i] != pattern[j]:
            j = prefix[j - 1]
        if text[i] == pattern[j]:
            j += 1
        if j == len(pattern):
            result.append(i - (j - 1))
            j = prefix[j - 1]
    return result

def create_bad_match_table(pattern):
    """
    Creates a bad match table for the given pattern.

    Args:
        pattern (str): The pattern to create the bad match table for.

    Returns:
        dict: The bad match table, where the keys are characters in the pattern
              and the values are the maximum number of positions to shift the pattern
              when a mismatch occurs.

    """
    table = {}
    pattern_length = len(pattern)
    for i in range(pattern_length):
        table[pattern[i]] = max(1, pattern_length - i - 1)
    return table

def boyer_moore(text, pattern):
    """
    Performs the Boyer-Moore algorithm to search for a pattern in a given text.

    Args:
        text (str): The text to search in.
        pattern (str): The pattern to search for.

    Returns:
        int: The index of the first occurrence of the pattern in the text, or -1 if not found.
    """
    table = create_bad_match_table(pattern)
    pattern_length = len(pattern)
    text_length = len(text)
    i = pattern_length - 1
    while i < text_length:
        j = pattern_length - 1
        while text[i] == pattern[j]:
            if j == 0:
                return i
            i -= 1
            j -= 1
        i += table.get(text[i], pattern_length)
    return -1

def brute_force(text, pattern):
    """
    Searches for the first occurrence of a pattern in a given text using the brute force algorithm.

    Args:
        text (str): The text to search in.
        pattern (str): The pattern to search for.

    Returns:
        int: The index of the first occurrence of the pattern in the text, or -1 if the pattern is not found.
    """
    text_length = len(text)
    pattern_length = len(pattern)
    for i in range(text_length - pattern_length + 1):
        j = 0
        while j < pattern_length and text[i + j] == pattern[j]:
            j += 1
        if j == pattern_length:
            return i
    return -1

def generate_fixed_length_paragraph(length):
    """
    Generates a paragraph of fixed length.

    Parameters:
    length (int): The desired length of the paragraph.

    Returns:
    str: The generated paragraph.

    """
    fake = Faker()
    paragraph = ''
    while len(paragraph) < length:
        sentence = fake.sentence()
        if len(paragraph) + len(sentence) + 1 > length:
            paragraph += sentence[:length - len(paragraph) - 1] + '.'
            break
        paragraph += sentence + ' '
    return paragraph.strip()

def generate_search_text(num_paragraphs, pattern, paragraph_length, insert=True):
    """
    Generate search text with specified number of paragraphs, pattern, and paragraph length.

    Args:
        num_paragraphs (int): The number of paragraphs to generate.
        pattern (str): The pattern to insert into the text.
        paragraph_length (int, optional): The length of each paragraph. Defaults to 1000.
        insert (bool, optional): Whether to insert the pattern into the text. Defaults to True.

    Returns:
        str: The generated search text.
    """
    paragraphs = [generate_fixed_length_paragraph(paragraph_length) for _ in range(num_paragraphs)]
    text = ' '.join(paragraphs)
    if insert:
        words = text.split()
        insert_position = random.randint(0, len(words) - 3)
        words.insert(insert_position, pattern)
        return ' '.join(words)
    return text

def average_timings(function, text, pattern, repetitions):
    """
    Calculates the average, minimum, and maximum time taken by a given function to execute.

    Parameters:
    function (callable): The function to be timed.
    text (str): The text input for the function.
    pattern (str): The pattern input for the function.
    repetitions (int): The number of times to repeat the timing.

    Returns:
    tuple: A tuple containing the average time, minimum time, and maximum time taken by the function.
    """
    timings = [] 
    for _ in range(repetitions):
        start_time = time()
        function(text, pattern)
        elapsed = time() - start_time
        timings.append(elapsed)

    average_time = np.mean(timings)
    min_time = np.min(timings)
    max_time = np.max(timings)
    return average_time, min_time, max_time

if __name__ == "__main__":

    num_paragraphs_list = [10, 50, 100, 500, 1000, 5000, 10000, 50000]
    pattern_lengths = [len(searchword), int(len(searchword)/2), int(len(searchword)/4), int(len(searchword)/8), len(three_word_pattern)]

    repetitions = 100
    paragraph_length = 200

    results = []

    for num_paragraphs in num_paragraphs_list:
        text = generate_search_text(num_paragraphs, three_word_pattern, paragraph_length, insert=True)
        text_length = len(text)
        logging.info(f"Text length: {text_length}")

        
        for pattern_length in pattern_lengths:
            if pattern_length == len(three_word_pattern):
                pattern = three_word_pattern
            else:
                pattern = searchword[:pattern_length]

            kmp_times = average_timings(KMP, text, pattern, repetitions)
            bm_times = average_timings(boyer_moore, text, pattern, repetitions)
            bf_times = average_timings(brute_force, text, pattern, repetitions)

            results.append([text_length, len(pattern), 'KMP', *kmp_times])
            results.append([text_length, len(pattern), 'Boyer-Moore', *bm_times])
            results.append([text_length, len(pattern), 'Brute Force', *bf_times])

    df = pd.DataFrame(results, columns=['Text Length', 'Pattern Length', 'Algorithm', 'Average Time (s)', 'Min Time (s)', 'Max Time (s)', 'Std Dev (s)'])
    df.to_csv('results.csv', index=False)
