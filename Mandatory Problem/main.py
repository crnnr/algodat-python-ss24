import logging
from faker import Faker
import random
import time
import csv
import datetime
import numpy as np
import pandas as pd
from time import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

searchword = "foobabuschubdidududubfoobabuschubdidududub"
three_word_pattern = "Match Multiword Strings!"

random.seed(0)
Faker.seed(0)

def compute_prefix_function(pattern):
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
    table = {}
    pattern_length = len(pattern)
    for i in range(pattern_length):
        table[pattern[i]] = max(1, pattern_length - i - 1)
    return table

def boyer_moore(text, pattern):
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
    text_length = len(text)
    pattern_length = len(pattern)
    for i in range(text_length - pattern_length + 1):
        j = 0
        while j < pattern_length and text[i + j] == pattern[j]:
            j += 1
        if j == pattern_length:
            return i
    return -1

def generate_search_text(num_paragraphs, pattern, insert=True):
    fake = Faker()
    paragraphs = [fake.paragraph() for _ in range(num_paragraphs)]
    text = ' '.join(paragraphs)
    if insert:
        words = text.split()
        insert_position = random.randint(0, len(words) - 3)
        words.insert(insert_position, pattern)
        return ' '.join(words)
    return text

def average_timings(function, text, pattern, repetitoins=100):
    timings = []
    #Return the time in seconds since the January 1, 1970, 00:00:00, GMT.
    start_time = time()
    for _ in range(repetitoins):
        function(text, pattern)
    elapsed = time() - start_time
    timings.append(elapsed)
    average_time = np.mean(timings)
    min_time = np.min(timings)
    max_time = np.max(timings)
    std_dev = np.std(timings)
    return average_time, min_time, max_time, std_dev

if __name__ == "__main__":
    num_paragraphs_list = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000]
    pattern_lengths = [len(searchword), int(len(searchword)/2), int(len(searchword)/4), int(len(searchword)/8), len(three_word_pattern)]

    repetitions = 10

    results = []

    for num_paragraphs in num_paragraphs_list:
        text = generate_search_text(num_paragraphs, three_word_pattern, insert=True)
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
