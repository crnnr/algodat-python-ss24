from faker import Faker
import random
import time

searchword = "foobabuschubdidududubfoobabuschubdidududub"

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
        while pattern[j] == text[i]:
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

def generate_lorem_ipsum(num_paragraphs):
    fake = Faker()
    paragraphs = [fake.paragraph() for _ in range(num_paragraphs)]
    text = ' '.join(paragraphs)
    words = text.split()
    insert_position = random.randint(0, len(words))
    words.insert(insert_position, searchword)
    with open('input.txt', 'w') as file:
        file.write(' '.join(words))

def average_timings(function, text, pattern, repetitions=100):
    total_time = 0
    for _ in range(repetitions):
        start_time = time.time()
        function(text, pattern)
        total_time += (time.time() - start_time)
    return total_time / repetitions

if __name__ == "__main__":
    num_paragraphs = 50000  # A number that is large but practical.
    generate_lorem_ipsum(num_paragraphs)
    with open('input.txt', 'r') as file:
        text = file.read().strip()

    patterns = [
        searchword,                        # original pattern
        searchword[:int(len(searchword)/2)],  # half the length of the original pattern
        searchword[:int(len(searchword)/4)],  # quarter the length
        # ... add more patterns with varying lengths as needed
    ]

    repetitions = 100  # Number of repetitions for each pattern.

    for pattern in patterns:
        print(f"Searching for pattern of length {len(pattern)}")

        kmp_average_time = average_timings(KMP, text, pattern, repetitions)
        print(f"Average KMP time: {kmp_average_time} seconds")

        bm_average_time = average_timings(boyer_moore, text, pattern, repetitions)
        print(f"Average Boyer-Moore time: {bm_average_time} seconds")

        bf_average_time = average_timings(brute_force, text, pattern, repetitions)
        print(f"Average Brute Force time: {bf_average_time} seconds\n")
