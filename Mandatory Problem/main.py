from faker import Faker
import random

searchword = "foobabuschubdidududub"

def compute_prefix_function(pattern):
    prefix = [0]
    j = 0
    for i in range(1, len(pattern)):
        while j > 0 and pattern[j] != pattern[i]:
            j = prefix[j - 1]
        if pattern[j] == pattern[i]:
            j += 1
        prefix.append(j)
    return prefix

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

if __name__ == "__main__":
    generate_lorem_ipsum(9000000) #Benchmarking lol
    with open('input.txt', 'r') as file:
        text = file.read().strip()

    print(KMP(text, searchword))
    print(boyer_moore(text, searchword))
    print(brute_force(text, searchword))