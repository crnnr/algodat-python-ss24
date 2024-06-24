from faker import Faker
import textwrap

def generate_fixed_length_paragraph(length):
    fake = Faker()
    paragraph = ''
    while len(paragraph) < length:
        sentence = fake.sentence()
        if len(paragraph) + len(sentence) + 1 > length:
            paragraph += sentence[:length - len(paragraph) - 1] + '.'
            break
        paragraph += sentence + ' '
    return paragraph.strip()

if __name__ == "__main__":
    fixed_length_paragraph = generate_fixed_length_paragraph(1000)
    print(fixed_length_paragraph)
    print("Length of paragraph:", len(fixed_length_paragraph))
