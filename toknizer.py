import re
from collections import Counter


def tokenize(text):
    # Convert all text to lowercase
    text = text.lower()

    # Remove all non-alphanumeric characters
    text = re.sub(r'[^a-z0-9 ]', '', text)

    # Split the text into words
    words = text.split()

    # Count the frequency of each word
    word_counts = Counter(words)

    # Create a dictionary that maps each word to a unique integer ID
    vocab = {word: i + 1 for i, (word, count) in enumerate(word_counts.most_common())}

    # Return the tokenized text as a list of integers
    return [vocab[word] for word in words]


tokens = tokenize("your text here")
