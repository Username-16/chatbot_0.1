import tf as tf
from tensorflow.python import data
from transformers import AutoTokenizer
# Split the data into input and target sentences
input_sentences = []
target_sentences = []
# noinspection PyTypeChecker
for line in data:
    input_sentence, target_sentence = line.split("\t")
    input_sentences.append(input_sentence)
    target_sentences.append(target_sentence)

# Instantiate the tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Tokenize and encode the sentences
input_ids = tokenizer(input_sentences, padding=True, truncation=True, return_tensors="tf").input_ids
target_ids = tokenizer(target_sentences, padding=True, truncation=True, return_tensors="tf").input_ids

# Split the data into train and test sets
test_input_ids = input_ids[train_size:]
test_target_ids = target_ids[train_size:]
train_input_ids = input_ids[:train_size]
train_target_ids = target_ids[:train_size]

# Create the dataset objects
train_inputs = tf.data.Dataset.from_tensor_slices(train_input_ids)
train_targets = tf.data.Dataset.from_tensor_slices(train_target_ids)
test_inputs = tf.data.Dataset.from_tensor_slices(test_input_ids)
test_targets = tf.data.Dataset.from_tensor_slices(test_target_ids)

# Return the dataset objects
return train_inputs, train_targets, test_inputs, test_targets