# This is the module that defines the chatbot class and methods
import tensorflow as tf
import os
from transformers import AutoTokenizer
from transformers import TFAutoModelForCausalLM
from transformers import TFGPT2LMHeadModel

model = TFGPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")


class Chatbot:
    """This class defines a chatbot model based on GPT-2 and TensorFlow."""

    def __init__(self):
        """Initialize the chatbot with a tokenizer and a model."""
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = TFAutoModelForCausalLM.from_pretrained("gpt2")

    def train(self, input_order, target_word):
        """Train the chatbot model on a given set of input and target sentences."""
        encoded_inputs = self.tokenizer.batch_encode_plus(input_order, padding=True, truncation=True,
                                                          return_tensors="tf", return_dict=True)
        encoded_targets = self.tokenizer.batch_encode_plus(target_word, padding=True, truncation=True,
                                                           return_tensors="tf", return_dict=True)
        target_ids = encoded_targets.input_ids
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy()
        self.model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
        dataset = tf.data.from_tensor_slices((encoded_inputs.input_ids, target_ids)).batch(32).shuffle(1000)
        # Split the dataset into train and validation sets
        train_size = int(0.8 * len(dataset))
        val_size = int(0.2 * len(dataset))
        train_dataset = dataset.take(train_size)
        val_dataset = dataset.skip(train_size).take(val_size)
        # Create a checkpoint callback to save the best model
        checkpoint_path = "best_model.h5"
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor="val_loss",
                                                                 save_best_only=True)
        # Train the model on the train dataset and validate on the val dataset
        self.model.fit(train_dataset, validation_data=val_dataset, callbacks=[checkpoint_callback])

    def generate_response(self, input_order):
        """Generate a response for a given input sentence."""
        # Check if the input sentence is valid
        if not input_order or not isinstance(input_order, str):
            return "Invalid input."
        # Load the best model from the checkpoint
        self.model.load_weights("best_model.h5")
        # Encode the input sentence and generate an output sentence
        input_ids = self.tokenizer.encode(input_order, return_tensors="tf")
        output_ids = self.model.generate(input_ids, max_length=50, num_beams=3,
                                         eos_token_id=self.tokenizer.eos_token_id)
        output_sentence = self.tokenizer.decode(output_ids[0])
        return output_sentence


chatbot = Chatbot()
input_sentences = []
target_sentences = []
# Check if the input file exists
if os.path.exists("dialogue.txt"):
    with open("dialogue.txt", "r") as f:
        for line in f:
            input_sentence, target_sentence = line.split("\t")
            input_sentences.append(input_sentence)
            target_sentences.append(target_sentence)
    chatbot.train(input_sentences, target_sentences)
else:
    print("Input file not found.")
