from keras.utils import to_categorical

import numpy as np
import json

class GetData:
    def __init__(self, sequence_max_len, chars_mapping):
        self.sequence_max_len = sequence_max_len
        self.chars_mapping = chars_mapping

    def get_corpus(self, file_path):
        # We will just use messages from customer service to train our model
        messages_from_customer_service = []        
        with open(file_path) as input_file:
            input_data = json.load(input_file)
        
        for issue in input_data['Issues']:
            for message in issue["Messages"]:
                if not message['IsFromCustomer']:
                    messages_from_customer_service.append(message['Text'].split())

        # Flatten the list
        flattened_messages_from_customer_service = []        
        for message in messages_from_customer_service:
            for word in message:
                flattened_messages_from_customer_service.append(word)

        # Create a text corpus
        text_corpus = " ".join(flattened_messages_from_customer_service)

        return text_corpus

    def get_x_y(self, text):
        # prefix_sequences will be sequence_max_len long and
        # target_character will be the character just after the prefix_sequences 
        prefix_sequences = []
        target_character = []
        for i in range (0, len(text) - self.sequence_max_len):
            prefix_sequences.append(text[i: i + self.sequence_max_len])
            target_character.append(text[i + self.sequence_max_len])

        # Each character is conveted to the specific integer value
        prefix_sequences_encoded = []
        for sequence in prefix_sequences:
            prefix_sequences_encoded.append([self.chars_mapping[char] for char in sequence])
        prefix_sequences_encoded = np.array(prefix_sequences_encoded)

        target_character_encoded = []
        for char in target_character:
            target_character_encoded.append(self.chars_mapping[char])
        target_character_encoded = np.array(target_character_encoded)

        # One-hot encoding: convert the integers to binary class matrix
        X = [to_categorical(x, num_classes=len(self.chars_mapping)) for x in prefix_sequences_encoded]
        X = np.array(X)

        y = to_categorical(target_character_encoded, num_classes=len(self.chars_mapping))

        return X, y
