import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformer import Transformer  # Assuming transformer.py contains the Transformer class

# Constants for special tokens
START_TOKEN = ""
END_TOKEN = ""
PADDING_TOKEN = ""

# Define Urdu vocabulary
urdu_vocabulary = list(dict.fromkeys([
    START_TOKEN, 
    ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', '<', '=', '>', '?', '،', '؛', '۔', 'ء', 'ٓ',
    'ا', 'آ', 'ب', 'پ', 'ت', 'ٹ', 'ث', 'ج', 'چ', 'ح', 'خ',
    'د', 'ڈ', 'ذ', 'ر', 'ڑ', 'ز', 'ژ', 'س', 'ش', 'ص', 'ض',
    'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ک', 'گ', 'ل', 'م',
    'ن', 'ں', 'و', 'ہ', 'ھ', 'ی', 'ے',
    'ً', 'ٌ', 'ٍ', 'َ', 'ُ', 'ِ', 'ّ', 'ْ',
    PADDING_TOKEN, END_TOKEN
]))

# Define English vocabulary
english_vocabulary = list(dict.fromkeys([
    START_TOKEN,  
    ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    ':', '<', '=', '>', '?', '@',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
    'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
    'Y', 'Z',
    '[', '\\', ']', '^', '_', '`',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
    'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
    'y', 'z',
    '{', '|', '}', '~', PADDING_TOKEN, END_TOKEN
]))

# Create dictionaries for token to index conversion
english_to_index = {char: idx for idx, char in enumerate(english_vocabulary)}
urdu_to_index = {char: idx for idx, char in enumerate(urdu_vocabulary)}
index_to_urdu = {idx: char for char, idx in urdu_to_index.items()}

# Load and preprocess dataset
translation_file = '/content/english_to_urdu_dataset.xlsx'
df = pd.read_excel(translation_file)
TOTAL_SENTENCES = 10000
df = df[:TOTAL_SENTENCES]

# Extract and clean sentences
english_sentences = df.iloc[:, 0].astype(str).str.strip().tolist()
urdu_sentences = df.iloc[:, 1].astype(str).str.strip().tolist()

max_sequence_length = 200

def valid_tokens(sentence, vocab):
    """Check if all tokens in a sentence are in the vocabulary"""
    for token in list(set(sentence)):
        if token not in vocab:
            return False
    return True

def valid_length(sentence, max_length):
    """Check if sentence length is within max_length"""
    return len(list(sentence)) <= max_length

# Filter valid sentences
valid_sentence_indicies = []
for index in range(len(urdu_sentences)):
    urdu_sentence, english_sentence = urdu_sentences[index], english_sentences[index]
    if (valid_length(urdu_sentence, max_sequence_length) and 
        valid_length(english_sentence, max_sequence_length) and 
        valid_tokens(urdu_sentence, urdu_vocabulary)):
        valid_sentence_indicies.append(index)

print(f"Number of sentences: {len(urdu_sentences)}")
print(f"Number of valid sentences: {len(valid_sentence_indicies)}")

# Filter sentences based on valid indices
english_sentences = [english_sentences[i] for i in valid_sentence_indicies]
urdu_sentences = [urdu_sentences[i] for i in valid_sentence_indicies]

# Model hyperparameters
d_model = 512
batch_size = 30
ffn_hidden = 2048
num_heads = 8
drop_prob = 0.1
num_layers = 1
urdu_vocab_size = len(urdu_vocabulary)

# Initialize Transformer model
transformer = Transformer(
    d_model=d_model,
    ffn_hidden=ffn_hidden,
    num_heads=num_heads,
    drop_prob=drop_prob,
    num_layers=num_layers,
    max_sequence_length=max_sequence_length,
    ur_vocab_size=urdu_vocab_size,
    english_to_index=english_to_index,
    urdu_to_index=urdu_to_index,
    START_TOKEN=START_TOKEN,
    END_TOKEN=END_TOKEN,
    PADDING_TOKEN=PADDING_TOKEN
)

# Custom Dataset class
class TextDataset(Dataset):
    def __init__(self, english_sentences, urdu_sentences):
        self.english_sentences = english_sentences
        self.urdu_sentences = urdu_sentences

    def __len__(self):
        return len(self.english_sentences)

    def __getitem__(self, index):
        return self.english_sentences[index], self.urdu_sentences[index]

# Create DataLoader
dataset = TextDataset(english_sentences, urdu_sentences)
dataloader = DataLoader(dataset, batch_size=batch_size)

# Initialize loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=urdu_to_index[PADDING_TOKEN], reduction='none')
optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Initialize model parameters
for params in transformer.parameters():
    if params.dim() > 1:
        nn.init.xavier_uniform_(params)

# Move model to device
transformer.to(device)
transformer.train()

NEG_INFTY = -1e9

def create_masks(eng_batch, ur_batch):
    """Create attention masks for encoder and decoder"""
    num_sentences = len(eng_batch)
    # Create look-ahead mask for decoder self-attention
    look_ahead_mask = torch.full([max_sequence_length, max_sequence_length], True)
    look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1)
    
    # Initialize padding masks
    encoder_padding_mask = torch.full([num_sentences, max_sequence_length, max_sequence_length], False)
    decoder_padding_mask_self_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length], False)
    decoder_padding_mask_cross_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length], False)

    # Apply padding masks for each sentence
    for idx in range(num_sentences):
        eng_sentence_length, ur_sentence_length = len(eng_batch[idx]), len(ur_batch[idx])
        eng_chars_to_padding_mask = np.arange(eng_sentence_length + 1, max_sequence_length)
        ur_chars_to_padding_mask = np.arange(ur_sentence_length + 1, max_sequence_length)
        encoder_padding_mask[idx, :, eng_chars_to_padding_mask] = True
        encoder_padding_mask[idx, eng_chars_to_padding_mask, :] = True
        decoder_padding_mask_self_attention[idx, :, ur_chars_to_padding_mask] = True
        decoder_padding_mask_self_attention[idx, ur_chars_to_padding_mask, :] = True
        decoder_padding_mask_cross_attention[idx, :, eng_chars_to_padding_mask] = True
        decoder_padding_mask_cross_attention[idx, ur_chars_to_padding_mask, :] = True

    # Convert boolean masks to numerical masks
    encoder_self_attention_mask = torch.where(encoder_padding_mask, NEG_INFTY, 0)
    decoder_self_attention_mask = torch.where(look_ahead_mask + decoder_padding_mask_self_attention, NEG_INFTY, 0)
    decoder_cross_attention_mask = torch.where(decoder_padding_mask_cross_attention, NEG_INFTY, 0)
    
    return encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for batch_num, batch in enumerate(dataloader):
        eng_batch, urdu_batch = batch

        # Create masks for the batch
        encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(
            eng_batch, urdu_batch
        )

        # Forward pass
        optimizer.zero_grad()
        ur_predictions = transformer(
            eng_batch,
            urdu_batch,
            encoder_self_attention_mask.to(device),
            decoder_self_attention_mask.to(device),
            decoder_cross_attention_mask.to(device),
            enc_start_token=False,
            enc_end_token=False,
            dec_start_token=True,
            dec_end_token=True
        )

        # Get target labels
        labels = transformer.decoder.sentence_embedding.batch_tokenize(
            urdu_batch,
            start_token=False,
            end_token=True
        )

        # Compute loss
        loss = criterion(
            ur_predictions.view(-1, urdu_vocab_size).to(device),
            labels.view(-1).to(device)
        )

        # Mask out padding tokens for loss calculation
        valid_indices = torch.where(labels.view(-1) == urdu_to_index[PADDING_TOKEN], False, True)
        loss = loss.sum() / valid_indices.sum()

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Print progress every 100 batches
        if batch_num % 100 == 0:
            print(f"Epoch {epoch+1}, Iteration {batch_num} : Loss = {loss.item()}")
            print(f"English: {eng_batch[0]}")
            print(f"Urdu Target: {urdu_batch[0]}")

            # Generate prediction for first sentence in batch
            ur_sentence_predicted = torch.argmax(ur_predictions[0], axis=1)
            predicted_sentence = ""
            for idx in ur_sentence_predicted:
                if idx == urdu_to_index[END_TOKEN]:
                    break
                predicted_sentence += index_to_urdu.get(idx.item(), '')
            print(f"Urdu Prediction: {predicted_sentence}")

            # Evaluation mode for single sentence translation
            transformer.eval()
            eng_sentence = ("should we go to the mall?",)
            ur_sentence = ("",)

            for word_counter in range(max_sequence_length):
                enc_mask, dec_mask, cross_mask = create_masks(eng_sentence, ur_sentence)
                predictions = transformer(
                    eng_sentence,
                    ur_sentence,
                    enc_mask.to(device),
                    dec_mask.to(device),
                    cross_mask.to(device),
                    enc_start_token=False,
                    enc_end_token=False,
                    dec_start_token=True,
                    dec_end_token=False
                )

                next_token_dist = predictions[0][word_counter]
                next_token_idx = torch.argmax(next_token_dist).item()
                next_token = index_to_urdu.get(next_token_idx, '')
                ur_sentence = (ur_sentence[0] + next_token,)

                if next_token == END_TOKEN:
                    break

            print(f"Evaluation translation (should we go to the mall?): {ur_sentence}")
            print("-" * 50)
            transformer.train()