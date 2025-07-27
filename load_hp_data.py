import re

def load_data(data_file, device='cpu'):
    with open(data_file, 'r', encoding='utf-8') as f:
        text = f.read()

    print("length of raw dataset in characters: ", len(text))

    # Unique characters in the raw dataset
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print("Characters present in the raw dataset: ", ''.join(chars))
    print("Vocabulary size of the raw dataset: ", vocab_size)
    print("\n")

    # text = text.lower() # Convert all text to lowercase

    # --- Augmentations ---

    # Remove certain characters
    # Remove digits
    text = re.sub(r'[0-9]', '', text)
    # Remove ( and ) characters
    text = text.replace('(', '').replace(')', '')
    # Remove * characters
    text = text.replace('*', '')


    # Standardize common punctuation
    text = text.replace('–', '-')
    text = text.replace('—', '-')
    text = text.replace('‘', "'").replace('’', "'")
    text = text.replace('“', '"').replace('”', '"')
    # text = text.replace('…', '...') # Ellipsis character to three periods


    # Remove extra whitespace
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'[^\S\n]+', ' ', text)
    text = text.strip()

    # --- End Augmentations ---

    print("length of dataset after augmentation in characters: ", len(text))

    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print("Characters present in the augmented dataset: ", ''.join(chars))
    print("Vocabulary size of the augmented dataset: ", vocab_size)
    print("\n")

    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }

    # create a mapping from characters to integers and vice versa
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    return text, vocab_size, encode, decode