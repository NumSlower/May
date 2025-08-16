import os
import random
import string
import json
from multiprocessing import Pool, cpu_count
from functools import partial
import time

# Constants
TARGET_VOCAB_SIZE = 1000000
MAX_WORD_LENGTH = 15
MIN_WORD_LENGTH = 2
OUTPUT_FILE = "vocabulary.json"
BATCH_SIZE = 10000
NUM_CORES = cpu_count()

# Basic English components
PREFIXES = ["un", "re", "pre", "dis", "in", "over", "under", "mis", "sub", "super", "anti", "inter"]
ROOTS = [
    "act", "form", "play", "work", "move", "call", "turn", "build", "grow", "think", "speak", "write",
    "read", "run", "jump", "stand", "sit", "walk", "look", "hear", "see", "feel", "touch", "taste"
]
SUFFIXES = [
    "ing", "ed", "er", "est", "ly", "ion", "ment", "ness", "able", "ible", "al", "ful", "less", "y"
]

VOWELS = set("aeiou")
CONSONANTS = set(string.ascii_lowercase) - VOWELS

# Seed words
SEED_WORDS = {
    "happy", "sad", "love", "hate", "good", "bad", "big", "small", "run", "walk", "talk", "listen",
    "see", "hear", "eat", "drink", "sleep", "wake", "think", "know", "learn", "teach", "write", "read"
}

# Precompute valid combinations
PRECOMPUTED_WORDS = set()
for prefix in [""] + PREFIXES:
    for root in ROOTS:
        for suffix in [""] + SUFFIXES:
            word = prefix + root + suffix
            if MIN_WORD_LENGTH <= len(word) <= MAX_WORD_LENGTH and any(v in word for v in VOWELS):
                PRECOMPUTED_WORDS.add(word)

# Fast English-like check
def is_english_like(word):
    length = len(word)
    if length < MIN_WORD_LENGTH or length > MAX_WORD_LENGTH:
        return False
    has_vowel = any(char in VOWELS for char in word)
    if not has_vowel:
        return False
    consonant_run = 0
    for char in word:
        if char in CONSONANTS:
            consonant_run += 1
            if consonant_run > 4:
                return False
        else:
            consonant_run = 0
    return True

# Load base words
def load_base_words(filename="/usr/share/dict/words"):
    base_words = set()
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            base_words = {line.strip().lower() for line in f if is_english_like(line.strip().lower())}
    return base_words

# Generate synthetic word (optimized)
def generate_synthetic_word(length=None):
    if length is None:
        length = random.randint(MIN_WORD_LENGTH, MAX_WORD_LENGTH)
    chars = []
    for i in range(length):
        if i > 0 and chars[-1] in CONSONANTS and random.random() < 0.4:
            chars.append(random.choice("aeiou"))
        else:
            chars.append(random.choice(string.ascii_lowercase))
    word = "".join(chars)
    return word if is_english_like(word) else generate_synthetic_word(length)

# Generate related words (optimized)
def generate_related_words(word):
    if word in SEED_WORDS:
        synonym_map = {"happy": "joy", "sad": "unhappy", "love": "like", "hate": "dislike", "good": "great"}
        antonym_map = {"happy": "sad", "sad": "happy", "love": "hate", "hate": "love", "good": "bad"}
        synonym = synonym_map.get(word, word + "ly")
        antonym = antonym_map.get(word, "un" + word)
    else:
        synonym = word + "ing" if not word.endswith("ing") else word[:-3] + "ed"
        antonym = "un" + word if not word.startswith("un") else word[2:]
    return synonym if is_english_like(synonym) else "", antonym if is_english_like(antonym) else ""

# Worker function for multiprocessing
def generate_vocab_batch(size, seed_words, precomputed_words):
    """Generate a batch of vocabulary entries"""
    random.seed(os.urandom(16))  # Unique seed per process
    batch = set()
    while len(batch) < size:
        if random.random() < 0.7 and precomputed_words:
            word = random.choice(list(precomputed_words))
        else:
            word = generate_synthetic_word()
        if word not in seed_words:
            batch.add(word)
    
    vocab_batch = []
    for word in batch:
        synonym, antonym = generate_related_words(word)
        sentiment = random.randint(-1, 1)
        vocab_batch.append({
            "word": word,
            "synonym": synonym,
            "antonym": antonym,
            "sentiment_score": sentiment
        })
    return vocab_batch

# Main vocabulary generation
def generate_vocabulary():
    start_time = time.time()
    
    # Load base words
    base_words = load_base_words()
    if not base_words:
        print("No base word list found; using seed words.")
        base_words = SEED_WORDS.copy()
    print(f"Loaded {len(base_words)} base words.")

    # Calculate remaining words needed
    remaining = TARGET_VOCAB_SIZE - len(base_words)
    batches_needed = (remaining + BATCH_SIZE - 1) // BATCH_SIZE
    
    # Use multiprocessing to generate batches
    with Pool(NUM_CORES) as pool:
        batch_func = partial(generate_vocab_batch, seed_words=base_words, precomputed_words=PRECOMPUTED_WORDS)
        results = pool.map(batch_func, [BATCH_SIZE] * batches_needed)
    
    # Combine results
    vocab_list = []
    seen_words = set(base_words)
    for batch in results:
        for entry in batch:
            if entry["word"] not in seen_words:
                vocab_list.append(entry)
                seen_words.add(entry["word"])
    
    # Add base words
    for word in base_words:
        synonym, antonym = generate_related_words(word)
        vocab_list.append({
            "word": word,
            "synonym": synonym,
            "antonym": antonym,
            "sentiment_score": random.randint(-1, 1)
        })
    
    # Trim to exact size
    vocab_list = vocab_list[:TARGET_VOCAB_SIZE]
    
    print(f"Generated {len(vocab_list)} words in {time.time() - start_time:.2f} seconds.")
    return vocab_list

# Save vocabulary
def save_vocabulary(vocab_list):
    with open(OUTPUT_FILE, 'w') as f:
        json.dump({"vocabulary": vocab_list}, f, separators=(',', ':'))
    file_size = os.path.getsize(OUTPUT_FILE) / 1024
    print(f"Vocabulary saved to {OUTPUT_FILE} ({file_size:.2f} KB)")

if __name__ == "__main__":
    print("Generating vocabulary with 1,000,000 English words...")
    vocab = generate_vocabulary()
    save_vocabulary(vocab)
    print(f"Generated and saved {len(vocab)} words.")