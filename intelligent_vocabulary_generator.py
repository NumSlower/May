#!/usr/bin/env python3
"""
Intelligent Vocabulary Generator
Generates comprehensive vocabulary database for AI chatbot training with
semantic relationships, sentiment analysis, and linguistic features.
"""

import os
import json
import random
import string
import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, asdict
from multiprocessing import Pool, cpu_count
from functools import partial
import time
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class VocabularyEntry:
    """Structured vocabulary entry with linguistic metadata"""
    word: str
    word_type: str  # noun, verb, adjective, etc.
    definition: str
    synonym: str
    antonym: str
    sentiment_score: int  # -2 (very negative) to 2 (very positive)
    frequency_score: float  # 0.0 to 1.0
    difficulty_level: int  # 1 (basic) to 5 (advanced)
    category: str  # semantic category
    etymology: str  # word origin/root
    example_sentence: str
    phonetic: str  # pronunciation guide
    syllable_count: int
    length: int
    hash_id: str  # unique identifier

class IntelligentVocabularyGenerator:
    """Advanced vocabulary generator with linguistic intelligence"""
    
    # Configuration constants
    TARGET_VOCAB_SIZE = 100000  # Reduced for better quality
    MIN_WORD_LENGTH = 2
    MAX_WORD_LENGTH = 20
    BATCH_SIZE = 1000
    OUTPUT_DB = "intelligent_vocabulary.db"
    OUTPUT_JSON = "vocabulary_export.json"
    
    # Linguistic components
    VOWELS = set("aeiouAEIOU")
    CONSONANTS = set(string.ascii_letters) - VOWELS
    
    # Morphological components
    PREFIXES = {
        "un": {"meaning": "not, opposite", "type": "negative"},
        "re": {"meaning": "again, back", "type": "repetition"},
        "pre": {"meaning": "before", "type": "time"},
        "dis": {"meaning": "not, apart", "type": "negative"},
        "in": {"meaning": "not, into", "type": "negative"},
        "over": {"meaning": "too much, above", "type": "intensity"},
        "under": {"meaning": "below, insufficient", "type": "position"},
        "mis": {"meaning": "wrongly", "type": "error"},
        "sub": {"meaning": "below, under", "type": "position"},
        "super": {"meaning": "above, beyond", "type": "intensity"},
        "anti": {"meaning": "against", "type": "opposition"},
        "inter": {"meaning": "between", "type": "relation"},
        "multi": {"meaning": "many", "type": "quantity"},
        "semi": {"meaning": "half", "type": "partial"},
        "auto": {"meaning": "self", "type": "reflexive"}
    }
    
    SUFFIXES = {
        "ing": {"meaning": "action/state", "type": "verb_form", "pos": "verb"},
        "ed": {"meaning": "past action", "type": "verb_form", "pos": "verb"},
        "er": {"meaning": "person who", "type": "agent", "pos": "noun"},
        "est": {"meaning": "most", "type": "superlative", "pos": "adjective"},
        "ly": {"meaning": "in manner of", "type": "adverb", "pos": "adverb"},
        "ion": {"meaning": "action/state", "type": "noun_form", "pos": "noun"},
        "ment": {"meaning": "result/state", "type": "noun_form", "pos": "noun"},
        "ness": {"meaning": "quality of", "type": "noun_form", "pos": "noun"},
        "able": {"meaning": "capable of", "type": "ability", "pos": "adjective"},
        "ible": {"meaning": "capable of", "type": "ability", "pos": "adjective"},
        "al": {"meaning": "relating to", "type": "relation", "pos": "adjective"},
        "ful": {"meaning": "full of", "type": "abundance", "pos": "adjective"},
        "less": {"meaning": "without", "type": "absence", "pos": "adjective"},
        "ous": {"meaning": "full of", "type": "characteristic", "pos": "adjective"},
        "ive": {"meaning": "tending to", "type": "tendency", "pos": "adjective"},
        "ize": {"meaning": "to make", "type": "causative", "pos": "verb"}
    }
    
    # Root words with semantic information
    ROOT_WORDS = {
        "act": {"meaning": "do, perform", "category": "action", "pos": "verb"},
        "form": {"meaning": "shape", "category": "physical", "pos": "noun"},
        "play": {"meaning": "engage in activity", "category": "recreation", "pos": "verb"},
        "work": {"meaning": "labor, function", "category": "activity", "pos": "verb"},
        "move": {"meaning": "change position", "category": "motion", "pos": "verb"},
        "think": {"meaning": "use mind", "category": "cognitive", "pos": "verb"},
        "speak": {"meaning": "talk", "category": "communication", "pos": "verb"},
        "write": {"meaning": "record words", "category": "communication", "pos": "verb"},
        "read": {"meaning": "interpret text", "category": "communication", "pos": "verb"},
        "learn": {"meaning": "acquire knowledge", "category": "education", "pos": "verb"},
        "teach": {"meaning": "impart knowledge", "category": "education", "pos": "verb"},
        "love": {"meaning": "deep affection", "category": "emotion", "pos": "verb"},
        "hate": {"meaning": "strong dislike", "category": "emotion", "pos": "verb"},
        "help": {"meaning": "assist", "category": "social", "pos": "verb"},
        "build": {"meaning": "construct", "category": "creation", "pos": "verb"},
        "break": {"meaning": "damage", "category": "destruction", "pos": "verb"},
        "grow": {"meaning": "increase", "category": "development", "pos": "verb"},
        "know": {"meaning": "have knowledge", "category": "cognitive", "pos": "verb"},
        "feel": {"meaning": "experience emotion", "category": "emotion", "pos": "verb"},
        "see": {"meaning": "perceive visually", "category": "perception", "pos": "verb"}
    }
    
    # Semantic categories for organization
    CATEGORIES = {
        "emotion": {"sentiment": 1, "examples": ["happy", "sad", "angry", "peaceful"]},
        "action": {"sentiment": 0, "examples": ["run", "jump", "sit", "stand"]},
        "physical": {"sentiment": 0, "examples": ["big", "small", "round", "square"]},
        "cognitive": {"sentiment": 1, "examples": ["smart", "wise", "confused", "clear"]},
        "social": {"sentiment": 1, "examples": ["friend", "family", "community", "team"]},
        "nature": {"sentiment": 1, "examples": ["tree", "flower", "mountain", "river"]},
        "technology": {"sentiment": 0, "examples": ["computer", "phone", "internet", "software"]},
        "education": {"sentiment": 1, "examples": ["school", "book", "student", "teacher"]},
        "health": {"sentiment": 1, "examples": ["medicine", "doctor", "exercise", "nutrition"]},
        "recreation": {"sentiment": 1, "examples": ["game", "sport", "music", "art"]}
    }
    
    def __init__(self, target_size: int = None):
        """Initialize the vocabulary generator"""
        self.target_size = target_size or self.TARGET_VOCAB_SIZE
        self.db_path = Path(self.OUTPUT_DB)
        self.json_path = Path(self.OUTPUT_JSON)
        self.generated_words: Set[str] = set()
        self.word_patterns: Dict[str, int] = {}
        
    def setup_database(self) -> sqlite3.Connection:
        """Set up SQLite database for vocabulary storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vocabulary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word TEXT UNIQUE NOT NULL,
                word_type TEXT NOT NULL,
                definition TEXT,
                synonym TEXT,
                antonym TEXT,
                sentiment_score INTEGER,
                frequency_score REAL,
                difficulty_level INTEGER,
                category TEXT,
                etymology TEXT,
                example_sentence TEXT,
                phonetic TEXT,
                syllable_count INTEGER,
                length INTEGER,
                hash_id TEXT UNIQUE,
                created_timestamp REAL
            )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_word ON vocabulary(word)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_category ON vocabulary(category)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_difficulty ON vocabulary(difficulty_level)')
        
        conn.commit()
        return conn
        
    def generate_hash_id(self, word: str) -> str:
        """Generate unique hash ID for word"""
        return hashlib.md5(f"{word}_{time.time()}".encode()).hexdigest()[:12]
        
    def count_syllables(self, word: str) -> int:
        """Estimate syllable count using vowel clusters"""
        word = word.lower()
        vowel_groups = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in "aeiou"
            if is_vowel and not previous_was_vowel:
                vowel_groups += 1
            previous_was_vowel = is_vowel
            
        # Handle silent 'e'
        if word.endswith('e') and vowel_groups > 1:
            vowel_groups -= 1
            
        return max(1, vowel_groups)
        
    def generate_phonetic(self, word: str) -> str:
        """Generate simple phonetic representation"""
        # Simple phonetic mapping (basic version)
        phonetic_map = {
            'ph': 'f', 'gh': 'f', 'ck': 'k', 'qu': 'kw',
            'ch': 'tʃ', 'sh': 'ʃ', 'th': 'θ', 'ng': 'ŋ'
        }
        
        phonetic = word.lower()
        for pattern, replacement in phonetic_map.items():
            phonetic = phonetic.replace(pattern, replacement)
            
        return f"/{phonetic}/"
        
    def is_valid_english_pattern(self, word: str) -> bool:
        """Check if word follows English phonological patterns"""
        word = word.lower()
        
        # Must have at least one vowel
        if not any(v in word for v in "aeiou"):
            return False
            
        # Check for excessive consonant clusters
        consonant_run = 0
        for char in word:
            if char in "bcdfghjklmnpqrstvwxyz":
                consonant_run += 1
                if consonant_run > 4:  # Max 4 consecutive consonants
                    return False
            else:
                consonant_run = 0
                
        # Check for common invalid patterns
        invalid_patterns = ['qq', 'xx', 'jj', 'vv', 'ww', 'yy']
        if any(pattern in word for pattern in invalid_patterns):
            return False
            
        return True
        
    def create_word_from_morphemes(self, prefix: str = "", root: str = "", suffix: str = "") -> Optional[VocabularyEntry]:
        """Create vocabulary entry from morphological components"""
        word = prefix + root + suffix
        
        if not self.is_valid_english_pattern(word) or word in self.generated_words:
            return None
            
        if not (self.MIN_WORD_LENGTH <= len(word) <= self.MAX_WORD_LENGTH):
            return None
            
        # Determine word type based on suffix
        word_type = "noun"  # default
        if suffix in self.SUFFIXES:
            word_type = self.SUFFIXES[suffix].get("pos", "noun")
        elif root in self.ROOT_WORDS:
            word_type = self.ROOT_WORDS[root]["pos"]
            
        # Build definition
        definition_parts = []
        if prefix in self.PREFIXES:
            definition_parts.append(self.PREFIXES[prefix]["meaning"])
        if root in self.ROOT_WORDS:
            definition_parts.append(self.ROOT_WORDS[root]["meaning"])
        if suffix in self.SUFFIXES:
            definition_parts.append(self.SUFFIXES[suffix]["meaning"])
            
        definition = "; ".join(definition_parts) if definition_parts else f"Related to {root}"
        
        # Determine category and sentiment
        category = "general"
        sentiment = 0
        if root in self.ROOT_WORDS:
            category = self.ROOT_WORDS[root]["category"]
            if category in self.CATEGORIES:
                sentiment = self.CATEGORIES[category]["sentiment"]
                
        # Adjust sentiment based on prefix
        if prefix in ["un", "dis", "mis"]:
            sentiment = max(-2, sentiment - 1)
        elif prefix in ["super", "over"]:
            sentiment = min(2, sentiment + 1)
            
        # Generate related words
        synonym = self.generate_synonym(word, root, category)
        antonym = self.generate_antonym(word, prefix, root, sentiment)
        
        # Calculate difficulty and frequency
        difficulty = self.calculate_difficulty(word, prefix, suffix)
        frequency = self.calculate_frequency(word, category)
        
        # Generate example sentence
        example = self.generate_example_sentence(word, word_type, category)
        
        # Create entry
        entry = VocabularyEntry(
            word=word,
            word_type=word_type,
            definition=definition,
            synonym=synonym,
            antonym=antonym,
            sentiment_score=sentiment,
            frequency_score=frequency,
            difficulty_level=difficulty,
            category=category,
            etymology=f"{prefix}+{root}+{suffix}".strip("+"),
            example_sentence=example,
            phonetic=self.generate_phonetic(word),
            syllable_count=self.count_syllables(word),
            length=len(word),
            hash_id=self.generate_hash_id(word)
        )
        
        self.generated_words.add(word)
        return entry
        
    def generate_synonym(self, word: str, root: str, category: str) -> str:
        """Generate plausible synonym"""
        synonyms = {
            "action": ["do", "perform", "execute", "carry out"],
            "emotion": ["feel", "experience", "sense", "undergo"],
            "cognitive": ["think", "understand", "comprehend", "realize"],
            "physical": ["form", "shape", "structure", "build"],
            "social": ["connect", "relate", "interact", "engage"]
        }
        
        if category in synonyms:
            return random.choice(synonyms[category])
        
        # Generate based on root transformation
        if root and len(root) > 3:
            variations = [
                root + "ing",
                root + "ed", 
                root + "er",
                "re" + root
            ]
            return random.choice([v for v in variations if self.is_valid_english_pattern(v)])
            
        return word + "ly" if not word.endswith("ly") else word[:-2]
        
    def generate_antonym(self, word: str, prefix: str, root: str, sentiment: int) -> str:
        """Generate plausible antonym"""
        # If word already has negative prefix, remove it
        if prefix in ["un", "dis", "in", "mis"]:
            return root
            
        # Add negative prefix
        negative_prefixes = ["un", "dis", "non"]
        chosen_prefix = random.choice(negative_prefixes)
        
        antonym = chosen_prefix + word
        if self.is_valid_english_pattern(antonym):
            return antonym
            
        # Fallback to semantic opposites
        opposites = {
            "good": "bad", "big": "small", "hot": "cold", "fast": "slow",
            "happy": "sad", "love": "hate", "up": "down", "in": "out"
        }
        
        return opposites.get(word, "opposite_" + word)
        
    def calculate_difficulty(self, word: str, prefix: str, suffix: str) -> int:
        """Calculate difficulty level 1-5"""
        base_difficulty = 1
        
        # Length factor
        if len(word) > 8:
            base_difficulty += 1
        if len(word) > 12:
            base_difficulty += 1
            
        # Morphological complexity
        if prefix:
            base_difficulty += 1
        if suffix in ["tion", "sion", "ment", "ness"]:
            base_difficulty += 1
            
        # Syllable count
        syllables = self.count_syllables(word)
        if syllables > 3:
            base_difficulty += 1
            
        return min(5, base_difficulty)
        
    def calculate_frequency(self, word: str, category: str) -> float:
        """Calculate frequency score 0.0-1.0"""
        # Base frequency by category
        base_frequencies = {
            "emotion": 0.8, "action": 0.9, "physical": 0.7,
            "cognitive": 0.6, "social": 0.8, "general": 0.5
        }
        
        base_freq = base_frequencies.get(category, 0.5)
        
        # Adjust by word length (shorter words more frequent)
        length_factor = max(0.1, 1.0 - (len(word) - 4) * 0.05)
        
        return min(1.0, base_freq * length_factor)
        
    def generate_example_sentence(self, word: str, word_type: str, category: str) -> str:
        """Generate contextual example sentence"""
        templates = {
            "noun": [
                f"The {word} was very important.",
                f"She found a beautiful {word}.",
                f"Every {word} has its place."
            ],
            "verb": [
                f"They {word} every day.",
                f"I like to {word} in the morning.",
                f"We should {word} more often."
            ],
            "adjective": [
                f"The house looks very {word}.",
                f"This solution is quite {word}.",
                f"Her approach was {word}."
            ],
            "adverb": [
                f"He spoke {word} to everyone.",
                f"The work was done {word}.",
                f"She moved {word} through the crowd."
            ]
        }
        
        if word_type in templates:
            return random.choice(templates[word_type])
        
        return f"The concept of {word} is important to understand."
        
    def generate_vocabulary_batch(self, batch_size: int) -> List[VocabularyEntry]:
        """Generate a batch of vocabulary entries"""
        batch = []
        attempts = 0
        max_attempts = batch_size * 10
        
        while len(batch) < batch_size and attempts < max_attempts:
            attempts += 1
            
            # Choose generation method
            if random.random() < 0.7:
                # Morphological generation
                prefix = random.choice([""] + list(self.PREFIXES.keys())) if random.random() < 0.3 else ""
                root = random.choice(list(self.ROOT_WORDS.keys()))
                suffix = random.choice([""] + list(self.SUFFIXES.keys())) if random.random() < 0.4 else ""
                
                entry = self.create_word_from_morphemes(prefix, root, suffix)
                if entry:
                    batch.append(entry)
            else:
                # Pure synthetic generation
                entry = self.generate_synthetic_word()
                if entry:
                    batch.append(entry)
                    
        return batch
        
    def generate_synthetic_word(self) -> Optional[VocabularyEntry]:
        """Generate completely synthetic but English-like word"""
        # Common English patterns
        patterns = [
            "CVCV", "CVC", "CVCVC", "CVCC", "CCVC", "CVCVV", "VCVC"
        ]
        
        pattern = random.choice(patterns)
        word_chars = []
        
        for char_type in pattern:
            if char_type == 'C':
                # Weight common consonants higher
                consonants = "bcdfghjklmnprstw" + "ntdslr" * 2  # Double common ones
                word_chars.append(random.choice(consonants))
            else:  # 'V'
                vowels = "aeiou" + "ae" * 2  # Double common vowels
                word_chars.append(random.choice(vowels))
                
        word = "".join(word_chars)
        
        if not self.is_valid_english_pattern(word) or word in self.generated_words:
            return None
            
        # Assign random properties
        category = random.choice(list(self.CATEGORIES.keys()))
        word_type = random.choice(["noun", "verb", "adjective", "adverb"])
        
        entry = VocabularyEntry(
            word=word,
            word_type=word_type,
            definition=f"A {word_type} related to {category}",
            synonym=word + "ing" if word_type == "verb" else word + "s",
            antonym="un" + word if len("un" + word) <= self.MAX_WORD_LENGTH else word + "less",
            sentiment_score=random.choice([-1, 0, 1]),
            frequency_score=random.uniform(0.1, 0.8),
            difficulty_level=random.randint(1, 4),
            category=category,
            etymology="synthetic",
            example_sentence=self.generate_example_sentence(word, word_type, category),
            phonetic=self.generate_phonetic(word),
            syllable_count=self.count_syllables(word),
            length=len(word),
            hash_id=self.generate_hash_id(word)
        )
        
        self.generated_words.add(word)
        return entry
        
    def load_external_wordlist(self, filepath: str) -> List[str]:
        """Load external word list if available"""
        try:
            if Path(filepath).exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    words = [line.strip().lower() for line in f 
                           if line.strip() and self.is_valid_english_pattern(line.strip())]
                logger.info(f"Loaded {len(words)} words from {filepath}")
                return words
        except Exception as e:
            logger.warning(f"Could not load wordlist from {filepath}: {e}")
        return []
        
    def generate_vocabulary_database(self) -> None:
        """Main method to generate complete vocabulary database"""
        logger.info(f"Starting vocabulary generation for {self.target_size:,} words")
        start_time = time.time()
        
        # Set up database
        conn = self.setup_database()
        
        # Load existing words to avoid duplicates
        cursor = conn.cursor()
        cursor.execute("SELECT word FROM vocabulary")
        existing_words = {row[0] for row in cursor.fetchall()}
        self.generated_words.update(existing_words)
        
        # Try to load external wordlist
        system_words = self.load_external_wordlist("/usr/share/dict/words")
        if system_words:
            # Process system words first
            logger.info("Processing system dictionary words...")
            self.process_system_words(conn, system_words[:self.target_size // 4])
        
        # Generate remaining words
        remaining = self.target_size - len(self.generated_words)
        logger.info(f"Generating {remaining:,} synthetic words...")
        
        # Use multiprocessing for faster generation
        with Pool(cpu_count()) as pool:
            batch_size = min(self.BATCH_SIZE, remaining // cpu_count() + 1)
            batches_needed = (remaining + batch_size - 1) // batch_size
            
            batch_func = partial(self.generate_vocabulary_batch_worker, batch_size)
            results = pool.map(batch_func, range(batches_needed))
            
            # Insert batches into database
            total_inserted = 0
            for batch_entries in results:
                if batch_entries:
                    self.insert_batch_to_db(conn, batch_entries)
                    total_inserted += len(batch_entries)
                    
            logger.info(f"Inserted {total_inserted:,} new vocabulary entries")
        
        # Create indices and finalize
        self.create_database_views(conn)
        conn.close()
        
        # Export to JSON
        self.export_to_json()
        
        elapsed = time.time() - start_time
        logger.info(f"Vocabulary generation completed in {elapsed:.2f} seconds")
        
    def generate_vocabulary_batch_worker(self, batch_size: int, batch_id: int) -> List[VocabularyEntry]:
        """Worker function for multiprocessing"""
        random.seed(os.urandom(16))  # Unique seed per process
        return self.generate_vocabulary_batch(batch_size)
        
    def process_system_words(self, conn: sqlite3.Connection, words: List[str]) -> None:
        """Process and enhance system dictionary words"""
        entries = []
        for word in words:
            if word not in self.generated_words and self.MIN_WORD_LENGTH <= len(word) <= self.MAX_WORD_LENGTH:
                # Create enhanced entry for real word
                category = self.classify_word_category(word)
                word_type = self.classify_word_type(word)
                
                entry = VocabularyEntry(
                    word=word,
                    word_type=word_type,
                    definition=f"Dictionary word ({word_type})",
                    synonym=self.find_similar_word(word),
                    antonym=self.generate_antonym(word, "", word, 0),
                    sentiment_score=self.analyze_word_sentiment(word),
                    frequency_score=self.estimate_word_frequency(word),
                    difficulty_level=self.calculate_difficulty(word, "", ""),
                    category=category,
                    etymology="dictionary",
                    example_sentence=self.generate_example_sentence(word, word_type, category),
                    phonetic=self.generate_phonetic(word),
                    syllable_count=self.count_syllables(word),
                    length=len(word),
                    hash_id=self.generate_hash_id(word)
                )
                entries.append(entry)
                self.generated_words.add(word)
                
        if entries:
            self.insert_batch_to_db(conn, entries)
            
    def classify_word_category(self, word: str) -> str:
        """Classify word into semantic category"""
        category_keywords = {
            "emotion": ["happy", "sad", "love", "hate", "joy", "fear", "anger"],
            "action": ["run", "walk", "jump", "move", "go", "come", "work"],
            "physical": ["big", "small", "tall", "short", "round", "square"],
            "cognitive": ["think", "know", "learn", "understand", "remember"],
            "social": ["friend", "family", "team", "group", "community"],
            "nature": ["tree", "flower", "water", "mountain", "sky", "earth"],
            "technology": ["computer", "phone", "internet", "software", "code"]
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in word for keyword in keywords):
                return category
                
        return "general"
        
    def classify_word_type(self, word: str) -> str:
        """Simple word type classification"""
        if word.endswith(("ing", "ed", "er", "en")):
            return "verb"
        elif word.endswith(("ly", "ward", "wise")):
            return "adverb"
        elif word.endswith(("ful", "less", "able", "ible", "ous", "al", "ic")):
            return "adjective"
        else:
            return "noun"
            
    def find_similar_word(self, word: str) -> str:
        """Find similar word for synonym"""
        if len(word) > 4:
            # Generate by modifying suffix
            return word[:-2] + "ly" if not word.endswith("ly") else word + "ing"
        return word + "s"
        
    def analyze_word_sentiment(self, word: str) -> int:
        """Basic sentiment analysis"""
        positive_words = {"good", "great", "happy", "love", "nice", "beautiful", "wonderful"}
        negative_words = {"bad", "terrible", "sad", "hate", "ugly", "awful", "horrible"}
        
        if any(pos in word for pos in positive_words):
            return 1
        elif any(neg in word for neg in negative_words):
            return -1
        return 0
        
    def estimate_word_frequency(self, word: str) -> float:
        """Estimate word frequency based on length and patterns"""
        base_freq = max(0.1, 1.0 - (len(word) - 3) * 0.05)
        
        # Common endings are more frequent
        if word.endswith(("ing", "ed", "er", "ly")):
            base_freq *= 1.2
            
        return min(1.0, base_freq)
        
    def insert_batch_to_db(self, conn: sqlite3.Connection, entries: List[VocabularyEntry]) -> None:
        """Insert batch of entries to database"""
        cursor = conn.cursor()
        
        insert_sql = '''
            INSERT OR IGNORE INTO vocabulary 
            (word, word_type, definition, synonym, antonym, sentiment_score, 
             frequency_score, difficulty_level, category, etymology, 
             example_sentence, phonetic, syllable_count, length, hash_id, created_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        
        data = [
            (
                entry.word, entry.word_type, entry.definition, entry.synonym, entry.antonym,
                entry.sentiment_score, entry.frequency_score, entry.difficulty_level,
                entry.category, entry.etymology, entry.example_sentence, entry.phonetic,
                entry.syllable_count, entry.length, entry.hash_id, time.time()
            )
            for entry in entries
        ]
        
        cursor.executemany(insert_sql, data)
        conn.commit()
        
    def create_database_views(self, conn: sqlite3.Connection) -> None:
        """Create useful database views"""
        cursor = conn.cursor()
        
        # View for words by difficulty
        cursor.execute('''
            CREATE VIEW IF NOT EXISTS words_by_difficulty AS
            SELECT difficulty_level, COUNT(*) as word_count, 
                   AVG(frequency_score) as avg_frequency,
                   AVG(length) as avg_length
            FROM vocabulary 
            GROUP BY difficulty_level
        ''')
        
        # View for words by category
        cursor.execute('''
            CREATE VIEW IF NOT EXISTS words_by_category AS
            SELECT category, COUNT(*) as word_count,
                   AVG(sentiment_score) as avg_sentiment,
                   AVG(difficulty_level) as avg_difficulty
            FROM vocabulary 
            GROUP BY category
        ''')
        
        conn.commit()
        
    def export_to_json(self) -> None:
        """Export vocabulary to JSON format"""
        logger.info("Exporting vocabulary to JSON...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM vocabulary ORDER BY word")
        columns = [description[0] for description in cursor.description]
        
        vocabulary_data = []
        for row in cursor.fetchall():
            entry_dict = dict(zip(columns, row))
            vocabulary_data.append(entry_dict)
            
        # Create export data structure
        export_data = {
            "metadata": {
                "total_words": len(vocabulary_data),
                "generated_timestamp": time.time(),
                "generator_version": "1.0",
                "database_file": str(self.db_path)
            },
            "statistics": self.generate_statistics(cursor),
            "vocabulary": vocabulary_data
        }
        
        # Save to JSON file
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
            
        conn.close()
        
        file_size = self.json_path.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"Vocabulary exported to {self.json_path} ({file_size:.2f} MB)")
        
    def generate_statistics(self, cursor: sqlite3.Cursor) -> Dict[str, any]:
        """Generate vocabulary statistics"""
        stats = {}
        
        # Basic counts
        cursor.execute("SELECT COUNT(*) FROM vocabulary")
        stats["total_words"] = cursor.fetchone()[0]
        
        # Word length distribution
        cursor.execute("SELECT length, COUNT(*) FROM vocabulary GROUP BY length ORDER BY length")
        stats["length_distribution"] = {str(length): count for length, count in cursor.fetchall()}
        
        # Difficulty distribution
        cursor.execute("SELECT difficulty_level, COUNT(*) FROM vocabulary GROUP BY difficulty_level")
        stats["difficulty_distribution"] = {str(level): count for level, count in cursor.fetchall()}
        
        # Category distribution
        cursor.execute("SELECT category, COUNT(*) FROM vocabulary GROUP BY category ORDER BY COUNT(*) DESC")
        stats["category_distribution"] = {category: count for category, count in cursor.fetchall()}
        
        # Word type distribution
        cursor.execute("SELECT word_type, COUNT(*) FROM vocabulary GROUP BY word_type")
        stats["word_type_distribution"] = {word_type: count for word_type, count in cursor.fetchall()}
        
        # Sentiment distribution
        cursor.execute("SELECT sentiment_score, COUNT(*) FROM vocabulary GROUP BY sentiment_score")
        stats["sentiment_distribution"] = {str(score): count for score, count in cursor.fetchall()}
        
        return stats
        
    def print_summary(self) -> None:
        """Print generation summary"""
        if not self.db_path.exists():
            print("❌ Database not found. Run generation first.")
            return
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM vocabulary")
        total_words = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT category) FROM vocabulary")
        total_categories = cursor.fetchone()[0]
        
        cursor.execute("SELECT AVG(difficulty_level) FROM vocabulary")
        avg_difficulty = cursor.fetchone()[0]
        
        cursor.execute("SELECT AVG(length) FROM vocabulary")
        avg_length = cursor.fetchone()[0]
        
        conn.close()
        
        print(f"\n📚 Intelligent Vocabulary Database Summary")
        print(f"="*50)
        print(f"📊 Total Words: {total_words:,}")
        print(f"🗂️  Categories: {total_categories}")
        print(f"📈 Average Difficulty: {avg_difficulty:.1f}/5")
        print(f"📏 Average Word Length: {avg_length:.1f} characters")
        print(f"💾 Database File: {self.db_path}")
        print(f"📄 JSON Export: {self.json_path}")
        print(f"💿 Database Size: {self.db_path.stat().st_size / 1024:.1f} KB")

def main():
    """Main execution function"""
    logger.info("Starting Intelligent Vocabulary Generator")
    
    generator = IntelligentVocabularyGenerator()
    
    try:
        # Generate vocabulary database
        generator.generate_vocabulary_database()
        
        # Print summary
        generator.print_summary()
        
        print(f"\n✅ Vocabulary generation completed successfully!")
        print(f"🎯 Use the SQLite database '{generator.OUTPUT_DB}' with your AI bot")
        print(f"📋 JSON export available at '{generator.OUTPUT_JSON}'")
        
    except KeyboardInterrupt:
        logger.info("Generation interrupted by user")
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise

if __name__ == "__main__":
    main()