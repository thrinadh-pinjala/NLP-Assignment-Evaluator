"""
NLP-Based Assignment Evaluator
----------------------------------------
This script evaluates the semantic similarity of a student's assignment answer
against a model answer using SpaCy and NLTK.

Features:
- Text preprocessing with NLTK (stopwords removal, tokenization)
- Vector representation using SpaCy pre-trained word embeddings
- Cosine similarity between model and student answers
- Threshold-based grading for auto evaluation

Usage:
- Make sure to install dependencies:
    pip install spacy nltk
- Download SpaCy model (only once):
    python -m spacy download en_core_web_md
- Run the script:
    python nlp_assignment_evaluator.py
"""

import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import sys

# Download required NLTK data files if not already present
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)  # Add this line

# Load SpaCy medium English model (has word vectors)
nlp = spacy.load("en_core_web_md")

# Set of English stopwords
stop_words = set(stopwords.words('english'))

def preprocess(text):
    """
    Tokenize, lowercase, remove stopwords and punctuation from text.
    Return a list of clean tokens.
    """
    tokens = word_tokenize(text.lower())
    filtered_tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return filtered_tokens

def get_sentence_vector(text):
    """
    Get the average SpaCy word vector for the given text (after preprocessing)
    Return a numpy array vector.
    If no known tokens, return zero vector.
    """
    tokens = preprocess(text)
    if not tokens:
        return np.zeros(nlp.vocab.vectors_length)

    # Sum vectors for tokens if in SpaCy vocab
    vectors = [nlp(token).vector for token in tokens if nlp.vocab.has_vector(token)]
    if not vectors:
        return np.zeros(nlp.vocab.vectors_length)
    avg_vector = np.mean(vectors, axis=0)
    return avg_vector

def cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two vectors.
    Handle zero vectors gracefully by returning 0 similarity.
    """
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def evaluate_assignment(model_answer, student_answer):
    """
    Compute similarity score and assign a grade based on thresholds.
    """
    vec_model = get_sentence_vector(model_answer)
    vec_student = get_sentence_vector(student_answer)
    similarity = cosine_similarity(vec_model, vec_student)

    similarity_percent = similarity * 100

    # Thresholds for grading (can be fine-tuned)
    if similarity >= 0.85:
        grade = 'A (Excellent)'
    elif similarity >= 0.70:
        grade = 'B (Good)'
    elif similarity >= 0.50:
        grade = 'C (Fair)'
    else:
        grade = 'F (Needs Improvement)'

    return similarity_percent, grade

def main():
    print("NLP-Based Assignment Evaluator")
    print("-" * 35)
    print("Enter the model answer:")
    model_answer = sys.stdin.readline().strip()
    print("\nEnter the student answer:")
    student_answer = sys.stdin.readline().strip()

    similarity_percent, grade = evaluate_assignment(model_answer, student_answer)
    print("\nEvaluation Result:")
    print(f"Semantic Similarity Score: {similarity_percent:.2f}%")
    print(f"Assigned Grade: {grade}")

if __name__ == "__main__":
    main()

