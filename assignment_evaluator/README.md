# NLP-Based Assignment Evaluator

## Overview
The NLP-Based Assignment Evaluator is a Python application that evaluates the semantic similarity of a student's assignment answer against a model answer using Natural Language Processing (NLP) techniques. It leverages the capabilities of SpaCy and NLTK for text preprocessing and vector representation, providing an automated grading system based on cosine similarity.

## Features
- Text preprocessing with NLTK (stopwords removal, tokenization)
- Vector representation using SpaCy pre-trained word embeddings
- Cosine similarity calculation between model and student answers
- Threshold-based grading for auto evaluation

## Installation

### Prerequisites
Make sure you have Python 3.6 or higher installed on your machine.

### Dependencies
To install the required dependencies, run the following command:

```
pip install -r requirements.txt
```

Additionally, you need to download the SpaCy medium English model. You can do this by running:

```
python -m spacy download en_core_web_md
```

## Usage
To run the application, execute the following command in your terminal:

```
python src/nlp_assignment_evaluator.py
```

You will be prompted to enter the model answer and the student answer. The application will then evaluate the similarity and provide a grade based on predefined thresholds.

## Testing
To ensure the functionality of the application, unit tests are provided in the `tests/test_nlp_assignment_evaluator.py` file. You can run the tests using:

```
pytest tests/
```

## Contribution
Contributions are welcome! If you would like to contribute to this project, please fork the repository and submit a pull request with your changes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.