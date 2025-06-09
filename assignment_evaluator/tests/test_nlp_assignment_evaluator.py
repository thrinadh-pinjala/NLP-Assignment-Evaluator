import unittest
from src.nlp_assignment_evaluator import preprocess, get_sentence_vector, cosine_similarity, evaluate_assignment

class TestNlpAssignmentEvaluator(unittest.TestCase):

    def test_preprocess(self):
        text = "This is a sample text for testing."
        expected_output = ['sample', 'text', 'testing']
        self.assertEqual(preprocess(text), expected_output)

    def test_get_sentence_vector(self):
        text = "This is a sample text."
        vector = get_sentence_vector(text)
        self.assertEqual(len(vector), 300)  # Assuming the SpaCy model has 300 dimensions

    def test_cosine_similarity(self):
        vec1 = [1, 0, 0]
        vec2 = [0, 1, 0]
        self.assertEqual(cosine_similarity(vec1, vec2), 0.0)

        vec1 = [1, 0, 0]
        vec2 = [1, 0, 0]
        self.assertEqual(cosine_similarity(vec1, vec2), 1.0)

    def test_evaluate_assignment(self):
        model_answer = "The quick brown fox jumps over the lazy dog."
        student_answer = "The quick brown fox."
        similarity_percent, grade = evaluate_assignment(model_answer, student_answer)
        self.assertIsInstance(similarity_percent, float)
        self.assertIn(grade, ['A (Excellent)', 'B (Good)', 'C (Fair)', 'F (Needs Improvement)'])

if __name__ == '__main__':
    unittest.main()