"""Tests for ContextualRefinery."""

import unittest
from chonkie.refinery import ContextualRefinery
from chonkie.types import Chunk


class TestContextualRefinery(unittest.TestCase):
    """Test cases for ContextualRefinery."""

    def setUp(self):
        """Set up test fixtures."""
        self.refinery = ContextualRefinery(
            context_window=50,
            min_context_score=0.3,
            scoring_method="frequency",
            inplace=False,
        )
        
        # Complete test text
        self.full_text = (
            "Machine learning is a field of study that gives computers the ability to learn "
            "without being explicitly programmed. It is considered a subset of artificial intelligence. "
            "Deep learning is a part of machine learning based on artificial neural networks. "
            "Neural networks are inspired by the structure of the human brain and consist of layers of nodes. "
            "Convolutional Neural Networks (CNNs) are particularly useful for image recognition tasks. "
            "They use convolutional layers to automatically detect features in images. "
            "Recurrent Neural Networks (RNNs) are designed to work with sequential data like text or time series. "
            "They have connections that form directed cycles, allowing them to maintain memory of previous inputs."
        )
        
        # Create sample chunks from the text
        self.chunks = [
            # Chunk about machine learning
            Chunk(
                text=self.full_text[:125],
                start_index=0,
                end_index=125,
                token_count=125,
            ),
            # Chunk about deep learning and neural networks
            Chunk(
                text=self.full_text[125:300],
                start_index=125,
                end_index=300,
                token_count=175,
            ),
            # Chunk about CNNs
            Chunk(
                text=self.full_text[300:450],
                start_index=300,
                end_index=450,
                token_count=150,
            ),
            # Chunk about RNNs
            Chunk(
                text=self.full_text[450:],
                start_index=450,
                end_index=len(self.full_text),
                token_count=len(self.full_text[450:]),
            ),
        ]

    def test_refinery_initialization(self):
        """Test that the refinery initializes correctly."""
        self.assertEqual(self.refinery.context_window, 50)
        self.assertEqual(self.refinery.min_context_score, 0.3)
        self.assertEqual(self.refinery.scoring_method, "frequency")
        self.assertFalse(self.refinery.inplace)

    def test_is_available(self):
        """Test if the refinery is available."""
        # This may fail if required dependencies are not installed
        try:
            self.assertTrue(self.refinery.is_available())
        except ImportError:
            self.skipTest("Required dependencies are not installed")

    def test_extract_full_text(self):
        """Test extracting full text from chunks."""
        extracted_text = self.refinery._extract_full_text(self.chunks)
        self.assertEqual(extracted_text, self.full_text)

    def test_get_context_windows(self):
        """Test getting context windows."""
        # Get context windows for the third chunk (about CNNs)
        prefix, suffix = self.refinery._get_context_windows(self.full_text, self.chunks[2])
        
        # Check prefix context
        self.assertEqual(prefix, self.full_text[250:300])  # 50 chars before the chunk
        
        # Check suffix context
        self.assertEqual(suffix, self.full_text[450:500])  # 50 chars after the chunk (or to the end)

    def test_score_context_frequency(self):
        """Test scoring context relevance with frequency method."""
        chunk_text = "Neural networks are inspired by the structure of the human brain."
        context_text = "Deep learning uses neural networks with multiple layers for complex tasks."
        
        # The score should be positive since there's term overlap
        score = self.refinery._score_context_frequency(chunk_text, context_text)
        self.assertGreater(score, 0.0)

    def test_refine_chunks(self):
        """Test refining chunks with context."""
        # If dependencies aren't available, skip test
        if not self.refinery.is_available():
            self.skipTest("Required dependencies are not available")
        
        refined_chunks = self.refinery.refine(self.chunks)
        
        # We should have the same number of chunks
        self.assertEqual(len(refined_chunks), len(self.chunks))
        
        # Check that at least some chunks have been refined with context
        # (the exact number depends on scoring thresholds)
        has_refined = False
        for chunk, refined in zip(self.chunks, refined_chunks):
            if len(refined.text) > len(chunk.text):
                has_refined = True
                break
        
        self.assertTrue(has_refined, "No chunks were refined with context")
        
        # Check that refined chunks have proper indices
        for refined in refined_chunks:
            self.assertEqual(len(refined.text), refined.end_index - refined.start_index)

    def test_empty_chunks(self):
        """Test that refining empty chunks returns empty list."""
        refined_chunks = self.refinery.refine([])
        self.assertEqual(refined_chunks, [])

    def test_inplace_modification(self):
        """Test inplace modification vs. copying."""
        # Create an inplace refinery
        inplace_refinery = ContextualRefinery(
            context_window=50,
            min_context_score=0.3,
            scoring_method="frequency",
            inplace=True,
        )
        
        # Create a copy of the chunks for testing
        original_chunks = [chunk.copy() for chunk in self.chunks]
        
        # Refine with inplace=True
        refined_chunks = inplace_refinery.refine(original_chunks)
        
        # The refined chunks should be the same objects as the original chunks
        for original, refined in zip(original_chunks, refined_chunks):
            self.assertIs(original, refined)

if __name__ == "__main__":
    unittest.main() 