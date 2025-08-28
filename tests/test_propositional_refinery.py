"""Tests for PropositionalRefinery."""

import unittest
from chonkie.refinery import PropositionalRefinery
from chonkie.types import Chunk


class TestPropositionalRefinery(unittest.TestCase):
    """Test cases for PropositionalRefinery."""

    def setUp(self):
        """Set up test fixtures."""
        self.refinery = PropositionalRefinery(
            method="rule-based",
            min_prop_length=5,  # Very low threshold
            max_prop_length=500,  # Very high threshold
            min_token_count=1,  # Very low threshold
            max_token_count=100,  # Very high threshold
            merge_short=True,
            inplace=True,
        )
        
        # Sample text with multiple propositions
        self.sample_text = (
            "The Dense X Retrieval paper introduces propositions as atomic expressions. "
            "Propositions encapsulate distinct factoids in a concise format. "
            "Studies show that proposition-based retrieval outperforms traditional methods, "
            "because the retrieved texts are more condensed with relevant information."
        )
        
        # Create a sample chunk
        self.sample_chunk = Chunk(
            text=self.sample_text,
            start_index=0,
            end_index=len(self.sample_text),
            token_count=len(self.sample_text.split()),  # Simple word count as approximation
        )

    def test_refinery_initialization(self):
        """Test that the refinery initializes correctly."""
        self.assertEqual(self.refinery.method, "rule-based")
        self.assertEqual(self.refinery.min_prop_length, 5)
        self.assertEqual(self.refinery.max_token_count, 100)

    def test_is_available(self):
        """Test if the refinery is available."""
        # This may fail if spaCy is not installed
        try:
            import spacy
            self.assertTrue(self.refinery.is_available())
        except ImportError:
            # Skip the test if spaCy is not installed
            self.skipTest("spaCy is not installed")

    def test_extract_propositions(self):
        """Test proposition extraction."""
        # If spaCy is not installed, skip this test
        if not self.refinery.is_available():
            self.skipTest("Required dependencies are not available")
        
        propositions = self.refinery._extract_propositions(self.sample_text)
        
        # We should have at least one proposition
        self.assertGreater(len(propositions), 0)
        
        # Each proposition should be a non-empty string
        for prop in propositions:
            self.assertIsInstance(prop, str)
            self.assertGreater(len(prop), 0)

    def test_refine_chunks(self):
        """Test refining chunks into propositions."""
        # If spaCy is not installed, skip this test
        if not self.refinery.is_available():
            self.skipTest("Required dependencies are not available")
        
        # Create a list with a single chunk
        chunks = [self.sample_chunk]
        
        # Refine the chunks
        refined_chunks = self.refinery.refine(chunks)
        
        # We should have at least one refined chunk
        self.assertGreater(len(refined_chunks), 0)
        
        # Each refined chunk should be a Chunk instance
        for chunk in refined_chunks:
            self.assertIsInstance(chunk, Chunk)
            
            # The text should be a non-empty string
            self.assertIsInstance(chunk.text, str)
            self.assertGreater(len(chunk.text), 0)
            
            # Check that start_index and end_index are consistent
            self.assertLessEqual(chunk.start_index, chunk.end_index)
            self.assertEqual(len(chunk.text), chunk.end_index - chunk.start_index)
            
            # Check that the token count is reasonable
            self.assertGreater(chunk.token_count, 0)

    def test_empty_chunks(self):
        """Test that refining empty chunks returns empty list."""
        refined_chunks = self.refinery.refine([])
        self.assertEqual(refined_chunks, [])

    def test_chunk_with_no_propositions(self):
        """Test refining a chunk that doesn't yield any propositions."""
        # Create a chunk with text that's too short to yield propositions
        short_chunk = Chunk(
            text="Short.",
            start_index=0,
            end_index=6,
            token_count=1,
        )
        
        # Refine the chunk
        refined_chunks = self.refinery.refine([short_chunk])
        
        # We should still have one chunk (the original)
        self.assertEqual(len(refined_chunks), 1)
        self.assertEqual(refined_chunks[0].text, "Short.")


if __name__ == "__main__":
    unittest.main() 