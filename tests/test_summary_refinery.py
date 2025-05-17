"""Tests for SummaryRefinery."""

import unittest
from chonkie.refinery import SummaryRefinery
from chonkie.types import Chunk


class TestSummaryRefinery(unittest.TestCase):
    """Test cases for SummaryRefinery."""

    def setUp(self):
        """Set up test fixtures."""
        self.refinery = SummaryRefinery(
            method="extractive",
            max_summary_length=100,
            min_length=30,
            summary_location="context",
            inplace=True,
        )
        
        # Sample text for testing
        self.sample_text = (
            "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines. "
            "These machines are programmed to think and learn like humans. "
            "The term may also be applied to any machine that exhibits traits associated with a human mind. "
            "Such traits include learning and problem-solving abilities. "
            "The core problems of artificial intelligence include programming computers for certain traits. "
            "These traits include knowledge, reasoning, problem solving, perception, and learning. "
            "AI is being used in many industries today. "
            "Examples include healthcare, finance, education, and transportation. "
            "Machine learning is a subset of AI. "
            "It focuses on the development of computer programs that can access data and use it to learn for themselves."
        )
        
        # Create a sample chunk
        self.sample_chunk = Chunk(
            text=self.sample_text,
            start_index=0,
            end_index=len(self.sample_text),
            token_count=len(self.sample_text),  # Simple character count as approximation
        )

    def test_refinery_initialization(self):
        """Test that the refinery initializes correctly."""
        self.assertEqual(self.refinery.method, "extractive")
        self.assertEqual(self.refinery.max_summary_length, 100)
        self.assertEqual(self.refinery.min_length, 30)
        self.assertEqual(self.refinery.summary_location, "context")
        self.assertTrue(self.refinery.inplace)

    def test_is_available(self):
        """Test if the refinery is available."""
        # This may fail if required dependencies are not installed
        try:
            self.assertTrue(self.refinery.is_available())
        except ImportError:
            self.skipTest("Required dependencies are not installed")

    def test_summarize_extractive(self):
        """Test extractive summarization."""
        # If dependencies aren't available, skip test
        if not self.refinery.is_available():
            self.skipTest("Required dependencies are not available")
        
        summary = self.refinery._summarize_extractive(self.sample_text)
        
        # Check that summary is not empty
        self.assertGreater(len(summary), 0)
        
        # Check that summary is shorter than original text
        self.assertLess(len(summary), len(self.sample_text))
        
        # Check that summary doesn't exceed max length
        self.assertLessEqual(len(summary), self.refinery.max_summary_length)

    def test_apply_summary_context(self):
        """Test applying summary to chunk context."""
        # Create a test summary
        test_summary = "This is a test summary."
        
        # Apply the summary
        refined_chunk = self.refinery._apply_summary(self.sample_chunk, test_summary)
        
        # Check that summary was added to context
        self.assertIsNotNone(refined_chunk.context)
        self.assertIn("summary", refined_chunk.context)
        self.assertEqual(refined_chunk.context["summary"], test_summary)
        
        # Check that text was not modified (since summary_location is "context")
        self.assertEqual(refined_chunk.text, self.sample_chunk.text)

    def test_apply_summary_prepend(self):
        """Test applying summary by prepending to text."""
        # Create refinery with prepend option
        prepend_refinery = SummaryRefinery(
            method="extractive",
            max_summary_length=100,
            min_length=30,
            summary_location="prepend",
            inplace=True,
        )
        
        # Create a test summary
        test_summary = "This is a test summary."
        
        # Apply the summary
        chunk_copy = self.sample_chunk.copy()
        refined_chunk = prepend_refinery._apply_summary(chunk_copy, test_summary)
        
        # Check that summary was added to context
        self.assertIsNotNone(refined_chunk.context)
        self.assertIn("summary", refined_chunk.context)
        
        # Check that text was modified with summary prepended
        expected_text = test_summary + prepend_refinery.summary_separator + self.sample_chunk.text
        self.assertEqual(refined_chunk.text, expected_text)
        
        # Check that token count was updated
        self.assertEqual(refined_chunk.token_count, len(expected_text))

    def test_apply_summary_append(self):
        """Test applying summary by appending to text."""
        # Create refinery with append option
        append_refinery = SummaryRefinery(
            method="extractive",
            max_summary_length=100,
            min_length=30,
            summary_location="append",
            inplace=True,
        )
        
        # Create a test summary
        test_summary = "This is a test summary."
        
        # Apply the summary
        chunk_copy = self.sample_chunk.copy()
        refined_chunk = append_refinery._apply_summary(chunk_copy, test_summary)
        
        # Check that summary was added to context
        self.assertIsNotNone(refined_chunk.context)
        self.assertIn("summary", refined_chunk.context)
        
        # Check that text was modified with summary appended
        expected_text = self.sample_chunk.text + append_refinery.summary_separator + test_summary
        self.assertEqual(refined_chunk.text, expected_text)
        
        # Check that token count was updated
        self.assertEqual(refined_chunk.token_count, len(expected_text))

    def test_refine_chunks(self):
        """Test refining chunks with summaries."""
        # If dependencies aren't available, skip test
        if not self.refinery.is_available():
            self.skipTest("Required dependencies are not available")
        
        # Create a list with a single chunk
        chunks = [self.sample_chunk]
        
        # Refine the chunks
        refined_chunks = self.refinery.refine(chunks)
        
        # We should have one refined chunk
        self.assertEqual(len(refined_chunks), 1)
        
        # The refined chunk should have a summary in its context
        refined_chunk = refined_chunks[0]
        self.assertIsNotNone(refined_chunk.context)
        self.assertIn("summary", refined_chunk.context)
        
        # The summary should be non-empty
        self.assertGreater(len(refined_chunk.context["summary"]), 0)
        
        # The summary should be shorter than the original text
        self.assertLess(len(refined_chunk.context["summary"]), len(self.sample_chunk.text))

    def test_empty_chunks(self):
        """Test that refining empty chunks returns empty list."""
        refined_chunks = self.refinery.refine([])
        self.assertEqual(refined_chunks, [])

    def test_inplace_vs_copy(self):
        """Test inplace modification vs. copying."""
        # Create a refinery with inplace=False
        copy_refinery = SummaryRefinery(
            method="extractive",
            max_summary_length=100,
            min_length=30,
            summary_location="context",
            inplace=False,
        )
        
        # Create a copy of the sample chunk for testing
        original_chunk = self.sample_chunk.copy()
        
        # Refine with inplace=False
        refined_chunks = copy_refinery.refine([original_chunk])
        
        # The refined chunk should be a different object
        self.assertIsNot(refined_chunks[0], original_chunk)
        
        # Now test with inplace=True
        inplace_refinery = SummaryRefinery(
            method="extractive",
            max_summary_length=100,
            min_length=30,
            summary_location="context",
            inplace=True,
        )
        
        # Create another copy of the sample chunk
        inplace_chunk = self.sample_chunk.copy()
        
        # Refine with inplace=True
        inplace_refined = inplace_refinery.refine([inplace_chunk])
        
        # The refined chunk should be the same object
        self.assertIs(inplace_refined[0], inplace_chunk)


if __name__ == "__main__":
    unittest.main() 