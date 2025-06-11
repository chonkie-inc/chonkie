"""Tests for the Chomp pipeline."""

import sys
import os
import unittest
from unittest.mock import MagicMock

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chonkie.chomp import Chomp
from src.chonkie.chefs.base import BaseChef
from src.chonkie.chunker.base import BaseChunker
from src.chonkie.refinery.base import BaseRefinery
from src.chonkie.friends import BasePorter, BaseHandshake
from src.chonkie.types import Chunk


class MockChef(BaseChef):
    """Mock chef for testing."""
    
    def __init__(self, available=True, prefix="CHEF_"):
        """Initialize the mock chef."""
        super().__init__()
        self._available = available
        self.prefix = prefix
        
    def is_available(self):
        """Check if the chef is available."""
        return self._available
        
    def preprocess(self, text):
        """Mock preprocessing."""
        return f"{self.prefix}{text}"


class MockChunker(BaseChunker):
    """Mock chunker for testing."""
    
    def __init__(self, return_value=None):
        """Initialize the mock chunker."""
        super().__init__(tokenizer_or_token_counter="character")
        self.return_value = return_value or []
        
    def chunk(self, text):
        """Mock chunking."""
        # Create a simple chunk from the text
        if not self.return_value:
            return [
                Chunk(
                    text=text,
                    start_index=0,
                    end_index=len(text),
                    token_count=len(text),
                )
            ]
        return self.return_value


class MockRefinery(BaseRefinery):
    """Mock refinery for testing."""
    
    def __init__(self, available=True, suffix="_REFINED"):
        """Initialize the mock refinery."""
        self._available = available
        self.suffix = suffix
        
    def is_available(self):
        """Check if the refinery is available."""
        return self._available
        
    def refine(self, chunks):
        """Mock refining."""
        for chunk in chunks:
            chunk.text += self.suffix
        return chunks


class TestChomp(unittest.TestCase):
    """Test the Chomp pipeline."""
    
    def test_pipeline_creation(self):
        """Test creating a pipeline."""
        pipeline = Chomp()
        self.assertIsInstance(pipeline, Chomp)
        self.assertEqual(len(pipeline.chefs), 0)
        self.assertIsNone(pipeline.chunker)
        self.assertEqual(len(pipeline.refineries), 0)
        self.assertIsNone(pipeline.porter)
        self.assertIsNone(pipeline.handshake)
        
    def test_add_chef(self):
        """Test adding a chef to the pipeline."""
        pipeline = Chomp()
        chef = MockChef()
        result = pipeline.add_chef(chef)
        
        self.assertEqual(pipeline.chefs, [chef])
        self.assertIs(result, pipeline)  # Should return self for chaining
        
    def test_add_unavailable_chef(self):
        """Test adding an unavailable chef raises ValueError."""
        pipeline = Chomp()
        chef = MockChef(available=False)
        
        with self.assertRaises(ValueError):
            pipeline.add_chef(chef)
            
    def test_set_chunker(self):
        """Test setting a chunker in the pipeline."""
        pipeline = Chomp()
        chunker = MockChunker()
        result = pipeline.set_chunker(chunker)
        
        self.assertEqual(pipeline.chunker, chunker)
        self.assertIs(result, pipeline)  # Should return self for chaining
        
    def test_set_duplicate_chunker(self):
        """Test setting a chunker when one is already set raises ValueError."""
        pipeline = Chomp()
        chunker1 = MockChunker()
        chunker2 = MockChunker()
        
        pipeline.set_chunker(chunker1)
        with self.assertRaises(ValueError):
            pipeline.set_chunker(chunker2)
            
    def test_add_refinery(self):
        """Test adding a refinery to the pipeline."""
        pipeline = Chomp()
        refinery = MockRefinery()
        result = pipeline.add_refinery(refinery)
        
        self.assertEqual(pipeline.refineries, [refinery])
        self.assertIs(result, pipeline)  # Should return self for chaining
        
    def test_add_unavailable_refinery(self):
        """Test adding an unavailable refinery raises ValueError."""
        pipeline = Chomp()
        refinery = MockRefinery(available=False)
        
        with self.assertRaises(ValueError):
            pipeline.add_refinery(refinery)
            
    def test_build(self):
        """Test building the pipeline."""
        pipeline = Chomp()
        pipeline.set_chunker(MockChunker())
        result = pipeline.build()
        
        self.assertTrue(pipeline._is_built)
        self.assertIs(result, pipeline)  # Should return self for chaining
        
    def test_build_no_chunker(self):
        """Test building a pipeline without a chunker raises ValueError."""
        pipeline = Chomp()
        
        with self.assertRaises(ValueError):
            pipeline.build()
            
    def test_modify_after_build(self):
        """Test modifying the pipeline after building raises ValueError."""
        pipeline = Chomp().set_chunker(MockChunker()).build()
        
        with self.assertRaises(ValueError):
            pipeline.add_chef(MockChef())
            
        with self.assertRaises(ValueError):
            pipeline.set_chunker(MockChunker())
            
        with self.assertRaises(ValueError):
            pipeline.add_refinery(MockRefinery())
            
    def test_process(self):
        """Test processing text through the pipeline."""
        chef1 = MockChef(prefix="CHEF1_")
        chef2 = MockChef(prefix="CHEF2_")
        chunker = MockChunker()
        refinery = MockRefinery(suffix="_REF")
        
        pipeline = (
            Chomp()
            .add_chef(chef1)
            .add_chef(chef2)
            .set_chunker(chunker)
            .add_refinery(refinery)
            .build()
        )
        
        result = pipeline.process("test")
        
        # The text should be processed by both chefs in order, then chunked, then refined
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].text, "CHEF2_CHEF1_test_REF")
        
    def test_process_not_built(self):
        """Test processing text through a pipeline that is not built raises ValueError."""
        pipeline = Chomp()
        
        with self.assertRaises(ValueError):
            pipeline.process("test")
            
    def test_call(self):
        """Test calling the pipeline as a function."""
        pipeline = Chomp().set_chunker(MockChunker()).build()
        
        # Mock the process method
        pipeline.process = MagicMock(return_value="result")
        
        result = pipeline("test")
        
        # The __call__ method should call process
        pipeline.process.assert_called_once_with("test")
        self.assertEqual(result, "result")
        
    def test_porter_and_handshake(self):
        """Test exporting with a porter and handshake."""
        # Create mock porter and handshake
        porter = MagicMock(spec=BasePorter)
        porter.return_value = "porter_result"
        
        handshake = MagicMock(spec=BaseHandshake)
        handshake.return_value = "handshake_result"
        
        # Create pipeline
        chunker = MockChunker()
        pipeline = (
            Chomp()
            .set_chunker(chunker)
            .set_porter(porter)
            .set_handshake(handshake)
            .build()
        )
        
        result = pipeline("test")
        
        # The porter should be called first, then the handshake
        porter.assert_called_once()
        handshake.assert_called_once()
        self.assertEqual(result, "handshake_result")


if __name__ == "__main__":
    unittest.main() 