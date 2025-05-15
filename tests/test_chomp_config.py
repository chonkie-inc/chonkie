"""Tests for the CHOMP configuration system."""

import sys
import os
import unittest
import tempfile
import json
import pickle

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chonkie import Chomp, ChompConfig, TextCleanerChef, RecursiveChunker
from src.chonkie.chomp.config import ChompConfig


class MockChef:
    """Simple mock chef for tests."""
    
    def __init__(self, available=True, prefix=""):
        """Initialize the mock chef.
        
        Args:
            available: Whether the chef is available.
            prefix: Prefix to add to the text.
        """
        self.available = available
        self.prefix = prefix
    
    def is_available(self):
        """Check if the chef is available.
        
        Returns:
            bool: True if the chef is available, False otherwise.
        """
        return self.available
    
    def preprocess(self, text):
        """Add a prefix to the text.
        
        Args:
            text: The text to preprocess.
            
        Returns:
            str: The preprocessed text.
        """
        return self.prefix + text


class TestChompConfig(unittest.TestCase):
    """Test the ChompConfig class."""
    
    def test_serialize_deserialize(self):
        """Test serialization and deserialization."""
        # Create pipeline components
        chef = TextCleanerChef(normalize_whitespace=True, strip=True)
        chunker = RecursiveChunker(chunk_size=100)
        
        # Create a pipeline
        pipeline = Chomp().add_chef(chef).set_chunker(chunker).build()
        
        # Serialize the config
        config = pipeline.to_config()
        
        # Print the config for debugging
        print("Serialized config:", json.dumps(config, default=str, indent=2))
        
        # Check that the config has the expected structure
        self.assertIn("chefs", config)
        self.assertIn("chunker", config)
        self.assertEqual(len(config["chefs"]), 1)
        self.assertEqual(config["chefs"][0]["type"], "TextCleanerChef")
        self.assertEqual(config["chunker"]["type"], "RecursiveChunker")
        
        # Deserialize the config
        loaded_pipeline = Chomp.from_config(config)
        
        # Check that the loaded pipeline has the expected components
        self.assertEqual(len(loaded_pipeline.chefs), 1)
        self.assertIsInstance(loaded_pipeline.chefs[0], TextCleanerChef)
        self.assertIsInstance(loaded_pipeline.chunker, RecursiveChunker)
        self.assertTrue(loaded_pipeline._is_built)
        
    def test_recursive_chunker_config(self):
        """Test specific serialization and deserialization of RecursiveChunker."""
        # Create chunker with basic parameters
        chunker = RecursiveChunker(chunk_size=100)
        
        # Create a pipeline
        pipeline = Chomp().set_chunker(chunker).build()
        
        # Serialize the config
        config = pipeline.to_config()
        
        # Print the chunker config for debugging
        print("RecursiveChunker config:", json.dumps(config["chunker"], default=str, indent=2))
        
        # Deserialize the config and check
        try:
            loaded_pipeline = Chomp.from_config(config)
            self.assertIsInstance(loaded_pipeline.chunker, RecursiveChunker)
            self.assertEqual(loaded_pipeline.chunker.chunk_size, 100)
        except Exception as e:
            self.fail(f"Failed to deserialize RecursiveChunker: {e}")
            
    def test_save_load_json(self):
        """Test saving and loading from JSON."""
        # Create a simple pipeline
        pipeline = (
            Chomp()
            .add_chef(TextCleanerChef(normalize_whitespace=True))
            .set_chunker(RecursiveChunker())
            .build()
        )
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp:
            filepath = temp.name
            
        try:
            # Save the config
            pipeline.save_config(filepath)
            
            # Verify the file exists and has content
            self.assertTrue(os.path.exists(filepath))
            with open(filepath, 'r') as f:
                content = f.read()
            self.assertGreater(len(content), 10)  # Should have some content
            
            # Load the config
            loaded_pipeline = Chomp.load_config(filepath)
            
            # Verify the loaded pipeline has the right components
            self.assertEqual(len(loaded_pipeline.chefs), 1)
            self.assertIsInstance(loaded_pipeline.chefs[0], TextCleanerChef)
            self.assertIsInstance(loaded_pipeline.chunker, RecursiveChunker)
            
        finally:
            # Clean up
            if os.path.exists(filepath):
                os.remove(filepath)
                
    def test_save_load_pickle(self):
        """Test saving and loading from pickle."""
        # Create a simple pipeline
        pipeline = (
            Chomp()
            .add_chef(TextCleanerChef(normalize_whitespace=True))
            .set_chunker(RecursiveChunker())
            .build()
        )
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp:
            filepath = temp.name
            
        try:
            # Save the config
            pipeline.save_config(filepath, format="pickle")
            
            # Verify the file exists and has content
            self.assertTrue(os.path.exists(filepath))
            with open(filepath, 'rb') as f:
                content = f.read()
            self.assertGreater(len(content), 10)  # Should have some content
            
            # Load the config
            loaded_pipeline = Chomp.load_config(filepath, format="pickle")
            
            # Verify the loaded pipeline has the right components
            self.assertEqual(len(loaded_pipeline.chefs), 1)
            self.assertIsInstance(loaded_pipeline.chefs[0], TextCleanerChef)
            self.assertIsInstance(loaded_pipeline.chunker, RecursiveChunker)
            
        finally:
            # Clean up
            if os.path.exists(filepath):
                os.remove(filepath)


if __name__ == "__main__":
    unittest.main() 