"""Tests for structured data chef classes."""

import sys
import os
import unittest
import json

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chonkie.chefs.structured import JSONCleanerChef, CSVCleanerChef


class TestJSONCleanerChef(unittest.TestCase):
    """Test the JSONCleanerChef class."""
    
    def test_empty_input(self):
        """Test processing an empty string."""
        chef = JSONCleanerChef()
        self.assertEqual(chef.preprocess(""), "")
        self.assertEqual(chef.preprocess("  "), "")
    
    def test_invalid_json(self):
        """Test processing invalid JSON."""
        chef = JSONCleanerChef()
        result = chef.preprocess("{invalid json}")
        self.assertTrue(result.startswith("Error parsing JSON"))
        
        # Test that errors are raised if handle_errors is False
        chef = JSONCleanerChef(handle_errors=False)
        with self.assertRaises(json.JSONDecodeError):
            chef.preprocess("{invalid json}")
    
    def test_basic_json(self):
        """Test processing basic JSON."""
        chef = JSONCleanerChef()
        json_str = '{"name": "John", "age": 30, "city": "New York"}'
        result = chef.preprocess(json_str)
        self.assertIn("name: John", result)
        self.assertIn("age: 30", result)
        self.assertIn("city: New York", result)
        
    def test_nested_json(self):
        """Test processing nested JSON."""
        chef = JSONCleanerChef()
        json_str = '''
        {
            "user": {
                "name": "John",
                "profile": {
                    "age": 30,
                    "city": "New York"
                }
            },
            "status": "active"
        }
        '''
        result = chef.preprocess(json_str)
        # Default behavior is to join only top-level fields
        self.assertIn("user: {", result)
        self.assertIn("status: active", result)
        
    def test_flatten_json(self):
        """Test flattening nested JSON."""
        chef = JSONCleanerChef(flatten=True)
        json_str = '''
        {
            "user": {
                "name": "John",
                "profile": {
                    "age": 30,
                    "city": "New York"
                }
            },
            "status": "active"
        }
        '''
        result = chef.preprocess(json_str)
        self.assertIn("user_name: John", result)
        self.assertIn("user_profile_age: 30", result)
        self.assertIn("user_profile_city: New York", result)
        self.assertIn("status: active", result)
        
    def test_extract_fields(self):
        """Test extracting specific fields from JSON."""
        chef = JSONCleanerChef(extract_fields=["name", "user.profile.city"])
        json_str = '''
        {
            "name": "Example",
            "user": {
                "name": "John",
                "profile": {
                    "age": 30,
                    "city": "New York"
                }
            },
            "status": "active"
        }
        '''
        result = chef.preprocess(json_str)
        self.assertIn("name: Example", result)
        self.assertIn("user.profile.city: New York", result)
        self.assertNotIn("status: active", result)
        
    def test_no_join_text_fields(self):
        """Test returning JSON without joining text fields."""
        chef = JSONCleanerChef(join_text_fields=False)
        json_str = '{"name": "John", "age": 30}'
        result = chef.preprocess(json_str)
        # Result should be a formatted JSON string
        parsed = json.loads(result)
        self.assertEqual(parsed["name"], "John")
        self.assertEqual(parsed["age"], 30)


class TestCSVCleanerChef(unittest.TestCase):
    """Test the CSVCleanerChef class."""
    
    def test_empty_input(self):
        """Test processing an empty string."""
        chef = CSVCleanerChef()
        self.assertEqual(chef.preprocess(""), "")
        self.assertEqual(chef.preprocess("  "), "")
    
    def test_basic_csv(self):
        """Test processing basic CSV."""
        chef = CSVCleanerChef()
        csv_str = "name,age,city\nJohn,30,New York\nJane,25,Boston"
        result = chef.preprocess(csv_str)
        
        # Check that the rows were processed
        self.assertIn("name: John", result)
        self.assertIn("age: 30", result)
        self.assertIn("city: New York", result)
        self.assertIn("name: Jane", result)
        self.assertIn("age: 25", result)
        self.assertIn("city: Boston", result)
        
    def test_no_header(self):
        """Test processing CSV without a header."""
        chef = CSVCleanerChef(has_header=False)
        csv_str = "John,30,New York\nJane,25,Boston"
        result = chef.preprocess(csv_str)
        
        # Check that column names are generated
        self.assertIn("col_0: John", result)
        self.assertIn("col_1: 30", result)
        self.assertIn("col_2: New York", result)
        
    def test_custom_delimiter(self):
        """Test processing CSV with a custom delimiter."""
        chef = CSVCleanerChef(delimiter=";")
        csv_str = "name;age;city\nJohn;30;New York\nJane;25;Boston"
        result = chef.preprocess(csv_str)
        
        self.assertIn("name: John", result)
        self.assertIn("age: 30", result)
        self.assertIn("city: New York", result)
        
    def test_extract_columns(self):
        """Test extracting specific columns from CSV."""
        # Test with column names
        chef = CSVCleanerChef(extract_columns=["name", "city"])
        csv_str = "name,age,city\nJohn,30,New York\nJane,25,Boston"
        result = chef.preprocess(csv_str)
        
        self.assertIn("name: John", result)
        self.assertIn("city: New York", result)
        self.assertNotIn("age: 30", result)
        
        # Test with column indices
        chef = CSVCleanerChef(extract_columns=[0, 2])
        csv_str = "name,age,city\nJohn,30,New York\nJane,25,Boston"
        result = chef.preprocess(csv_str)
        
        self.assertIn("name: John", result)
        self.assertIn("city: New York", result)
        self.assertNotIn("age: 30", result)
        
    def test_skip_lines(self):
        """Test skipping initial lines in CSV."""
        chef = CSVCleanerChef(skip_lines=1)
        csv_str = "This is a header comment\nname,age,city\nJohn,30,New York"
        result = chef.preprocess(csv_str)
        
        self.assertIn("name: John", result)
        self.assertIn("age: 30", result)
        self.assertIn("city: New York", result)
        
    def test_no_join_columns(self):
        """Test returning CSV as JSON without joining columns."""
        chef = CSVCleanerChef(join_columns=False)
        csv_str = "name,age,city\nJohn,30,New York\nJane,25,Boston"
        result = chef.preprocess(csv_str)
        
        # Result should be a JSON array of row objects
        parsed = json.loads(result)
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["name"], "John")
        self.assertEqual(parsed[0]["age"], "30")
        self.assertEqual(parsed[1]["name"], "Jane")
        self.assertEqual(parsed[1]["city"], "Boston")


if __name__ == "__main__":
    unittest.main() 