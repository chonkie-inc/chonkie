"""Structured data preprocessing chef implementations."""

import importlib.util as importutil
import json
import re
from typing import Any, Dict, List, Optional, Callable, Union, Tuple

from .base import BaseChef


class JSONCleanerChef(BaseChef):
    """Chef for processing JSON data."""
    
    def __init__(
        self,
        extract_fields: Optional[List[str]] = None,
        flatten: bool = False,
        join_text_fields: bool = True,
        join_separator: str = " ",
        handle_errors: bool = True,
    ):
        """Initialize the JSONCleanerChef.
        
        Args:
            extract_fields: List of specific fields to extract from the JSON. If None, all fields are included.
            flatten: Whether to flatten nested JSON objects.
            join_text_fields: Whether to join all text fields into a single text.
            join_separator: The separator to use when joining text fields.
            handle_errors: Whether to handle JSON parsing errors gracefully.
        """
        super().__init__()
        self.extract_fields = extract_fields
        self.flatten = flatten
        self.join_text_fields = join_text_fields
        self.join_separator = join_separator
        self.handle_errors = handle_errors
        
    def is_available(self) -> bool:
        """Check if the chef is available.
        
        This chef has no external dependencies, so it's always available.
        
        Returns:
            bool: True
        """
        return True
    
    def _flatten_json(self, json_obj: Union[Dict, List], separator: str = "_") -> Dict:
        """Flatten a nested JSON object into a flat dictionary with compound keys.
        
        Args:
            json_obj: The JSON object to flatten.
            separator: The separator to use for compound keys.
            
        Returns:
            Dict: The flattened JSON object.
        """
        out = {}
        
        def _flatten(x: Any, name: str = ""):
            if isinstance(x, dict):
                for key, value in x.items():
                    _flatten(value, f"{name}{separator}{key}" if name else key)
            elif isinstance(x, list):
                for i, item in enumerate(x):
                    _flatten(item, f"{name}{separator}{i}" if name else str(i))
            else:
                out[name] = x
                
        _flatten(json_obj)
        return out
    
    def _extract_fields(self, json_obj: Dict, fields: List[str]) -> Dict:
        """Extract specific fields from a JSON object.
        
        Args:
            json_obj: The JSON object to extract fields from.
            fields: The fields to extract.
            
        Returns:
            Dict: The extracted fields.
        """
        result = {}
        
        for field in fields:
            # Handle nested fields (e.g., "user.name")
            parts = field.split(".")
            value = json_obj
            
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    value = None
                    break
                    
            if value is not None:
                result[field] = value
                
        return result
        
    def _join_text_fields(self, json_obj: Dict) -> str:
        """Join all text fields in a JSON object into a single string.
        
        Args:
            json_obj: The JSON object to join text fields from.
            
        Returns:
            str: The joined text.
        """
        text_values = []
        
        for key, value in json_obj.items():
            if isinstance(value, str):
                text_values.append(f"{key}: {value}")
            elif isinstance(value, (int, float, bool)):
                text_values.append(f"{key}: {str(value)}")
            elif isinstance(value, dict):
                # Handle nested objects by serializing them to a JSON string
                text_values.append(f"{key}: {json.dumps(value)}")
            elif isinstance(value, list):
                # Handle arrays by joining their string representation
                list_str = ", ".join(str(item) for item in value)
                text_values.append(f"{key}: [{list_str}]")
                
        return self.join_separator.join(text_values)
    
    def preprocess(self, text: str) -> str:
        """Process JSON data.
        
        Args:
            text: The JSON text to process.
            
        Returns:
            str: The processed text.
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string.")
            
        # Return quickly if the input is empty
        if not text.strip():
            return ""
            
        try:
            # Parse JSON
            json_obj = json.loads(text)
            
            # Flatten if needed
            if self.flatten:
                json_obj = self._flatten_json(json_obj)
                
            # Extract specific fields if specified
            if self.extract_fields:
                json_obj = self._extract_fields(json_obj, self.extract_fields)
                
            # Join text fields if specified
            if self.join_text_fields:
                return self._join_text_fields(json_obj)
            else:
                # Return the JSON as a formatted string
                return json.dumps(json_obj, ensure_ascii=False, indent=2)
                
        except json.JSONDecodeError as e:
            if self.handle_errors:
                return f"Error parsing JSON: {str(e)}"
            else:
                raise
    
    def __repr__(self) -> str:
        """Return the string representation of the chef."""
        return (
            f"JSONCleanerChef(extract_fields={self.extract_fields}, "
            f"flatten={self.flatten}, join_text_fields={self.join_text_fields}, "
            f"join_separator='{self.join_separator}', handle_errors={self.handle_errors})"
        )


class CSVCleanerChef(BaseChef):
    """Chef for processing CSV data."""
    
    def __init__(
        self,
        delimiter: str = ",",
        extract_columns: Optional[List[Union[str, int]]] = None,
        has_header: bool = True,
        skip_lines: int = 0,
        join_columns: bool = True,
        join_separator: str = " ",
        handle_errors: bool = True,
    ):
        """Initialize the CSVCleanerChef.
        
        Args:
            delimiter: The delimiter character used in the CSV.
            extract_columns: List of column names or indices to extract. If None, all columns are included.
            has_header: Whether the CSV has a header row.
            skip_lines: Number of initial lines to skip.
            join_columns: Whether to join all columns into a single text.
            join_separator: The separator to use when joining columns.
            handle_errors: Whether to handle CSV parsing errors gracefully.
        """
        super().__init__()
        self.delimiter = delimiter
        self.extract_columns = extract_columns
        self.has_header = has_header
        self.skip_lines = skip_lines
        self.join_columns = join_columns
        self.join_separator = join_separator
        self.handle_errors = handle_errors
        self._csv_available = self._check_dependencies()
        
    def _check_dependencies(self) -> bool:
        """Check if csv is available (it's part of standard library, but check anyway)."""
        return importutil.find_spec("csv") is not None
        
    def _import_dependencies(self) -> None:
        """Import csv."""
        if self._csv_available:
            global csv
            import csv
        else:
            raise ImportError("csv module is not available.")
    
    def is_available(self) -> bool:
        """Check if the chef is available.
        
        Returns:
            bool: True if the chef dependencies are available, False otherwise.
        """
        return self._csv_available
    
    def _extract_rows(
        self, reader: Any, columns: List[Union[str, int]], has_header: bool
    ) -> List[Dict[str, str]]:
        """Extract specific columns from CSV rows.
        
        Args:
            reader: CSV reader object.
            columns: The columns to extract (either names or indices).
            has_header: Whether the CSV has a header row.
            
        Returns:
            List[Dict[str, str]]: The extracted data.
        """
        rows = []
        headers = []
        
        # Read header row if present
        if has_header:
            try:
                headers = next(reader)
            except StopIteration:
                return []
        
        # Process each row
        for row in reader:
            if has_header:
                # If we have headers, match column names
                row_dict = {header: value for header, value in zip(headers, row)}
                
                if all(isinstance(col, str) for col in columns):
                    # Filter by column names
                    extracted = {col: row_dict.get(col, "") for col in columns if col in row_dict}
                else:
                    # Filter by column indices
                    extracted = {
                        headers[i] if i < len(headers) else f"col_{i}": row[i] 
                        for i in columns if i < len(row)
                    }
            else:
                # Without headers, we can only extract by index
                extracted = {f"col_{i}": row[i] for i in columns if i < len(row)}
                
            rows.append(extracted)
            
        return rows
    
    def _join_row(self, row: Dict[str, str]) -> str:
        """Join a row into a single string.
        
        Args:
            row: The row to join.
            
        Returns:
            str: The joined row.
        """
        joined = []
        for key, value in row.items():
            if value:  # Only include non-empty values
                joined.append(f"{key}: {value}")
        
        return self.join_separator.join(joined)
    
    def preprocess(self, text: str) -> str:
        """Process CSV data.
        
        Args:
            text: The CSV text to process.
            
        Returns:
            str: The processed text.
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string.")
            
        # Return quickly if the input is empty
        if not text.strip():
            return ""
            
        try:
            # Import CSV module
            self._import_dependencies()
            
            # Split text into lines
            lines = text.splitlines()
            
            # Skip initial lines if needed
            if self.skip_lines > 0:
                lines = lines[self.skip_lines:]
                
            # Create CSV reader
            reader = csv.reader(lines, delimiter=self.delimiter)
            
            # Extract specific columns if specified
            if self.extract_columns:
                rows = self._extract_rows(reader, self.extract_columns, self.has_header)
            else:
                # Read all columns
                if self.has_header:
                    try:
                        headers = next(reader)
                        rows = [dict(zip(headers, row)) for row in reader]
                    except StopIteration:
                        return ""
                else:
                    rows = [
                        {f"col_{i}": value for i, value in enumerate(row)}
                        for row in reader
                    ]
            
            # Join rows if specified
            if self.join_columns:
                # Join each row, then join all rows with newlines
                return "\n".join(self._join_row(row) for row in rows)
            else:
                # Return as JSON
                return json.dumps(rows, ensure_ascii=False, indent=2)
                
        except Exception as e:
            if self.handle_errors:
                return f"Error processing CSV: {str(e)}"
            else:
                raise
    
    def __repr__(self) -> str:
        """Return the string representation of the chef."""
        return (
            f"CSVCleanerChef(delimiter='{self.delimiter}', "
            f"extract_columns={self.extract_columns}, has_header={self.has_header}, "
            f"skip_lines={self.skip_lines}, join_columns={self.join_columns}, "
            f"join_separator='{self.join_separator}', handle_errors={self.handle_errors})"
        ) 