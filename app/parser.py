"""Parser module for processing and parsing data."""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class Parser:
    """Base parser class for handling data parsing operations."""

    def __init__(self) -> None:
        """Initialize the parser."""
        self.data: Dict[str, Any] = {}

    def parse(self, content: str) -> Dict[str, Any]:
        """Parse the given content.

        Args:
            content: The content to parse.

        Returns:
            A dictionary containing parsed data.
        """
        return {"content": content}

    def parse_json(self, content: str) -> Dict[str, Any]:
        """Parse JSON content.

        Args:
            content: JSON string to parse.

        Returns:
            Parsed JSON as dictionary.

        Raises:
            ValueError: If content is not valid JSON.
        """
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON content: {e}") from e

    def parse_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Parse content from a file.

        Args:
            file_path: Path to the file to parse.

        Returns:
            Parsed file content.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If file content is invalid.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        content = path.read_text(encoding="utf-8")

        if path.suffix == ".json":
            return self.parse_json(content)

        return self.parse(content)

    def extract_fields(
        self, content: str, fields: List[str]
    ) -> Dict[str, Optional[str]]:
        """Extract specific fields from content.

        Args:
            content: Content to extract fields from.
            fields: List of field names to extract.

        Returns:
            Dictionary mapping field names to extracted values.
        """
        result: Dict[str, Optional[str]] = {}
        for field in fields:
            pattern = rf"{field}:\s*(.+?)(?:\n|$)"
            match = re.search(pattern, content, re.IGNORECASE)
            result[field] = match.group(1).strip() if match else None
        return result

    def validate(self, data: Dict[str, Any], required_fields: List[str]) -> bool:
        """Validate that required fields are present in data.

        Args:
            data: Data dictionary to validate.
            required_fields: List of required field names.

        Returns:
            True if all required fields are present, False otherwise.
        """
        return all(field in data for field in required_fields)

    def clean_text(self, text: str) -> str:
        """Clean and normalize text content.

        Args:
            text: Text to clean.

        Returns:
            Cleaned text.
        """
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text


class TextParser(Parser):
    """Parser specialized for text content."""

    def parse(self, content: str) -> Dict[str, Any]:
        """Parse text content into structured data.

        Args:
            content: Text content to parse.

        Returns:
            Dictionary with parsed text data.
        """
        cleaned = self.clean_text(content)
        words = cleaned.split()
        lines = content.strip().split("\n")

        return {
            "content": cleaned,
            "word_count": len(words),
            "line_count": len(lines),
            "char_count": len(content),
        }

    def parse_key_value(self, content: str, separator: str = "=") -> Dict[str, str]:
        """Parse key-value pairs from text.

        Args:
            content: Text containing key-value pairs.
            separator: Character separating keys from values.

        Returns:
            Dictionary of parsed key-value pairs.
        """
        result: Dict[str, str] = {}
        for line in content.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if separator in line:
                key, value = line.split(separator, 1)
                result[key.strip()] = value.strip()
        return result


class DataParser(Parser):
    """Parser for structured data formats."""

    def parse_csv_line(self, line: str, delimiter: str = ",") -> List[str]:
        """Parse a single CSV line.

        Args:
            line: CSV line to parse.
            delimiter: Field delimiter.

        Returns:
            List of field values.
        """
        return [field.strip() for field in line.split(delimiter)]

    def parse_csv(self, content: str, delimiter: str = ",") -> List[Dict[str, str]]:
        """Parse CSV content.

        Args:
            content: CSV content to parse.
            delimiter: Field delimiter.

        Returns:
            List of dictionaries representing rows.
        """
        lines = content.strip().split("\n")
        if not lines:
            return []

        headers = self.parse_csv_line(lines[0], delimiter)
        result: List[Dict[str, str]] = []

        for line in lines[1:]:
            if line.strip():
                values = self.parse_csv_line(line, delimiter)
                row = dict(zip(headers, values))
                result.append(row)

        return result
