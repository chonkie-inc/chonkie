import logging
from typing import List, Any

# -----------------------------
# Logging setup
# -----------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')


class BaseChunker:
    def __init__(self, data: List[Any] = None):
        """
        Initialize the BaseChunker with optional data
        """
        self.data = data or []
        logger.info(f"BaseChunker initialized with {len(self.data)} items")

    def add_data(self, new_data: List[Any]):
        """
        Add new data to the chunker
        """
        if not new_data:
            logger.warning("No new data provided to add")
            return
        self.data.extend(new_data)
        logger.info(f"Added {len(new_data)} items. Total data count: {len(self.data)}")

    def chunk(self, size: int = 10) -> List[List[Any]]:
        """
        Split data into chunks of given size
        """
        if not self.data:
            logger.warning("No data to chunk!")
            return []

        chunks = [self.data[i:i + size] for i in range(0, len(self.data), size)]
        logger.info(f"Created {len(chunks)} chunks with size {size}")
        return chunks


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    example_data = list(range(1, 21))  # Example data
    chunker = BaseChunker(data=example_data)
    chunker.add_data([21, 22, 23])  # Add more data
    result = chunker.chunk(size=5)  # Chunk into size 5
    logger.info(f"Chunks result: {result}")
