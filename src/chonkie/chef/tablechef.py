"""TableChef is a chef that processes tabular data from files (e.g., CSV, Excel) and markdown strings."""

import re
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, List, Union

from .base import BaseChef

if TYPE_CHECKING:
    import pandas as pd


class TableChef(BaseChef):
    """TableChef processes CSV files and returns pandas DataFrames."""

    table_pattern = re.compile(r"(\|.*?\n(?:\|[-: ]+\|.*?\n)?(?:\|.*?\n)+)")

    def _lazy_import_pandas(self):
        global pd
        import pandas as pd

    def process(
        self, path: Union[str, Path], **kwargs
    ) -> Union["pd.DataFrame", List["pd.DataFrame"], None]:
        """Process a CSV file and return a pandas DataFrame.

        Args:
            path (Union[str, Path]): Path to the CSV file.
            **kwargs: Additional keyword arguments for pandas.read_csv.

        Returns:
            pandas.DataFrame: DataFrame containing the CSV data.

        """
        self._lazy_import_pandas()
        # if file exists
        if Path(path).is_file():
            str_path = str(path)
            if str_path.endswith(".csv"):
                return pd.read_csv(str_path, **kwargs)
            elif str_path.endswith(".xls") or str_path.endswith(".xlsx"):
                return pd.read_excel(str_path, **kwargs)
        # string is a markedown table
        else:
            table_mds = self.extract_tables_from_markdown(str(path))
            # no tables
            if len(table_mds) == 0:
                return None
            # one table
            elif len(table_mds) == 1:
                df = pd.read_csv(
                    StringIO(table_mds[0]), sep="|", header=0, skipinitialspace=True
                )
                df = df.iloc[
                    1:, 1:-1
                ]  # remove first and last empty columns and first row
            # multiple tables
            elif len(table_mds) > 1:
                out = []
                for table_md in table_mds:
                    df = pd.read_csv(
                        StringIO(table_md), sep="|", header=0, skipinitialspace=True
                    )
                    out.append(
                        df.iloc[
                            1:, 1:-1
                        ]  # remove first and last empty columns and first row
                    )
                return out

    def process_batch(
        self, paths: Union[List[str], List[Path]], **kwargs
    ) -> List["pd.DataFrame"]:
        """Process multiple CSV files and return a list of DataFrames.

        Args:
            paths (Union[List[str], List[Path]]): Paths to the CSV files.
            **kwargs: Additional keyword arguments for pandas.read_csv.

        Returns:
            List[pd.DataFrame]: List of DataFrames.

        """
        return [self.process(path, **kwargs) for path in paths]

    def __call__(
        self, path: Union[str, Path, List[str], List[Path]], **kwargs
    ) -> Union[
        "pd.DataFrame",
        List["pd.DataFrame"],
        None,
        List[Union["pd.DataFrame", List["pd.DataFrame"], None]],
    ]:
        """Process one or more CSV files and return DataFrame(s)."""
        if isinstance(path, (list, tuple)):
            return self.process_batch(path, **kwargs)
        elif isinstance(path, (str, Path)):
            return self.process(path, **kwargs)
        else:
            raise TypeError(f"Unsupported type: {type(path)}")

    def extract_tables_from_markdown(self, markdown: str) -> List[str]:
        """Extract markdown tables from a markdown string.

        Args:
            markdown (str): The markdown text containing tables.

        Returns:
            List[str]: A list of strings, each representing a markdown table found in the input.

        """
        tables: List[str] = []
        for match in self.table_pattern.finditer(markdown):
            tables.append(match.group(0))
        return tables

    def __repr__(self) -> str:
        """Return a string representation of the chef."""
        return f"{self.__class__.__name__}()"
