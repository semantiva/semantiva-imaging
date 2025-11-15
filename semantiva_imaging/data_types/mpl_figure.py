# Copyright 2025 Semantiva authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Matplotlib figure data types for Semantiva imaging pipelines."""

from __future__ import annotations
import weakref
from typing import Any, Iterator, List
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from semantiva.data_types import BaseDataType, DataCollectionType


class MatplotlibFigure(BaseDataType):
    """Semantiva data type wrapping a matplotlib Figure with lifecycle safety net.

    This wrapper automatically closes the underlying matplotlib figure when the
    wrapper is garbage-collected, preventing resource leaks. This complements,
    but does not replace, explicit close_after logic in rendering processors.
    """

    def __init__(self, data: Figure) -> None:
        super().__init__(data)
        # Safety net: when this wrapper is GC'd, close the figure.
        # This complements explicit close_after=True in processors.
        self._finalizer = weakref.finalize(self, plt.close, data)

    def validate(self, data: Any) -> bool:
        assert isinstance(
            data, Figure
        ), f"Expected matplotlib.figure.Figure, got {type(data)}"
        return True

    def __str__(self) -> str:
        fig: Figure = self._data
        w, h = fig.get_size_inches()
        return f"{self.__class__.__name__}(dpi={fig.dpi}, size_in={w:.2f}x{h:.2f}, axes={len(fig.axes)})"

    __repr__ = __str__


class MatplotlibFigureCollection(
    DataCollectionType[MatplotlibFigure, List[MatplotlibFigure]]
):
    """Collection of MatplotlibFigure items."""

    @classmethod
    def _initialize_empty(cls) -> List[MatplotlibFigure]:
        return []

    def validate(self, data: Any) -> bool:
        assert isinstance(
            data, list
        ), f"Expected list[MatplotlibFigure], got {type(data)}"
        for i, item in enumerate(data):
            assert isinstance(
                item, MatplotlibFigure
            ), f"Index {i}: expected MatplotlibFigure, got {type(item)}"
        return True

    def __iter__(self) -> Iterator[MatplotlibFigure]:
        """Iterate over the MatplotlibFigure items."""
        return iter(self._data)

    def append(self, item: MatplotlibFigure) -> None:
        """
        Append a MatplotlibFigure to the collection.

        Args:
            item: The MatplotlibFigure to append.

        Raises:
            TypeError: If the item is not a MatplotlibFigure.
        """
        if not isinstance(item, MatplotlibFigure):
            raise TypeError(f"Expected MatplotlibFigure, got {type(item)}")
        self._data.append(item)

    def __len__(self) -> int:
        """Return the number of figures in the collection."""
        return len(self._data)
