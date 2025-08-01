"""Utilities for generating HTTP/HTTPS loaders."""

from __future__ import annotations

import os
import tempfile
from typing import Type
from urllib.parse import urlparse
from urllib.request import urlopen

from semantiva.data_io import DataSource


def UrlLoader(
    loader_cls: Type[DataSource],
    *,
    timeout: float = 10.0,
    max_bytes: int = 10 * 1024 * 1024,
) -> Type[DataSource]:
    """Factory to create a URL-based loader from a file-based loader class.

    Parameters
    ----------
    loader_cls:
        Existing loader class expecting a file path.
    timeout:
        Timeout for HTTP requests in seconds.
    max_bytes:
        Maximum number of bytes to download.

    Returns
    -------
    Type[DataSource]
        A new loader class accepting ``url`` instead of ``path``.
    """

    if not issubclass(loader_cls, DataSource):
        raise TypeError("loader_cls must be a DataSource")

    def _get_data(cls, url: str, **kwargs):
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")

        with urlopen(url, timeout=timeout) as resp:
            data = resp.read(max_bytes + 1)
            if len(data) > max_bytes:
                raise ValueError("Download exceeds maximum allowed size")
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(data)
                temp_path = tmp.name
        try:
            return loader_cls._get_data(temp_path, **kwargs)
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass

    attrs = {
        "_timeout": timeout,
        "_max_bytes": max_bytes,
        "_delegate_cls": loader_cls,
        "_get_data": classmethod(_get_data),
        "__doc__": f"URL-based loader wrapping {loader_cls.__name__}.",
    }

    return type(f"Url{loader_cls.__name__}", (loader_cls,), attrs)
