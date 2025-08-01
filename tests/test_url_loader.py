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

import os
import threading
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

import numpy as np
import pytest

from semantiva_imaging.data_types import SingleChannelImage
from semantiva_imaging.data_io import (
    PngSingleChannelImageSaver,
    PngSingleChannelImageLoader,
)
from semantiva_imaging.data_io.url_loader import UrlLoader


@pytest.fixture
def http_image_server(tmp_path):
    img = SingleChannelImage(np.random.randint(0, 255, (5, 5), dtype=np.uint8))
    fname = tmp_path / "img.png"
    PngSingleChannelImageSaver().send_data(img, str(fname))

    cwd = os.getcwd()
    os.chdir(tmp_path)
    server = ThreadingHTTPServer(("localhost", 0), SimpleHTTPRequestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    port = server.server_address[1]
    yield f"http://localhost:{port}/img.png", img

    server.shutdown()
    thread.join()
    os.chdir(cwd)


def test_url_loader_success(http_image_server):
    url, original = http_image_server
    Loader = UrlLoader(PngSingleChannelImageLoader)
    loaded = Loader.get_data(url)
    assert np.array_equal(loaded.data, original.data)


def test_url_loader_invalid_scheme():
    Loader = UrlLoader(PngSingleChannelImageLoader)
    with pytest.raises(ValueError):
        Loader.get_data("ftp://example.com/img.png")


def test_url_loader_size_limit(http_image_server):
    url, _ = http_image_server
    Loader = UrlLoader(PngSingleChannelImageLoader, max_bytes=10)
    with pytest.raises(ValueError):
        Loader.get_data(url)
