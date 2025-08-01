import os
import threading
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

import numpy as np
import pytest

from semantiva_imaging.data_types import SingleChannelImage
from semantiva_imaging.data_io import PngImageSaver, PngImageLoader
from semantiva_imaging.data_io.url_loader import UrlLoader


@pytest.fixture
def http_image_server(tmp_path):
    img = SingleChannelImage(np.random.randint(0, 255, (5, 5), dtype=np.uint8))
    fname = tmp_path / "img.png"
    PngImageSaver().send_data(img, str(fname))

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
    Loader = UrlLoader(PngImageLoader)
    loaded = Loader.get_data(url)
    assert np.array_equal(loaded.data, original.data)


def test_url_loader_invalid_scheme():
    Loader = UrlLoader(PngImageLoader)
    with pytest.raises(ValueError):
        Loader.get_data("ftp://example.com/img.png")


def test_url_loader_size_limit(http_image_server):
    url, _ = http_image_server
    Loader = UrlLoader(PngImageLoader, max_bytes=10)
    with pytest.raises(ValueError):
        Loader.get_data(url)
