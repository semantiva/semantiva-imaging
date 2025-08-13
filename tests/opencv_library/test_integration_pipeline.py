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

import numpy as np
import cv2
from semantiva.registry.plugin_registry import load_extensions
from semantiva.pipeline import Pipeline
from semantiva.context_processors.context_types import ContextType
from semantiva.pipeline.payload import Payload

from semantiva_imaging.data_types import RGBImage


def test_pipeline_smoke():
    load_extensions("semantiva_imaging")
    img_arr = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
    img = RGBImage(img_arr)
    pipeline = Pipeline(
        [
            {
                "processor": "GaussianBlurRGBImageProcessor",
                "parameters": {
                    "ksize": (3, 3),
                    "sigmaX": 0,
                    "sigmaY": 0,
                    "borderType": None,
                    "dst": None,
                    "hint": None,
                },
            },
            {
                "processor": "ResizeRGBImageProcessor",
                "parameters": {
                    "dsize": (20, 20),
                    "dst": None,
                    "fx": None,
                    "fy": None,
                    "interpolation": None,
                },
            },
            {
                "processor": "RGB2SingleChannelImageProcessor",
                "parameters": {
                    "code": cv2.COLOR_RGB2GRAY,
                    "dst": None,
                    "dstCn": None,
                    "hint": None,
                },
            },
            {
                "processor": "CannyEdgeSingleChannelImageProcessor",
                "parameters": {
                    "threshold1": 50,
                    "threshold2": 100,
                    "edges": None,
                    "apertureSize": None,
                    "L2gradient": None,
                },
            },
        ]
    )
    payload_out = pipeline.process(Payload(img, ContextType()))
    out = payload_out.data
    assert out.data.shape == (20, 20)
    assert out.data.mean() >= 0
