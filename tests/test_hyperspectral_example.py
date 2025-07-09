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
import importlib.util
from pathlib import Path
from semantiva_imaging.data_types import SingleChannelImage
from semantiva_imaging.probes.probes import BasicImageProbe

# Dynamically load the example module
spec = importlib.util.spec_from_file_location(
    "hyperspectral_example",
    Path(__file__).parent.parent / "examples" / "hyperspectral_example.py",
)
assert spec is not None
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_stack_and_dimensions():
    cube = np.random.rand(1, 2, 2, 4).astype(np.float32)
    info = [f"b{i}" for i in range(4)]
    stack = module.NChannelImageStack(cube, info)
    assert stack.data.shape == (1, 2, 2, 4)
    assert list(stack.channel_info) == info


def test_module_objects_exist():
    assert hasattr(module, "img") and hasattr(module, "stack")
    assert isinstance(module.img, module.NChannelImage)
    assert isinstance(module.stack, module.NChannelImageStack)


def test_basic_probe_per_band():
    probe = BasicImageProbe()
    img = module.img
    # For each spectral band, verify BasicImageProbe stats match manual computation
    for idx, name in enumerate(img.channel_info):
        data = img.data[..., idx]
        single = SingleChannelImage(data)
        result = probe.process(single)
        expected = {
            "mean": float(data.mean()),
            "sum": float(data.sum()),
            "min": float(data.min()),
            "max": float(data.max()),
        }
        assert result == expected, f"Mismatch for band {name}: {result} vs {expected}"
