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

import pytest
import numpy as np

from semantiva_imaging.data_io.parametric_base import ExpressionEnv


@pytest.fixture
def env():
    return ExpressionEnv()


def test_legitimate_keyword_args(env):
    x = np.linspace(0, 5, 6)
    # Legitimate calls with keyword arguments should evaluate
    out = env.evaluate("clip(x, a_min=1, a_max=3)", x=x)
    assert isinstance(out, np.ndarray)
    assert out.min() >= 1 and out.max() <= 3


@pytest.mark.parametrize(
    "expr",
    [
        "sin(x, where=__import__('os'))",
        "clip(x, min=eval('1+1'), max=2)",
        "sin(__import__('sys').exit(0))",
        "max(x, key=lambda y: __import__('os').system('echo pwned'))",
    ],
)
def test_malicious_keyword_or_nested_calls_blocked(env, expr):
    x = np.linspace(0, 1, 5)
    with pytest.raises(ValueError):
        env.evaluate(expr, x=x)
