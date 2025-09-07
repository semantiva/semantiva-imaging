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

# Ensure registry initialization for tests to avoid import-time side-effects.


def pytest_sessionstart(session):
    """Called after the `Session` object has been created and before performing collection.

    We initialize the ClassRegistry here so tests that depend on registered
    components (including semantiva-imaging loaders/savers) work even when
    the application CLI usually performs registration at runtime.
    """
    try:
        # Import locally from the semantiva package; it's expected to be available
        # on sys.path when running the tests from the repository.
        from semantiva.registry.class_registry import ClassRegistry

        ClassRegistry.initialize_default_modules()
    except Exception:
        # Don't fail tests if semantiva isn't importable here; let the test run and
        # surface import errors normally.
        pass
