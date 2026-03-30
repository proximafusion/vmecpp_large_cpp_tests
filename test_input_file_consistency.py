# SPDX-FileCopyrightText: 2024-present Proxima Fusion GmbH <info@proximafusion.com>
#
# SPDX-License-Identifier: MIT
"""Tests for consistency between main and large_cpp_tests input files.

The large_cpp_tests test_data contains extended versions of the main repo's
input files, with additional dump_* diagnostic flags. After conversion via
indata_to_json (which strips non-standard fields), the JSON output must be
identical.

Run via: pytest src/vmecpp/cpp/vmecpp_large_cpp_tests/test_input_file_consistency.py -v
"""

import json
import tempfile
from pathlib import Path

import pytest

from vmecpp import _util

REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent
MAIN_TEST_DATA = REPO_ROOT / "src" / "vmecpp" / "cpp" / "vmecpp" / "test_data"
LARGE_TEST_DATA = (
    REPO_ROOT / "src" / "vmecpp" / "cpp" / "vmecpp_large_cpp_tests" / "test_data"
)


def _get_input_files(directory: Path) -> set[str]:
    """Return the set of input.* filenames in the given directory."""
    return {p.name for p in directory.glob("input.*")}


def test_no_extra_input_files_in_large_tests():
    """Every large tests input file must also exist in the main repo.

    The main repo may contain additional input files that are not (yet) in
    the large tests repo, but the reverse is not allowed.
    """
    main_files = _get_input_files(MAIN_TEST_DATA)
    large_files = _get_input_files(LARGE_TEST_DATA)

    extra = large_files - main_files
    assert not extra, (
        f"Large tests input files not present in main repo: {sorted(extra)}. "
        f"Add them to vmecpp/test_data/ or remove them from "
        f"vmecpp_large_cpp_tests/test_data/."
    )


def _discover_large_test_input_files() -> list[str]:
    """Discover input files in the large tests directory."""
    return sorted(_get_input_files(LARGE_TEST_DATA))


@pytest.mark.parametrize("input_file", _discover_large_test_input_files())
def test_input_files_produce_identical_json(input_file: str):
    """Input files in both repos must produce identical JSON after conversion."""
    with tempfile.TemporaryDirectory() as tmpdir:
        main_json_path = _util.indata_to_json(
            MAIN_TEST_DATA / input_file,
            output_override=Path(tmpdir) / "main.json",
        )
        large_json_path = _util.indata_to_json(
            LARGE_TEST_DATA / input_file,
            output_override=Path(tmpdir) / "large.json",
        )

        main_data = json.loads(main_json_path.read_text())
        large_data = json.loads(large_json_path.read_text())

    assert main_data == large_data, (
        f"JSON mismatch for {input_file}. "
        f"Keys only in main: {set(main_data) - set(large_data)}, "
        f"Keys only in large: {set(large_data) - set(main_data)}"
    )
