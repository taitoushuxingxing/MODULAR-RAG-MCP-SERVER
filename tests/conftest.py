"""Pytest configuration and shared fixtures.

This module contains pytest configuration and fixtures that are shared
across all test modules.
"""

import sys
from pathlib import Path
import pytest

#add project root to the Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT

@pytest.fixture
def sample_document_dir(project_root: Path) -> Path:
    return project_root / "tests" / "fixtures" / "sample_documents"

@pytest.fixture
def config_dir(project_root: Path) -> Path:
    return project_root / "config"
