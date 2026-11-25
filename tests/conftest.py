from pathlib import Path

import pytest

from bionetgen_tools import BNGLSimulator

ROOT = Path(__file__).resolve().parents[1]
FCERI = ROOT / "models" / "fceri_starter.bngl"
RAS = ROOT / "models" / "RasModel.bngl"


@pytest.fixture(scope="session")
def fceri():
    return BNGLSimulator(FCERI)


@pytest.fixture(scope="session")
def ras():
    return BNGLSimulator(RAS)
