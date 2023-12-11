import pathlib
import os

import pytest


def pytest_configure():
    this_dir = pathlib.Path(__file__).parent
    pytest.EXAMPLE_PATH = os.path.abspath(str(this_dir.joinpath("example")))
    pytest.TEMP_PATH = os.path.abspath(str(this_dir.parent.joinpath(".pytest_temp")))
    if not os.path.exists(pytest.TEMP_PATH):
        os.mkdir(pytest.TEMP_PATH)


class Helpers:
    
    @staticmethod
    def remove_files(*files):
        for file in files:
            if os.path.exists(file):
                os.remove(file)

@pytest.fixture(scope="session")
def helpers():
    return Helpers


@pytest.fixture(scope="session")
def zinc1k():
    return os.path.join(pytest.EXAMPLE_PATH, "zinc1k.csv")


@pytest.fixture(scope="session")
def smiles_batch():
    return ["CC(=O)OC1=CC=CC=C1C(=O)O", "CCO", "CC(=O)O"]