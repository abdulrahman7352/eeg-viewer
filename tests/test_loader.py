import pytest
from eeg_viewer.loader import EDFLoader

def test_loader_raises_on_missing_file():
    with pytest.raises(FileNotFoundError):
        loader = EDFLoader("nonexistent.edf")
