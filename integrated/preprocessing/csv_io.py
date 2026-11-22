"""CSV reader shared by the energy-label extraction adapter."""
import csv
from pathlib import Path

def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))
