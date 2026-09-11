"""Published split identity readers."""
from pathlib import Path
import csv

def read_ids(path):
    p=Path(path)
    if p.suffix.lower()=='.csv':
        with p.open(newline='',encoding='utf-8-sig') as f:return [r['segment'] for r in csv.DictReader(f)]
    return [line.strip() for line in p.read_text().splitlines() if line.strip()]
