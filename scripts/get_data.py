"""
Re-download the Ames Housing dataset from its authoritative source.

You do NOT need to run this. `data/ames.csv` is already in this repository.
This script exists so the data's provenance is reproducible rather than
"a CSV someone put on the internet".

Source
------
Dean De Cock (2011), "Ames, Iowa: Alternative to the Boston Housing Data as an
End of Semester Regression Project", Journal of Statistics Education 19(3).

The tidied copy used here is the `ames_raw` table from the `AmesHousing` R
package maintained by Max Kuhn: https://github.com/topepo/AmesHousing

2,930 residential sales recorded by the Ames City Assessor's Office, 2006-2010,
described by 82 columns.

Usage
-----
    pip install pyreadr
    python scripts/get_data.py
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = "https://github.com/topepo/AmesHousing.git"
OUT = Path(__file__).resolve().parent.parent / "data" / "ames.csv"


def main() -> None:
    try:
        import pyreadr
    except ImportError:
        sys.exit("Missing dependency. Run:  pip install pyreadr")

    OUT.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        clone = Path(tmp) / "AmesHousing"
        print(f"Cloning {REPO} ...")
        subprocess.run(
            ["git", "clone", "--depth", "1", "--quiet", REPO, str(clone)],
            check=True,
        )

        rda = clone / "data" / "ames_raw.rda"
        if not rda.exists():
            sys.exit(f"Expected {rda} in the cloned repository but it is not there.")

        print("Converting the R data file to CSV ...")
        frame = pyreadr.read_r(str(rda))["ames_raw"]
        frame.to_csv(OUT, index=False)

    rows, cols = frame.shape
    print(f"Wrote {OUT}  ({rows:,} rows x {cols} columns)")
    if (rows, cols) != (2930, 82):
        print("WARNING: expected 2,930 rows x 82 columns. The upstream data may have changed.")


if __name__ == "__main__":
    main()
