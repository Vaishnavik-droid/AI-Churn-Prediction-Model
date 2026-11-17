"""
Script to execute notebooks from cell 0 to end using NotebookClient,
saving executed outputs and verifying zero errors.
"""

from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"
notebooks = [
    "01_data_cleaning.ipynb",
    "02_exploratory_analysis.ipynb",
    "03_feature_engineering.ipynb"
]

for nb_name in notebooks:
    nb_path = NOTEBOOKS_DIR / nb_name
    print(f"Executing {nb_name}...")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbf.read(f, as_version=4)
    
    # Execute notebook with venv kernel
    client = NotebookClient(nb, timeout=600, kernel_name="venv", resources={"metadata": {"path": str(NOTEBOOKS_DIR)}})
    client.execute()
    
    # Write back notebook with executed outputs
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    
    print(f"Successfully executed and saved: {nb_name}")

print("\nALL NOTEBOOKS EXECUTED CLEANLY!")
