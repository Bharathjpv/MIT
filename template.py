import os
from pathlib import Path

list_of_files = [
    f"MIT/__init__.py",
    f"MIT/cloud_storage/__init__.py",
    f"MIT/components/__init__.py",
    f"MIT/constants/__init__.py",
    f"MIT/entity/__init__.py",
    f"MIT/exceptions/__init__.py",
    f"MIT/logger/__init__.py",
    f"MIT/pipeline/__init__.py",
    f"MIT/utils/__init__.py",
    f'setup.py'
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
    else:
        print(f"file is already present at: {filepath}")