import os
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent
source_dir = BASE / "raw_crease_patterns"
cp_output = BASE / "dataset" / "cp_files"
png_output = BASE / "dataset" / "png_files"

os.makedirs(cp_output, exist_ok=True)
os.makedirs(png_output, exist_ok=True)

for root, dirs, files in os.walk(source_dir):
    for file in files:
        src = os.path.join(root, file)
        if file.endswith(".cp"):
            shutil.copy2(src, os.path.join(cp_output, file))
        elif file.endswith(".png") or file.endswith(".jpg"):
            shutil.copy2(src, os.path.join(png_output, file))

print(f"CP files: {len(os.listdir(cp_output))}")
print(f"PNG files: {len(os.listdir(png_output))}")
