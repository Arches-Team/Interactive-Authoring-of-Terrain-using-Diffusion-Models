import os
import zipfile

# === CONFIG ===
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
ADDON_NAME = "terrain_diffusion"
OUTPUT_ZIP = os.path.join(PROJECT_ROOT, "diffusion_terrain.zip")

INCLUDE_PATHS = [
    "models",
    "app",
    "__init__.py",
    "terrain_diffusion",
]

EXCLUDE_PATTERNS = [
    "__pycache__",
    ".git",
]

def should_exclude(path):
    return any(pattern in path for pattern in EXCLUDE_PATTERNS)


def zip_project():
    with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for item in INCLUDE_PATHS:
            full_path = os.path.join(PROJECT_ROOT, item)

            if not os.path.exists(full_path):
                print(f"Skipping missing: {item}")
                continue

            if os.path.isfile(full_path):
                arcname = os.path.join(ADDON_NAME, item)
                zipf.write(full_path, arcname)
                print(f"Added file: {arcname}")

            else:
                for root, dirs, files in os.walk(full_path):
                    dirs[:] = [d for d in dirs if not should_exclude(d)]

                    for file in files:
                        file_path = os.path.join(root, file)

                        if should_exclude(file_path):
                            continue

                        rel_path = os.path.relpath(file_path, PROJECT_ROOT)
                        arcname = os.path.join(ADDON_NAME, rel_path)

                        zipf.write(file_path, arcname)
                        print(f"Added: {arcname}")

    print(f"ZIP created at: {OUTPUT_ZIP}")



if __name__ == "__main__":
    zip_project()