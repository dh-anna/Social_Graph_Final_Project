import hashlib
from datetime import datetime
from pathlib import Path


def get_md5_hash(file_path: Path, chunk_size: int = 8192) -> str:
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def format_size(size_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"


def get_creation_time(file_path: Path) -> datetime:
    stat = file_path.stat()
    # On macOS, st_birthtime is the creation time
    # On Linux, st_ctime is the metadata change time (closest to creation)
    if hasattr(stat, "st_birthtime"):
        return datetime.fromtimestamp(stat.st_birthtime)
    return datetime.fromtimestamp(stat.st_ctime)


def hash_datasets(datasets_folder: str = "Datasets") -> None:
    datasets_path = Path(datasets_folder)

    if not datasets_path.exists():
        print(f"Error: Folder '{datasets_folder}' does not exist.")
        return

    if not datasets_path.is_dir():
        print(f"Error: '{datasets_folder}' is not a directory.")
        return

    files = sorted(datasets_path.rglob("*"))
    file_count = 0

    for file_path in files:
        if file_path.is_file():
            file_count += 1

            # Get file info
            file_name = file_path.name
            relative_path = file_path.relative_to(datasets_path)
            size = file_path.stat().st_size
            created_at = get_creation_time(file_path)
            md5_hash = get_md5_hash(file_path)

            print(f"\nFile #{file_count}")
            print("-" * 50)
            print(f"  Name:       {file_name}")
            print(f"  Path:       {relative_path}")
            print(f"  Size:       {format_size(size)} ({size:,} bytes)")
            print(f"  Created:    {created_at.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  MD5 Hash:   {md5_hash}")