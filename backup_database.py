#!/usr/bin/env python3
import shutil
import os
from datetime import datetime
from pathlib import Path


def backup_database():
    """Backup Milvus database and images"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path(f"backups/backup_{timestamp}")
    backup_dir.mkdir(parents=True, exist_ok=True)

    print(f"📦 Creating backup: {backup_dir}")

    # Backup Milvus database file
    if os.path.exists("milvus_demo.db"):
        shutil.copy2("milvus_demo.db", backup_dir / "milvus_demo.db")
        print(f"✅ Backed up database")

    # Backup images
    if os.path.exists("data/images") and os.path.isdir("data/images"):
        shutil.copytree("data/images", backup_dir / "images")
        image_count = len(list((backup_dir / "images").glob("*")))
        print(f"✅ Backed up {image_count} images")

    # Backup claims
    if os.path.exists("data/claims.json"):
        shutil.copy2("data/claims.json", backup_dir / "claims.json")
        print(f"✅ Backed up claims")

    # Keep only last 7 backups
    backups = sorted(Path("backups").glob("backup_*"))
    if len(backups) > 7:
        for old_backup in backups[:-7]:
            shutil.rmtree(old_backup)
            print(f"🗑️ Removed old backup: {old_backup.name}")

    print(
        f"✅ Backup complete! Total backups: {len(list(Path('backups').glob('backup_*')))}"
    )


if __name__ == "__main__":
    backup_database()
