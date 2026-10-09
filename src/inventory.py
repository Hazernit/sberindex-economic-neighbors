"""Inspect original files and ZIP members without modifying or extracting them."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(directory):
    records = []
    for path in sorted(Path(directory).rglob('*')):
        if not path.is_file() or path.name == '.gitkeep':
            continue
        record = {'file': str(path.relative_to(directory)),
                  'bytes': path.stat().st_size, 'sha256': sha256(path)}
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                record['members'] = [
                    {'name': entry.filename, 'bytes': entry.file_size,
                     'compressed_bytes': entry.compress_size}
                    for entry in archive.infolist() if not entry.is_dir()
                ]
        records.append(record)
    return {'status': 'inputs_present' if records else 'awaiting_official_data',
            'files': records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.directory.is_dir():
        parser.error('Input directory does not exist')
    result = inventory(args.directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(result['status'])


if __name__ == '__main__':
    main()
