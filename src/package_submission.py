"""Package the reviewed competition files using a bounded, explicit allowlist.

Run from the project root: python -m src.package_submission
Original datasets are acquired separately from the official source links.
Computer settings, logs and third-party UI code are never included.
"""
from pathlib import Path
import hashlib
import json
import zipfile


ROOT_FILES = [
    '.gitignore', 'README.md', 'LICENSE', 'DATA-LICENSE.md',
    'requirements.txt', 'config.json', 'data/metadata.json',
    'data/processed/excluded.csv', 'data/processed/panel-audit.json',
    'data/processed/municipalities.csv',
    'docs/competition.md', 'docs/metrics.md', 'docs/methodology-design.md',
    'docs/methodological-report.md', 'docs/application-draft.md',
    'notebooks/economic-neighbors.ipynb', 'outputs/input-inventory.json',
    'output/pdf/economic-neighbors-v3.pdf',
    'output/presentation/economic-neighbors-v3.pptx',
]
ANALYSIS_FILES = [
    'comparison.csv', 'dynamics.csv', 'proximity-full.csv',
    'explorer-nodes.json', 'market-associations.csv', 'assignments.csv',
    'selected.json', 'model.npz', 'boundary-nodes.csv',
    'smoothing-sensitivity.csv', 'seed-stability.csv', 'transitions.csv',
    'profiles.csv', 'same-month-year.csv', 'proximity-common-population.csv',
    'summary.json',
]


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def package(root=Path('.')):
    root = root.resolve()
    names = ROOT_FILES + ['outputs/analysis/' + name for name in ANALYSIS_FILES]
    for folder in ['src', 'tests']:
        names += [str(p.relative_to(root)) for p in (root / folder).glob('*.py')]
    names = sorted(set(names))
    contents = {}
    for name in names:
        source = root / name
        if source.is_symlink() or not source.is_file():
            raise ValueError(f'Missing or symlinked submission file: {name}')
        if root not in source.resolve().parents:
            raise ValueError(f'File escapes project root: {name}')
        blob = source.read_bytes()
        if source.suffix in {'.md', '.py', '.json', '.ipynb', '.txt', '.csv'}:
            text = blob.decode('utf-8')
            # These indicators identify host-specific paths or connection secrets.
            for forbidden in ['/'+'Users/', 'vless'+':/'+'/', 'vmess'+':/'+'/']:
                if forbidden in text:
                    raise ValueError(f'Private local content in submission: {name}')
        contents[name] = blob
    manifest = {
        'purpose': 'SberIndex competition 2026, clustering',
        'raw_data': 'Acquire separately from the official links in README.md',
        'files': [{'file': name, 'bytes': len(blob), 'sha256': digest(blob)}
                  for name, blob in contents.items()],
    }
    manifest_blob = (json.dumps(manifest, ensure_ascii=False, indent=2)+'\n').encode()
    destination = root / 'delivery'
    destination.mkdir(exist_ok=True)
    archive_path = destination / 'sberindex-submission.zip'
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, blob in contents.items():
            archive.writestr(name, blob)
        archive.writestr('submission-manifest.json', manifest_blob)
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError('Submission ZIP integrity failure')
        for record in manifest['files']:
            if digest(archive.read(record['file'])) != record['sha256']:
                raise ValueError('Submission checksum mismatch')
    (destination / 'submission-manifest.json').write_bytes(manifest_blob)
    print(json.dumps({'archive': str(archive_path.relative_to(root)),
                      'files': len(contents), 'bytes': archive_path.stat().st_size,
                      'sha256': digest(archive_path.read_bytes()),
                      'zip_integrity': 'passed'}, ensure_ascii=False))


if __name__ == '__main__':
    package()
