"""Check the one release manifest and run synthetic contract/regression tests."""
import ast
import json
from pathlib import Path
import unittest

from evaluate import ROOT, verify_release


def main():
    release = verify_release()
    for name in release['sha256']:
        if name.endswith('.py'):
            ast.parse((ROOT / name).read_text())
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    evidence = {'release': release['release'], 'frozen_hashes_checked': len(release['sha256']),
        'tests_run': result.testsRun, 'failures': len(result.failures),
        'errors': len(result.errors), 'skipped': len(result.skipped),
        'successful': result.wasSuccessful(), 'test_labels_read': False}
    output = ROOT / 'results'
    output.mkdir(exist_ok=True)
    (output / 'audit.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps(evidence, indent=2))
    if not result.wasSuccessful() or result.skipped:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
