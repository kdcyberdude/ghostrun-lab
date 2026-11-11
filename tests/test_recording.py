"""Recording wrapper checks: no model or emulator is loaded."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('options,models,extra', [
    ([], ['v2', 'v1'], []),
    (['--dry-run'], ['v2', 'v1'], ['--dry-run']),
    (['--only', 'v1'], ['v1'], []),
    (['--levels', '1-1,4-4'], ['v2', 'v1'], ['--levels', '1-1,4-4']),
])
def test_recording_wrapper_optional_arguments(tmp_path, options, models, extra):
    root = Path(__file__).resolve().parents[1]
    calls = tmp_path / 'calls.jsonl'
    python = tmp_path / 'fake-python'
    python.write_text(f'#!{sys.executable}\n'
                      'import json, os, sys\n'
                      'with open(os.environ["RECORDING_TEST_CALLS"], "a") as f:\n'
                      '    f.write(json.dumps(sys.argv[1:]) + "\\n")\n')
    python.chmod(0o755)
    out = tmp_path / 'output with spaces'
    subprocess.run(['bash', str(root / 'scripts/record_all_levels.sh'), '--out', str(out), *options],
                   env={**os.environ, 'PYTHON': str(python), 'RECORDING_TEST_CALLS': str(calls)}, check=True)
    assert [json.loads(line) for line in calls.read_text().splitlines()] == [
        ['-u', f'scripts/recording/record_{model}.py', '--out', str(out), *extra] for model in models
    ]
