"""Build the standalone player into handoff/policy-player-live/ and link its model to the auto-updated export.

Run with .cache/mario-binary-runtime-venv/bin/python (numpy, Pillow, onnxruntime, PyInstaller; no torch).
"""
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / 'handoff/policy-player-live'
OLD_LICENSES = ROOT.parent / 'build_games_from_scratch_in_threejs/policy-player/LICENSES'
SOURCES = ('app.py', 'export.py', 'watch.py', 'build.py')


def main():
    with tempfile.TemporaryDirectory() as work:
        subprocess.run([sys.executable, '-m', 'PyInstaller', '--onefile', '--clean', '--noconfirm', '--name', 'pixel-policy',
                        '--distpath', str(PACKAGE), '--workpath', work, '--specpath', work,
                        '--exclude-module', 'torch', '--exclude-module', 'pytest', '--exclude-module', 'tkinter',
                        '--exclude-module', 'onnx', '--exclude-module', 'readline', '--collect-binaries', 'onnxruntime',
                        str(ROOT / 'rl/player/app.py')], cwd=ROOT, check=True)
    licenses = PACKAGE / 'LICENSES'
    shutil.rmtree(licenses, ignore_errors=True)
    licenses.mkdir()
    for name in ('numpy', 'pillow', 'onnxruntime', 'pyinstaller', 'system-runtime'):
        shutil.copytree(OLD_LICENSES / name, licenses / name)
    shutil.copyfile(OLD_LICENSES / 'Python-LICENSE.txt', licenses / 'Python-LICENSE.txt')
    with tarfile.open(licenses / 'policy-interface-source.tar.gz', 'w:gz') as archive:
        for name in SOURCES:
            archive.add(ROOT / 'rl/player' / name, arcname='rl/player/' + name)
    link = PACKAGE / 'model'
    link.unlink(missing_ok=True)
    link.symlink_to(ROOT / 'runs/rl/policy-export/current')


if __name__ == '__main__':
    main()
