"""
Installs the optional `infostop` dependency (needed only for the Infostop detector).

infostop 0.1.9 is distributed as source only and does not build out of the box:
- its C++ code uses M_PI, which MSVC does not define, so it fails to compile on Windows;
- it pins infomap==1.0.6, whose setup.py needs pkg_resources (removed in setuptools>=81).

This script downloads the infostop source, declares M_PI, and installs it
into the current Python environment with a compatible setuptools for the build.

Usage (from the environment where OptimalStop is installed):
    python install_infostop.py
"""

import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

INFOSTOP_VERSION = "0.1.9"
M_PI_DEFINE = "#ifndef M_PI\n#define M_PI 3.14159265358979323846\n#endif\n"


def pip(*args, env=None):
    subprocess.check_call([sys.executable, "-m", "pip", *args], env=env)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        pip("download", f"infostop=={INFOSTOP_VERSION}", "--no-deps", "--no-binary", ":all:", "-d", str(tmp))
        with tarfile.open(tmp / f"infostop-{INFOSTOP_VERSION}.tar.gz") as tar:
            tar.extractall(tmp)
        src = tmp / f"infostop-{INFOSTOP_VERSION}"

        # Declare M_PI after the last #include
        cpp = src / "cpputils" / "main.cpp"
        lines = cpp.read_text().splitlines(keepends=True)
        last_include = max(i for i, line in enumerate(lines) if line.lstrip().startswith("#include"))
        lines.insert(last_include + 1, M_PI_DEFINE)
        cpp.write_text("".join(lines))

        # infomap==1.0.6 needs pkg_resources at build time
        constraints = tmp / "build-constraints.txt"
        constraints.write_text("setuptools<81\n")
        env = dict(os.environ, PIP_CONSTRAINT=str(constraints))

        pip("install", str(src), env=env)

    print("infostop installed successfully.")


if __name__ == "__main__":
    main()
