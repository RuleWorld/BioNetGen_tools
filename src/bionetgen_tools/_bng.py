"""Helpers that call BNG2.pl directly.

bngsim loads a BNGL file by generating its reaction network with BNG2.pl.
The network-free engines (NFsim, RuleMonkey) need the model's rules instead,
as the BioNetGen XML that BNG2.pl's ``writeXML()`` action writes. This module
writes that file.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

_ACTION = re.compile(r"^\s*[A-Za-z_]\w*\s*\(")
_BEGIN = re.compile(r"^\s*begin\s+(\w[\w ]*?)\s*$", re.IGNORECASE)
_END = re.compile(r"^\s*end\s+(\w[\w ]*?)\s*$", re.IGNORECASE)


def find_bng2() -> Path:
    """Return the path to BNG2.pl.

    Looks at ``$BNG2_PL`` and ``$BNGPATH`` first, then asks bngsim, which finds
    the copy that ``pip install 'bngsim[bngl]'`` installs.
    """
    if os.environ.get("BNG2_PL"):
        return Path(os.environ["BNG2_PL"])
    if os.environ.get("BNGPATH"):
        return Path(os.environ["BNGPATH"]) / "BNG2.pl"
    try:  # private in bngsim 0.17; fall back below if it moves
        from bngsim._bngl_loader import resolve_bng2

        return Path(resolve_bng2())
    except Exception:
        pass
    found = shutil.which("BNG2.pl")
    if found:
        return Path(found)
    raise FileNotFoundError(
        "BNG2.pl not found. Install it with pip install 'bngsim[bngl]', "
        "or set BNGPATH to a BioNetGen installation."
    )


def model_block_only(bngl_text: str) -> str:
    """Return the BNGL text with every action removed.

    Keeps everything inside begin/end blocks and drops action calls that sit
    outside them (``generate_network(...)``, ``simulate(...)`` and so on).
    """
    out = []
    depth = 0
    for line in bngl_text.splitlines():
        if _BEGIN.match(line):
            depth += 1
            out.append(line)
            continue
        if _END.match(line):
            depth = max(0, depth - 1)
            out.append(line)
            continue
        if depth == 0 and _ACTION.match(line):
            continue
        out.append(line)
    return "\n".join(out) + "\n"


_XML_CACHE: dict[str, Path] = {}


def write_bng_xml(bngl_file: str | os.PathLike, timeout: int = 600) -> Path:
    """Write the BioNetGen XML for a BNGL model and return its path.

    The file is written to a temporary directory and cached for the session,
    keyed on the model text, so repeated calls do not rerun BNG2.pl.
    """
    text = Path(bngl_file).read_text()
    key = hashlib.sha256(text.encode()).hexdigest()[:16]
    if key in _XML_CACHE and _XML_CACHE[key].exists():
        return _XML_CACHE[key]

    workdir = Path(tempfile.mkdtemp(prefix="bngtools_xml_"))
    stem = Path(bngl_file).stem
    src = workdir / f"{stem}.bngl"
    src.write_text(model_block_only(text) + "\nwriteXML()\n")
    bng2 = find_bng2()
    proc = subprocess.run(
        ["perl", str(bng2), f"{stem}.bngl"],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    xml = workdir / f"{stem}.xml"
    if proc.returncode != 0 or not xml.exists():
        raise RuntimeError(
            f"BNG2.pl writeXML failed for {bngl_file}:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )
    _XML_CACHE[key] = xml
    return xml
