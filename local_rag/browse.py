"""List folders and indexable files on this computer for the UI picker."""

from __future__ import annotations

from pathlib import Path

from local_rag.extract import kind_for

SKIP_DIR_NAMES = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".cache",
    ".local",
    ".config",
    ".npm",
    ".cargo",
    "snap",
}


def roots() -> list[dict]:
    home = Path.home()
    candidates = [
        ("Home", home),
        ("Documents", home / "Documents"),
        ("Desktop", home / "Desktop"),
        ("Downloads", home / "Downloads"),
        ("Pictures", home / "Pictures"),
        ("Videos", home / "Videos"),
        ("School files", home / "School files"),
    ]
    out = []
    for label, path in candidates:
        if path.is_dir():
            out.append({"label": label, "path": str(path)})
    return out


def list_dir(path: str | None = None) -> dict:
    home = Path.home()
    current = Path(path).expanduser() if path else home
    if not current.is_absolute():
        current = (home / current).resolve()
    else:
        current = current.resolve()

    if not current.exists() or not current.is_dir():
        raise FileNotFoundError(f"Not a folder: {current}")

    # Stay under the user's home when possible; still allow absolute paths they type.
    try:
        current.relative_to(home)
        under_home = True
    except ValueError:
        under_home = False

    parent = current.parent if current != current.root else None
    if under_home and current == home:
        parent = None
    elif under_home and parent is not None:
        try:
            parent.relative_to(home)
        except ValueError:
            parent = home

    folders = []
    files = []
    try:
        entries = sorted(current.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except PermissionError as exc:
        raise PermissionError(f"Cannot read {current}: {exc}") from exc

    for entry in entries:
        name = entry.name
        if name.startswith("."):
            continue
        if entry.is_dir():
            if name in SKIP_DIR_NAMES:
                continue
            folders.append({"name": name, "path": str(entry)})
            continue
        if not entry.is_file():
            continue
        kind = kind_for(entry)
        if kind is None:
            continue
        try:
            size = entry.stat().st_size
        except OSError:
            continue
        if size == 0:
            continue
        files.append(
            {
                "name": name,
                "path": str(entry),
                "kind": kind,
                "size": size,
            }
        )

    return {
        "path": str(current),
        "parent": str(parent) if parent else None,
        "folders": folders,
        "files": files,
        "roots": roots(),
    }
