"""Extract the current release notes and package a CLI for distribution."""

import argparse
import hashlib
from pathlib import Path
import re
import tarfile
import tomllib
import zipfile


def release_notes(version):
    with Path("Cargo.toml").open("rb") as manifest:
        package_version = tomllib.load(manifest)["package"]["version"]
    if version is None:
        version = package_version
    if not re.fullmatch(r"(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*)){2}", version):
        raise ValueError("version must have the form X.Y.Z")
    if version != package_version:
        raise ValueError(f"requested version {version} differs from Cargo.toml ({package_version})")

    changelog = Path("CHANGELOG.md").read_text(encoding="utf-8")
    headings = list(re.finditer(r"^## ([^\n]+)$", changelog, re.MULTILINE))
    releases = [heading for heading in headings if heading[1] != "Unreleased"]
    title = re.compile(rf"\[{re.escape(version)}\](?: - \d{{4}}-\d{{2}}-\d{{2}})?")
    current = [heading for heading in releases if title.fullmatch(heading[1])]
    if len(current) != 1 or current[0] != releases[0]:
        raise ValueError(f"CHANGELOG.md must have one [{version}] section as its latest release")

    index = headings.index(current[0])
    end = headings[index + 1].start() if index + 1 < len(headings) else len(changelog)
    notes = changelog[current[0].end():end].strip()
    if not notes:
        raise ValueError(f"CHANGELOG.md has no notes for {version}")
    return version, notes + "\n"


def package(args):
    version, _ = release_notes(args.version)
    name = f"loci-search-v{version}-{args.target}"
    windows = args.target == "x86_64-pc-windows-msvc"
    binary = "loci.exe" if windows else "loci"
    files = [Path("target") / args.target / "release" / binary, Path("LICENSE"), Path("README.md")]
    args.output.mkdir(parents=True, exist_ok=True)
    if windows:
        archive = args.output / f"{name}.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
            for file in files:
                bundle.write(file, f"{name}/{file.name}")
    else:
        archive = args.output / f"{name}.tar.gz"
        with tarfile.open(archive, "w:gz") as bundle:
            for file in files:
                bundle.add(file, arcname=f"{name}/{file.name}")
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    archive.with_name(f"{archive.name}.sha256").write_text(
        f"{digest}  {archive.name}\n", encoding="ascii", newline="\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    notes = commands.add_parser("notes")
    notes.add_argument("--version")
    notes.add_argument("--output", required=True, type=Path)
    archive = commands.add_parser("package")
    archive.add_argument("--version")
    archive.add_argument("--target", required=True, choices=[
        "x86_64-unknown-linux-gnu", "x86_64-pc-windows-msvc",
    ])
    archive.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.command == "notes":
            _, body = release_notes(args.version)
            args.output.write_text(body, encoding="utf-8", newline="\n")
        else:
            package(args)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Release check failed: {error}\n")


if __name__ == "__main__":
    main()
