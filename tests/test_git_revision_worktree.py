# SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
# SPDX-License-Identifier: GPL-2.0-or-later

"""Regression test for revision discovery from a linked Git worktree."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path


def run(*args: str, cwd: Path) -> str:
    completed = subprocess.run(
        args,
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return completed.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmake", required=True)
    parser.add_argument("--module", required=True)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="shadps4-git-revision-") as temporary:
        root = Path(temporary)
        repository = root / "repository"
        worktree = root / "linked-worktree"
        build = root / "build"
        repository.mkdir()

        run("git", "init", cwd=repository)
        run("git", "config", "user.name", "shadPS4 tests", cwd=repository)
        run("git", "config", "user.email", "tests@shadps4.invalid", cwd=repository)
        (repository / "CMakeLists.txt").write_text(
            """
cmake_minimum_required(VERSION 3.16)
project(worktree_revision_test NONE)
include("${REVISION_MODULE}")
get_git_head_revision(GIT_REF_SPEC GIT_REV)
file(WRITE "${CMAKE_BINARY_DIR}/revision.txt" "${GIT_REV}")
""".lstrip(),
            encoding="utf-8",
        )
        run("git", "add", "CMakeLists.txt", cwd=repository)
        run("git", "commit", "-m", "seed", cwd=repository)
        expected_revision = run("git", "rev-parse", "HEAD", cwd=repository)
        run("git", "worktree", "add", "-b", "linked-test", str(worktree), cwd=repository)

        run(
            args.cmake,
            "-S",
            str(worktree),
            "-B",
            str(build),
            f"-DREVISION_MODULE={Path(args.module).resolve().as_posix()}",
            cwd=root,
        )
        actual_revision = (build / "revision.txt").read_text(encoding="utf-8")
        if actual_revision != expected_revision:
            raise AssertionError(
                "linked worktree revision mismatch: "
                f"expected {expected_revision!r}, got {actual_revision!r}"
            )


if __name__ == "__main__":
    main()
