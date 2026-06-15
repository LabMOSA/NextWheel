#!/usr/bin/env python
"""Check code quality before creating a code request."""

import subprocess
import webbrowser
import os


def _run_and_print(command: list[str]) -> None:
    """Run command and print the result in console."""
    try:
        output = subprocess.check_output(command).decode()
        print(output)
    except subprocess.CalledProcessError as e:
        error_output: str = (e.output or b"").decode()
        for line in error_output.split("\n"):
            print(line)


def run_style_formatter() -> None:  # pragma: no cover
    """Run style formatter (black)."""
    print("========================================")
    print("Reformatting to black and numpydoc")
    print("Running black...")
    _run_and_print(["black", "nextwheel"])
    print("Running docformatter...")
    _run_and_print(
        [
            "docformatter",
            "--style=numpy",
            "--in-place",
            "--recursive",
            "--pre-summary-newline",
            "--blank",
            ".",
        ]
    )


def run_static_type_checker() -> None:  # pragma: no cover
    """Run static typing checker (mypy)."""
    print("========================================")
    print("Checking Static Types (mypy)")
    _run_and_print(["mypy", "nextwheel"])


def run_pylint() -> None:  # pragma: no cover
    """Run code quality review (pylint)."""
    print("========================================")
    print("Code Quality Review (pylint)")
    _run_and_print(["pylint", "nextwheel"])


def run_unit_tests() -> None:  # pragma: no cover
    """Run all unit tests."""
    print("========================================")
    print("Running Unit Tests (coverage/pytest)")
    _run_and_print(
        [
            "coverage",
            "run",
            "--source",
            ".",
            "-m",
            "pytest",
        ]
    )
    subprocess.call(["coverage", "html", "-d", "reports/coverage"])
    webbrowser.open_new_tab(
        "file://" + os.path.abspath("reports/coverage/index.html")
    )


if __name__ == "__main__":  # pragma: no cover
    run_style_formatter()
    run_static_type_checker()
    run_pylint()
    run_unit_tests()
    print("========================================")
    print("Completed.")
    print("Do not forget to merge main into your")
    print("branch and run these checks again.")
    print("========================================")
