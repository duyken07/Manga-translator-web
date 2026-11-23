import subprocess
import sys


def test_runtime_has_no_pyside6_dependency(client, project):
    from conftest import upload
    upload(client, project, ["a.png"])
    assert not any(m == "PySide6" or m.startswith("PySide6.") or m.startswith("PyQt") for m in sys.modules)


def test_no_qt_import_in_source():
    """Không file nào của backend web / module Desktop đã copy còn import PySide6 (trừ chuỗi trong comment)."""
    out = subprocess.run(["grep", "-rnE", r"^\s*(from|import)\s+(PySide6|PyQt)", "webapp", "existing_app/app"],
                         capture_output=True, text=True).stdout
    assert out.strip() == "", out
