from __future__ import annotations

import json
import os
from pathlib import Path

import requests

BASE_URL = os.getenv("API_BASE_URL", "http://fastapi_service:8000")
ANNOTATE_ENDPOINT = f"{BASE_URL}/annotate-spectrum"
TEST_CASES_DIR = Path(os.getenv("TEST_CASES_DIR", "/service/test_cases"))
SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".mzml": "mzML",
    ".mgf": "MGF",
    ".json": "JSON",
    ".msp": "MSP",
}


def _collect_test_files() -> list[Path]:
    if not TEST_CASES_DIR.exists() or not TEST_CASES_DIR.is_dir():
        raise AssertionError(f"Test cases directory does not exist: {TEST_CASES_DIR}")

    files = sorted(path for path in TEST_CASES_DIR.iterdir() if path.is_file())
    if not files:
        raise AssertionError(f"No test files found in: {TEST_CASES_DIR}")
    return files


def _post_file(file_path: Path) -> requests.Response:
    with file_path.open("rb") as file_data:
        files = {"file": (file_path.name, file_data, "application/octet-stream")}
        return requests.post(ANNOTATE_ENDPOINT, files=files, timeout=120)


def _assert_success_payload(payload: dict, file_path: Path) -> None:
    assert payload.get("status") in {"accepted", "processed"}, payload
    assert payload.get("file_name") == file_path.name, payload
    expected_type = SUPPORTED_EXTENSIONS[file_path.suffix.lower()]
    assert payload.get("file_type") == expected_type, payload
    assert isinstance(payload.get("message"), str) and payload.get("message"), payload
    assert isinstance(payload.get("results"), list), payload


def run() -> None:
    print(f"Running API tests against: {ANNOTATE_ENDPOINT}")
    files = _collect_test_files()
    print(f"Found {len(files)} test files in {TEST_CASES_DIR}")

    passed = 0
    failed = 0

    for file_path in files:
        try:
            response = _post_file(file_path)
            extension = file_path.suffix.lower()

            assert response.status_code != 500, (
                f"{file_path.name}: server returned 500. Body: {response.text}"
            )

            if extension not in SUPPORTED_EXTENSIONS:
                assert response.status_code == 400, (
                    f"{file_path.name}: expected 400 for unsupported extension, got {response.status_code}. "
                    f"Body: {response.text}"
                )
                payload = response.json()
                assert isinstance(payload.get("detail"), str) and payload["detail"], payload
            elif response.status_code == 400:
                payload = response.json()
                assert isinstance(payload.get("detail"), str) and payload["detail"], payload
            else:
                assert response.status_code == 202, (
                    f"{file_path.name}: expected 202 or 400, got {response.status_code}. "
                    f"Body: {response.text}"
                )
                payload = response.json()
                _assert_success_payload(payload, file_path)

            passed += 1
            print(f"[PASS] {file_path.name} -> HTTP {response.status_code}")
        except Exception as error:
            failed += 1
            print(f"[FAIL] {file_path.name} -> {error}")

    print(f"\nFinished: passed={passed}, failed={failed}, total={len(files)}")
    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    run()
