from __future__ import annotations

import json
import mimetypes
import os
import uuid
from pathlib import Path
from typing import List, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen, urlretrieve

BASE_URL = os.getenv("AGRIVISION_API", "http://127.0.0.1:8000")
TMP_DIR = Path(__file__).resolve().parent / "tmp_test_images"

IMAGE_URL_CANDIDATES = [
	"https://upload.wikimedia.org/wikipedia/commons/4/45/Tomato_plant_leaf.jpg",
	"https://upload.wikimedia.org/wikipedia/commons/6/66/Tomato_Leaf_01.jpg",
	"https://upload.wikimedia.org/wikipedia/commons/0/0d/Tomato_leaf_closeup.jpg",
	"https://upload.wikimedia.org/wikipedia/commons/4/49/Tomato_leaf_%28Solanum_lycopersicum%29.jpg",
	"https://images.unsplash.com/photo-1592928302575-2f0cf30cf479?auto=format&fit=crop&w=800&q=80",
	"https://images.unsplash.com/photo-1582281298055-e25b84a30b0b?auto=format&fit=crop&w=800&q=80",
]


def _print_result(name: str, ok: bool, detail: str = "") -> None:
	status = "PASS" if ok else "FAIL"
	suffix = f" - {detail}" if detail else ""
	print(f"[{status}] {name}{suffix}")


def _http_json(method: str, endpoint: str, data: bytes | None = None, content_type: str | None = None):
	headers = {}
	if content_type:
		headers["Content-Type"] = content_type
	req = Request(f"{BASE_URL}{endpoint}", data=data, headers=headers, method=method)
	with urlopen(req, timeout=60) as resp:
		payload = resp.read().decode("utf-8")
		return resp.status, json.loads(payload)


def _build_multipart(fields: List[Tuple[str, str]], files: List[Tuple[str, Path]]) -> Tuple[bytes, str]:
	boundary = f"----AgriVisionBoundary{uuid.uuid4().hex}"
	lines: List[bytes] = []

	for name, value in fields:
		lines.append(f"--{boundary}".encode())
		lines.append(f'Content-Disposition: form-data; name="{name}"'.encode())
		lines.append(b"")
		lines.append(value.encode("utf-8"))

	for field_name, file_path in files:
		mime = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
		content = file_path.read_bytes()
		lines.append(f"--{boundary}".encode())
		lines.append(
			(
				f'Content-Disposition: form-data; name="{field_name}"; '
				f'filename="{file_path.name}"'
			).encode()
		)
		lines.append(f"Content-Type: {mime}".encode())
		lines.append(b"")
		lines.append(content)

	lines.append(f"--{boundary}--".encode())
	body = b"\r\n".join(lines) + b"\r\n"
	return body, f"multipart/form-data; boundary={boundary}"


def _download_test_images() -> List[Path]:
	TMP_DIR.mkdir(parents=True, exist_ok=True)
	saved: List[Path] = []
	for url in IMAGE_URL_CANDIDATES:
		if len(saved) >= 3:
			break
		idx = len(saved) + 1
		target = TMP_DIR / f"leaf_{idx}.jpg"
		try:
			urlretrieve(url, target)
			saved.append(target)
		except Exception:
			continue

	if len(saved) < 3:
		raise RuntimeError("Could not download at least 3 test images from internet sources.")

	return saved


def main() -> None:
	print(f"Testing API at {BASE_URL}")
	try:
		images = _download_test_images()
	except Exception as exc:  # pylint: disable=broad-except
		_print_result("Download sample images", False, str(exc))
		return

	# Test /health
	try:
		status, payload = _http_json("GET", "/health")
		ok = status == 200 and isinstance(payload, dict)
		_print_result("GET /health", ok, f"status={status}")
	except (HTTPError, URLError, TimeoutError) as exc:
		_print_result("GET /health", False, str(exc))

	# Test /classes
	try:
		status, payload = _http_json("GET", "/classes")
		classes = payload.get("classes", []) if isinstance(payload, dict) else []
		ok = status == 200 and isinstance(classes, list)
		_print_result("GET /classes", ok, f"class_count={len(classes)}")
	except (HTTPError, URLError, TimeoutError) as exc:
		_print_result("GET /classes", False, str(exc))

	# Test /predict
	predict_payload = None
	try:
		body, content_type = _build_multipart(
			fields=[("crop_hint", "Tomato")],
			files=[("image", images[0])],
		)
		status, predict_payload = _http_json("POST", "/predict", data=body, content_type=content_type)
		ok = status == 200 and isinstance(predict_payload, dict)
		_print_result("POST /predict", ok, f"status={status}")
		print("\nFull /predict response:")
		print(json.dumps(predict_payload, indent=2, ensure_ascii=True))
	except HTTPError as exc:
		detail = exc.read().decode("utf-8", errors="ignore")
		_print_result("POST /predict", False, f"HTTP {exc.code}: {detail}")
	except (URLError, TimeoutError) as exc:
		_print_result("POST /predict", False, str(exc))

	# Test /drone-scan with 3 images
	try:
		body, content_type = _build_multipart(
			fields=[],
			files=[("images", images[0]), ("images", images[1]), ("images", images[2])],
		)
		status, payload = _http_json("POST", "/drone-scan", data=body, content_type=content_type)
		ok = status == 200 and isinstance(payload, dict)
		_print_result("POST /drone-scan", ok, f"status={status}")
	except HTTPError as exc:
		detail = exc.read().decode("utf-8", errors="ignore")
		_print_result("POST /drone-scan", False, f"HTTP {exc.code}: {detail}")
	except (URLError, TimeoutError) as exc:
		_print_result("POST /drone-scan", False, str(exc))


if __name__ == "__main__":
	main()
