from __future__ import annotations

from typing import Optional

from fastapi import HTTPException, UploadFile

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024


def validate_file(file: UploadFile, data: bytes) -> None:
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image type. Allowed: image/jpeg, image/png, image/webp.",
        )

    if len(data) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="File too large. Max allowed size is 10MB.")


def validate_gps(lat: Optional[float], lng: Optional[float]) -> Optional[str]:
    if lat is None and lng is None:
        return None
    if lat is None or lng is None:
        return "Incomplete location coordinates provided. Send both latitude and longitude."

    if not (8.4 <= lat <= 37.6 and 68.7 <= lng <= 97.25):
        return "Location appears outside India bounds; recommendations may be less accurate."

    return None
