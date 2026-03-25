from src.services.clip_service import CLIPService

_clip_service: CLIPService | None = None


def set_clip_service(service: CLIPService | None) -> None:
    global _clip_service  # noqa: PLW0603
    _clip_service = service


def get_clip_service() -> CLIPService:
    if _clip_service is None:
        raise RuntimeError("CLIPService not initialized")
    return _clip_service
