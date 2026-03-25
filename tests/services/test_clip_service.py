import base64
import io
from unittest.mock import patch

import pytest
from PIL import Image
from src.services.clip_service import CLIPService


def _noop_init(self: CLIPService, **kwargs: object) -> None:
    pass


class TestDecodeImage:
    def test_valid_png(self) -> None:
        img = Image.new("RGB", (32, 32), "red")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        with patch.object(CLIPService, "__init__", _noop_init):
            service = CLIPService()
            result = service.decode_image(b64)
            assert result.size == (32, 32)
            assert result.mode == "RGB"

    def test_valid_jpeg(self) -> None:
        img = Image.new("RGB", (64, 48), "blue")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        with patch.object(CLIPService, "__init__", _noop_init):
            service = CLIPService()
            result = service.decode_image(b64)
            assert result.size == (64, 48)

    def test_invalid_base64(self) -> None:
        with patch.object(CLIPService, "__init__", _noop_init):
            service = CLIPService()
            with pytest.raises(ValueError, match="Invalid base64 image"):
                service.decode_image("not_valid!!!")

    def test_valid_base64_but_not_image(self) -> None:
        b64 = base64.b64encode(b"this is not an image").decode()
        with patch.object(CLIPService, "__init__", _noop_init):
            service = CLIPService()
            with pytest.raises(ValueError, match="Invalid base64 image"):
                service.decode_image(b64)
