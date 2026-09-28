from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from app.api import deps


def _request(ip: str):
    return SimpleNamespace(headers={}, client=SimpleNamespace(host=ip), state=SimpleNamespace())


@pytest.mark.parametrize(("setting", "expected"), [(None, 60), (0, 0), (500, 500)])
def test_trusted_ip_limit_is_separate(monkeypatch, setting, expected) -> None:
    monkeypatch.setattr(deps._settings, "rate_limit_per_minute", 60)
    monkeypatch.setattr(deps._settings, "trusted_rate_limit_per_minute", setting)
    principal = asyncio.run(deps.authenticate(_request("127.0.0.1")))
    assert principal.kind == "trusted_ip"
    assert principal.rate_limit == expected
