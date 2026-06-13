"""Make the offline-tests guarantee self-enforcing.

SPEC.md requires that tests never touch the network and that CI pass offline.
This blocks outbound network connections for the whole test session, so any
test that accidentally introduces a live YouTube/yt-dlp call fails loudly
instead of silently depending on the network. Loopback is left allowed for
local IPC (e.g. a future MCP stdio round-trip via a local port).
"""

import socket

import pytest

_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex
_LOCAL = {"127.0.0.1", "::1", "localhost"}


def _host_of(address: object) -> str | None:
    if isinstance(address, tuple) and address:
        return str(address[0])
    return None


def _guard(method):
    def wrapper(self, address, *args, **kwargs):
        host = _host_of(address)
        if host is not None and host not in _LOCAL:
            raise RuntimeError(
                f"Blocked network connection to {host!r} during tests. "
                "Tests must run offline against tests/fixtures/ — record real "
                "payloads with scripts/record_fixtures.py instead."
            )
        return method(self, address, *args, **kwargs)

    return wrapper


@pytest.fixture(autouse=True, scope="session")
def _block_network() -> None:
    socket.socket.connect = _guard(_real_connect)  # type: ignore[method-assign]
    socket.socket.connect_ex = _guard(_real_connect_ex)  # type: ignore[method-assign]
