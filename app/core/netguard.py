"""IP manzillarni SSRF nuqtai nazaridan tekshirish.

Bu server 25+ sayt va ichki servisni (Redis, Postfix, aaPanel :8888, PowerDNS)
host qiladi. Foydalanuvchi bergan URL'dan rasm yuklaydigan API — bu to'g'ridan
to'g'ri SSRF vektori: `http://127.0.0.1:8888/` yoki `http://169.254.169.254/`
kabi so'rovlar ichki tarmoqni tashqariga ochib qo'yadi.

Shuning uchun DNS natijasidagi HAR BIR IP shu yerda tekshiriladi.
"""

from __future__ import annotations

import ipaddress
import socket
from typing import Iterable

# Python versiyalari orasida `is_private` ta'rifi o'zgargan (masalan 100.64.0.0/10
# 3.12.4 dan keyin privat hisoblanmaydi). Shuning uchun muhim diapazonlarni
# standart bayroqlarga tayanmasdan, aniq ro'yxat bilan ham bloklaymiz.
_EXTRA_BLOCKED = (
    ipaddress.ip_network("100.64.0.0/10"),  # CGNAT
    ipaddress.ip_network("192.0.0.0/24"),  # IETF protokol tayinlovlari
    ipaddress.ip_network("192.0.2.0/24"),  # TEST-NET-1
    ipaddress.ip_network("198.18.0.0/15"),  # benchmark
    ipaddress.ip_network("198.51.100.0/24"),  # TEST-NET-2
    ipaddress.ip_network("203.0.113.0/24"),  # TEST-NET-3
    ipaddress.ip_network("64:ff9b::/96"),  # NAT64
    ipaddress.ip_network("100::/64"),  # discard-only
    ipaddress.ip_network("2001:db8::/32"),  # hujjatlar uchun
)


def normalize(ip: ipaddress.IPv4Address | ipaddress.IPv6Address):
    """IPv4-mapped IPv6 (`::ffff:127.0.0.1`) ni asl IPv4 ga qaytaradi.

    Busiz `::ffff:127.0.0.1` "global IPv6" ko'rinib, loopback tekshiruvidan
    o'tib ketardi.
    """
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped:
            return ip.ipv4_mapped
        if ip.sixtofour:
            return ip.sixtofour
    return ip


def is_public(raw: str) -> bool:
    """IP manzil tashqi (marshrutlanadigan) bo'lsa True."""
    try:
        ip = normalize(ipaddress.ip_address(raw))
    except ValueError:
        return False

    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    ):
        return False
    if any(ip in net for net in _EXTRA_BLOCKED if net.version == ip.version):
        return False
    # IPv6 unique-local (fc00::/7) — `is_private` odatda qamrab oladi, lekin
    # aniqlik uchun takroran tekshiramiz.
    return not (ip.version == 6 and ip in ipaddress.ip_network("fc00::/7"))


def resolve_all(host: str, port: int) -> list[tuple[int, str]]:
    """Hostni barcha IP'lariga hal qiladi.

    Qaytadi: `[(address_family, ip), ...]`. Hech bo'lmasa bittasi bo'ladi,
    aks holda `socket.gaierror` ko'tariladi.
    """
    infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    seen: set[str] = set()
    out: list[tuple[int, str]] = []
    for family, _type, _proto, _canon, sockaddr in infos:
        ip = sockaddr[0]
        if ip not in seen:
            seen.add(ip)
            out.append((family, ip))
    return out


def pick_safe_ip(candidates: Iterable[tuple[int, str]]) -> tuple[int, str] | None:
    """Birinchi xavfsiz IP ni tanlaydi.

    Agar hostning IP'laridan HAR QANDAY bittasi ichki bo'lsa — hech qaysisiga
    ishonmaymiz va None qaytaramiz. Chunki aralash javob (bitta tashqi + bitta
    ichki IP) — bu DNS rebinding hujumining klassik ko'rinishi.
    """
    items = list(candidates)
    if not items:
        return None
    if any(not is_public(ip) for _family, ip in items):
        return None
    return items[0]
