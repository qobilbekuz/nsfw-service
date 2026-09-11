"""Klassifikator va detektor natijalarini yagona verdictga birlashtirish.

Ballar shkalasi — HAMMA joyda foiz (0.00-100.00), 2 kasr raqamgacha.

`sfw + nsfw + nsfl = 100` (klassifikatorning softmax chiqishi).
`suggestive` — alohida hosila ko'rsatkich (bu yig'indiga kirmaydi): u
"ochiq-oydin 18+ emas, lekin shahvoniy ishora bor" holatini o'lchaydi va
asosan detektorning `COVERED` topilmalaridan keladi.
"""

from __future__ import annotations

from app.config import get_settings
from app.schemas import Detection, Scores, Verdict
from app.services import detector as det

_settings = get_settings()

# Detektor topilmasi verdictni `nsfw` ga ko'tarishi uchun kerakli minimal
# ishonch. Klassifikator chegarasidan qat'iyroq, chunki YOLO topilmalari
# yolg'on ijobiy berishi mumkin (masalan yaqindan olingan teri kadri).
EXPLICIT_PROMOTE_SCORE = 50.0
SUGGESTIVE_PROMOTE_SCORE = 40.0


def _peak(detections: list[Detection], labels: frozenset[str]) -> float:
    """Berilgan sinflar orasidagi eng yuqori ball (topilmasa 0)."""
    relevant = [d.score for d in detections if d.label in labels]
    return max(relevant) if relevant else 0.0


def evaluate(
    class_scores: dict[str, float],
    detections: list[Detection],
    detections_available: bool = True,
) -> tuple[Verdict, float, Scores, list[str]]:
    """Yakuniy verdict, ishonch, ballar va sabablar ro'yxatini qaytaradi.

    `detections_available` — detektor umuman ishlatildimi. Bo'sh `detections`
    ro'yxati ikki xil ma'noni bildirishi mumkin: "hech narsa topilmadi" yoki
    "detektor o'chirilgan". Oraliq zonadagi qaror shu farqqa bog'liq.
    """
    sfw = class_scores["sfw"]
    nsfw = class_scores["nsfw"]
    nsfl = class_scores["nsfl"]

    explicit_peak = _peak(detections, det.EXPLICIT_LABELS)
    covered_peak = _peak(detections, det.SUGGESTIVE_LABELS)

    # Hosila `suggestive`: yopiq/yarim ochiq topilmalar va klassifikatorning
    # o'rta darajadagi nsfw bahosidan eng kattasi olinadi. Ochiq-oydin
    # topilma bo'lsa bu ko'rsatkich ma'nosini yo'qotadi (verdict baribir nsfw).
    suggestive = max(covered_peak, nsfw if nsfw < _settings.threshold_nsfw else 0.0)

    scores = Scores(
        sfw=round(sfw, 2),
        suggestive=round(suggestive, 2),
        nsfw=round(nsfw, 2),
        nsfl=round(nsfl, 2),
    )

    reasons: list[str] = [f"classifier: sfw {sfw:.2f}% / nsfw {nsfw:.2f}% / nsfl {nsfl:.2f}%"]
    if detections:
        top = detections[0]
        reasons.append(f"detector: {top.label} {top.score:.2f}%")
        # Eng baland topilma ko'pincha `BELLY_EXPOSED` kabi *yumshoq* sinf
        # bo'ladi, qaror esa ochiq-oydin sinfga qarab chiqadi. Ikkalasi
        # boshqa-boshqa bo'lsa, qarorga asos bo'lganini ham ko'rsatamiz —
        # aks holda sabablar ro'yxati qarorni tushuntirmaydi.
        explicit = [d for d in detections if d.label in det.EXPLICIT_LABELS]
        if explicit and explicit[0].label != top.label:
            best = max(explicit, key=lambda d: d.score)
            reasons.append(f"detector (ochiq-oydin): {best.label} {best.score:.2f}%")

    # --- Verdict (tartib muhim: eng og'iri birinchi tekshiriladi) ---
    if nsfl >= _settings.threshold_nsfl:
        reasons.append(f"nsfl chegarasi oshdi ({nsfl:.2f}% >= {_settings.threshold_nsfl}%)")
        return Verdict.NSFL, round(nsfl, 2), scores, reasons

    if nsfw >= _settings.threshold_nsfw:
        # Klassifikator juda ishonchli — detektor tasdig'i shart emas.
        if nsfw >= _settings.threshold_nsfw_confident:
            reasons.append(
                f"nsfw yuqori ishonch ({nsfw:.2f}% >= {_settings.threshold_nsfw_confident}%)"
            )
            return Verdict.NSFW, round(nsfw, 2), scores, reasons

        # ORALIQ ZONA. Bu model ataylab qattiqqo'l: ochiq yelka yoki mayka
        # kiygan oddiy portretga ham 70-80% berishi mumkin (sinovda tasdiqlandi).
        # Shuning uchun bu zonada detektordan ochiq tana qismi topilishini
        # talab qilamiz. Topilmasa — `nsfw` emas, `suggestive` deb belgilaymiz:
        # signal saqlanadi, lekin oddiy portret 18+ deb bloklanmaydi.
        if explicit_peak >= EXPLICIT_PROMOTE_SCORE:
            reasons.append(
                f"nsfw oraliq zonasi ({nsfw:.2f}%) detektor bilan tasdiqlandi "
                f"({explicit_peak:.2f}%)"
            )
            return Verdict.NSFW, round(nsfw, 2), scores, reasons

        if not detections_available:
            # Detektor o'chirilgan — tasdiqlab bo'lmaydi, chegaraga tayanamiz.
            reasons.append(
                f"nsfw chegarasi oshdi ({nsfw:.2f}%), detektor o'chirilgan"
            )
            return Verdict.NSFW, round(nsfw, 2), scores, reasons

        # Ikki holatni ajratamiz: detektor hech narsa topmadi, yoki topdi-yu
        # ishonchi chegaradan past. Moderator uchun bu farq muhim — ikkinchisi
        # qo'lda ko'rib chiqishga arziydigan chegaraviy holat.
        if explicit_peak > 0:
            reasons.append(
                f"nsfw oraliq zonasi ({nsfw:.2f}%); ochiq tana qismi topildi, "
                f"lekin ishonchi past ({explicit_peak:.2f}% < "
                f"{EXPLICIT_PROMOTE_SCORE:.0f}%) -> suggestive"
            )
        else:
            reasons.append(
                f"nsfw oraliq zonasi ({nsfw:.2f}%), lekin detektor ochiq tana "
                "qismini topmadi -> suggestive"
            )
        return Verdict.SUGGESTIVE, round(nsfw, 2), scores, reasons

    if explicit_peak >= EXPLICIT_PROMOTE_SCORE:
        # Klassifikator ishonchsiz, lekin detektor ochiq tana qismini topdi —
        # moderatsiyada yolg'on salbiy yolg'on ijobiydan qimmatroq turadi.
        reasons.append(f"detektorda ochiq tana qismi ({explicit_peak:.2f}%)")
        return Verdict.NSFW, round(max(nsfw, explicit_peak), 2), scores, reasons

    if (
        suggestive >= _settings.threshold_suggestive
        or covered_peak >= SUGGESTIVE_PROMOTE_SCORE
    ):
        reasons.append(f"shahvoniy ishora ({suggestive:.2f}%)")
        return Verdict.SUGGESTIVE, round(suggestive, 2), scores, reasons

    return Verdict.SAFE, round(sfw, 2), scores, reasons


def is_nsfw(verdict: Verdict) -> bool:
    return verdict in (Verdict.NSFW, Verdict.NSFL)


# Chegaraga shu qadar yaqin natijalar "chetlab o'tildi" deb emas, "chegaraviy"
# deb qaraladi. 10 ball — kuzatilgan xatolar kengligiga qarab tanlangan
# (o'lchangan emas; kalibrlashdan keyin aniqlanadi).
REVIEW_MARGIN = 10.0


def needs_review(
    verdict: Verdict,
    class_scores: dict[str, float],
    detections: list[Detection],
) -> tuple[bool, str | None]:
    """Natija qo'lda ko'rib chiqishga arziydimi?

    `is_safe` dan mustaqil: bu bayroq verdictni o'zgartirmaydi, faqat
    "bu qaror ishonchsiz zonada chiqdi" degan signal beradi. Ikki maqsadi bor —
    moderator navbatini to'ldirish va kalibrlash uchun ma'lumot yig'ish
    (`training/` ga qarang).
    """
    nsfw = class_scores["nsfw"]
    nsfl = class_scores["nsfl"]
    explicit_peak = _peak(detections, det.EXPLICIT_LABELS)

    # 1. `suggestive` — ta'rifiga ko'ra chegaraviy holat.
    if verdict is Verdict.SUGGESTIVE:
        return True, "chegaraviy: suggestive"

    # 2. `nsfw` oraliq zonada, ya'ni qaror detektor tasdig'iga tayangan.
    #    Detektor xato qilsa verdict ham xato — shuning uchun ko'rib chiqiladi.
    if verdict is Verdict.NSFW and nsfw < _settings.threshold_nsfw_confident:
        return True, f"chegaraviy: nsfw oraliq zonada ({nsfw:.2f}%)"

    # 3. `nsfl` chegaradan sal yuqorida.
    if verdict is Verdict.NSFL and nsfl < _settings.threshold_nsfl + REVIEW_MARGIN:
        return True, f"chegaraviy: nsfl chegaraga yaqin ({nsfl:.2f}%)"

    # 4. `safe`, lekin oz farq bilan — kuchsiz ochiq-oydin topilma bor,
    #    yoki nsfw chegaraga yaqinlashgan.
    if verdict is Verdict.SAFE:
        if explicit_peak > 0:
            return True, (
                f"chegaraviy: kuchsiz ochiq-oydin topilma ({explicit_peak:.2f}%)"
            )
        if nsfw >= _settings.threshold_nsfw - REVIEW_MARGIN:
            return True, f"chegaraviy: nsfw chegaraga yaqin ({nsfw:.2f}%)"

    return False, None
