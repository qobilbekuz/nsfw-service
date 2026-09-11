"""Verdict mantiqi — chegaralar bo'yicha jadval testi (TASKS.md §8)."""

from __future__ import annotations

import pytest

from app.schemas import Box, Detection, Verdict
from app.services import scoring


def det(label: str, score: float) -> Detection:
    return Detection(label=label, score=score, box=Box(x=0, y=0, width=10, height=10))


def scores(sfw: float, nsfw: float, nsfl: float) -> dict[str, float]:
    return {"sfw": sfw, "nsfw": nsfw, "nsfl": nsfl}


@pytest.mark.parametrize(
    ("class_scores", "detections", "expected"),
    [
        # Toza SFW
        (scores(95, 3, 2), [], Verdict.SAFE),
        (scores(95, 3, 2), [det("FACE_MALE", 90)], Verdict.SAFE),
        # NSFL har doim ustun turadi
        (scores(20, 20, 60), [], Verdict.NSFL),
        (scores(5, 90, 5), [], Verdict.NSFW),
        # Yuqori ishonch: detektor tasdig'isiz ham nsfw
        (scores(3, 95, 2), [], Verdict.NSFW),
        # ORALIQ ZONA (60-90): detektor tasdiqlasa -> nsfw
        (scores(20, 75, 5), [det("FEMALE_BREAST_EXPOSED", 80)], Verdict.NSFW),
        # ORALIQ ZONA: ochiq tana qismi yo'q -> suggestive
        # (aynan shu holat oddiy portretni 18+ deb bloklashning oldini oladi)
        (scores(20, 75, 5), [det("FACE_FEMALE", 85)], Verdict.SUGGESTIVE),
        (scores(20, 75, 5), [], Verdict.SUGGESTIVE),
        # Klassifikator past, lekin detektor ishonchli -> nsfw
        (scores(80, 15, 5), [det("MALE_GENITALIA_EXPOSED", 70)], Verdict.NSFW),
        # Yopiq joylar -> suggestive
        (scores(70, 20, 10), [det("FEMALE_BREAST_COVERED", 60)], Verdict.SUGGESTIVE),
        # Zaif detektor signali verdictni ko'tarmaydi
        (scores(90, 6, 4), [det("FEMALE_BREAST_EXPOSED", 30)], Verdict.SAFE),
    ],
)
def test_verdict_table(class_scores, detections, expected) -> None:
    verdict, _confidence, _scores, _reasons = scoring.evaluate(class_scores, detections)
    assert verdict is expected


def test_detector_disabled_falls_back_to_threshold() -> None:
    """Detektor o'chirilganda oraliq zonani tasdiqlab bo'lmaydi.

    Bu holda chegaraga tayanamiz — aks holda `detect=false` bilan yuborilgan
    haqiqiy 18+ rasm jimgina `suggestive` bo'lib o'tib ketardi.
    """
    verdict, _c, _s, _r = scoring.evaluate(scores(20, 75, 5), [], detections_available=False)
    assert verdict is Verdict.NSFW

    verdict, _c, _s, _r = scoring.evaluate(scores(20, 75, 5), [], detections_available=True)
    assert verdict is Verdict.SUGGESTIVE


def test_scores_rounded_to_two_decimals() -> None:
    _v, _c, result, _r = scoring.evaluate(scores(33.333333, 33.333333, 33.333333), [])
    assert result.sfw == 33.33


def test_is_nsfw_mapping() -> None:
    assert scoring.is_nsfw(Verdict.NSFW) is True
    assert scoring.is_nsfw(Verdict.NSFL) is True
    assert scoring.is_nsfw(Verdict.SUGGESTIVE) is False
    assert scoring.is_nsfw(Verdict.SAFE) is False


def test_reasons_always_explain_decision() -> None:
    _v, _c, _s, reasons = scoring.evaluate(
        scores(20, 75, 5), [det("FACE_FEMALE", 85)]
    )
    assert any("classifier" in r for r in reasons)
    assert any("suggestive" in r for r in reasons)


# ---------------------------------------------------------------------------
#  needs_review — chegaraviy holatlar bayrog'i
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("class_scores", "detections", "expected"),
    [
        # Ishonchli safe — ko'rib chiqish shart emas
        (scores(95, 3, 2), [], False),
        (scores(95, 3, 2), [det("FACE_MALE", 90)], False),
        # Ishonchli nsfw (>= 90) — detektor tasdig'iga tayanmagan
        (scores(3, 95, 2), [], False),
        # Oraliq zonada chiqqan nsfw — tayangan, demak tekshirilsin
        (scores(20, 75, 5), [det("FEMALE_BREAST_EXPOSED", 80)], True),
        # Har qanday suggestive
        (scores(20, 75, 5), [det("FACE_FEMALE", 85)], True),
        (scores(70, 20, 10), [det("FEMALE_BREAST_COVERED", 60)], True),
        # safe, lekin kuchsiz ochiq-oydin topilma bor (aynan A rasm holati)
        (scores(90, 6, 4), [det("FEMALE_BREAST_EXPOSED", 30)], True),
        # safe, lekin nsfw chegaraga yaqin (60 - 10 = 50)
        (scores(45, 52, 3), [], True),
        # nsfl chegaraga yaqin
        (scores(20, 25, 55), [], True),
    ],
)
def test_needs_review_table(class_scores, detections, expected) -> None:
    verdict, _c, _s, _r = scoring.evaluate(class_scores, detections)
    review, _reason = scoring.needs_review(verdict, class_scores, detections)
    assert review is expected


def test_needs_review_does_not_change_verdict() -> None:
    """Bayroq faqat signal — u `is_safe` yoki verdictni o'zgartirmasligi shart."""
    cs = scoring.evaluate(scores(90, 6, 4), [det("FEMALE_BREAST_EXPOSED", 30)])
    verdict = cs[0]
    review, reason = scoring.needs_review(verdict, scores(90, 6, 4), [det("FEMALE_BREAST_EXPOSED", 30)])
    assert verdict is Verdict.SAFE
    assert scoring.is_nsfw(verdict) is False
    assert review is True
    assert reason and "chegaraviy" in reason
