"""Qaror sabablari — kod + parametrlar, matn esa uch tilda.

Ilgari `reasons` faqat o'zbekcha ERKIN MATN edi: mijoz uni ko'rsata
olardi-yu, ustida qaror qabul qila olmasdi, ingliz yoki rus tilidagi
interfeysda esa u begona ko'rinardi. Endi har bir sabab `code` va
`params` bilan keladi, matn esa xizmatning xato konvensiyasi bilan bir
xil — `messages: {en, uz, ru}`.
"""

from __future__ import annotations

from typing import Any

LOCALES = ("en", "uz", "ru")

TEXTS: dict[str, dict[str, str]] = {
    "classifier_scores": {
        "en": "Classifier: sfw {sfw}% / nsfw {nsfw}% / nsfl {nsfl}%",
        "uz": "Klassifikator: sfw {sfw}% / nsfw {nsfw}% / nsfl {nsfl}%",
        "ru": "Классификатор: sfw {sfw}% / nsfw {nsfw}% / nsfl {nsfl}%",
    },
    "detector_top": {
        "en": "Detector: {label} {score}%",
        "uz": "Detektor: {label} {score}%",
        "ru": "Детектор: {label} {score}%",
    },
    "detector_explicit": {
        "en": "Detector, explicit class: {label} {score}%",
        "uz": "Detektor, ochiq-oydin sinf: {label} {score}%",
        "ru": "Детектор, явный класс: {label} {score}%",
    },
    "nsfl_threshold": {
        "en": "nsfl is over the threshold ({score}% >= {threshold}%)",
        "uz": "nsfl chegarasi oshdi ({score}% >= {threshold}%)",
        "ru": "nsfl превысил порог ({score}% >= {threshold}%)",
    },
    "nsfw_confident": {
        "en": "nsfw with high confidence ({score}% >= {threshold}%)",
        "uz": "nsfw yuqori ishonch bilan ({score}% >= {threshold}%)",
        "ru": "nsfw с высокой уверенностью ({score}% >= {threshold}%)",
    },
    "nsfw_confirmed": {
        "en": "nsfw middle zone ({score}%) confirmed by the detector ({peak}%)",
        "uz": "nsfw oraliq zonasi ({score}%) detektor bilan tasdiqlandi ({peak}%)",
        "ru": "средняя зона nsfw ({score}%) подтверждена детектором ({peak}%)",
    },
    "nsfw_no_detector": {
        "en": "nsfw is over the threshold ({score}%), the detector is off",
        "uz": "nsfw chegarasi oshdi ({score}%), detektor o'chirilgan",
        "ru": "nsfw превысил порог ({score}%), детектор отключён",
    },
    "nsfw_weak_detection": {
        "en": "nsfw middle zone ({score}%); an explicit body part was found "
              "but with low confidence ({peak}% < {promote}%) -> suggestive",
        "uz": "nsfw oraliq zonasi ({score}%); ochiq tana qismi topildi, lekin "
              "ishonchi past ({peak}% < {promote}%) -> suggestive",
        "ru": "средняя зона nsfw ({score}%); явная часть тела найдена, но с "
              "низкой уверенностью ({peak}% < {promote}%) -> suggestive",
    },
    "nsfw_no_detection": {
        "en": "nsfw middle zone ({score}%), the detector found no explicit "
              "body part -> suggestive",
        "uz": "nsfw oraliq zonasi ({score}%), lekin detektor ochiq tana "
              "qismini topmadi -> suggestive",
        "ru": "средняя зона nsfw ({score}%), детектор не нашёл явных частей "
              "тела -> suggestive",
    },
    "explicit_detection": {
        "en": "The detector found an explicit body part ({peak}%)",
        "uz": "Detektorda ochiq tana qismi ({peak}%)",
        "ru": "Детектор нашёл явную часть тела ({peak}%)",
    },
    "suggestive_hint": {
        "en": "Suggestive content ({score}%)",
        "uz": "Shahvoniy ishora ({score}%)",
        "ru": "Намёк на откровенность ({score}%)",
    },
    "review_suggestive": {
        "en": "Borderline: suggestive",
        "uz": "Chegaraviy: suggestive",
        "ru": "Пограничный случай: suggestive",
    },
    "review_nsfw_middle": {
        "en": "Borderline: nsfw is in the middle zone ({score}%)",
        "uz": "Chegaraviy: nsfw oraliq zonada ({score}%)",
        "ru": "Пограничный случай: nsfw в средней зоне ({score}%)",
    },
    "review_nsfl_near": {
        "en": "Borderline: nsfl is close to the threshold ({score}%)",
        "uz": "Chegaraviy: nsfl chegaraga yaqin ({score}%)",
        "ru": "Пограничный случай: nsfl близок к порогу ({score}%)",
    },
    "review_weak_explicit": {
        "en": "Borderline: a weak explicit detection ({peak}%)",
        "uz": "Chegaraviy: kuchsiz ochiq-oydin topilma ({peak}%)",
        "ru": "Пограничный случай: слабое явное срабатывание ({peak}%)",
    },
    "review_nsfw_near": {
        "en": "Borderline: nsfw is close to the threshold ({score}%)",
        "uz": "Chegaraviy: nsfw chegaraga yaqin ({score}%)",
        "ru": "Пограничный случай: nsfw близок к порогу ({score}%)",
    },
    "small_image": {
        "en": "Note: the image is small ({width}x{height} < {input}px) — "
              "detections get weaker, send the original if you can",
        "uz": "Diqqat: rasm kichik ({width}x{height} < {input}px) — detektor "
              "topilmalari zaiflashadi, iloji bo'lsa asl rasmni yuboring",
        "ru": "Внимание: изображение маленькое ({width}x{height} < {input}px) "
              "— срабатывания слабеют, по возможности пришлите оригинал",
    },
}


def render(code: str, params: dict[str, Any]) -> dict[str, Any]:
    """`{code, params, messages}` — noma'lum kod HECH QACHON jim o'tmaydi."""
    table = TEXTS.get(code)
    if table is None:
        raise KeyError(f"reasons.TEXTS da yo'q kod: {code}")
    return {
        "code": code,
        "params": params,
        "messages": {loc: table[loc].format(**params) for loc in LOCALES},
    }
