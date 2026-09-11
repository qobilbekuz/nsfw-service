# NSFW Detection API — mijozlar uchun hujjat

**Bazaviy manzil:** `https://api.qobilbek.dev/nsfw`
**Swagger UI:** https://api.qobilbek.dev/nsfw/docs
**Versiya:** 1.0.0

---

## 1. Autentifikatsiya

Har bir so'rovga `X-API-Key` sarlavhasi qo'shiladi (`/v1/health` bundan mustasno).

```bash
curl https://api.qobilbek.dev/nsfw/v1/models -H "X-API-Key: nsfw_..."
```

Kalit olish (server tarafida):
```bash
cd /www/wwwroot/api.qobilbek.dev/nsfw-service
sudo -u www .venv/bin/python scripts/apikey.py create --name "loyiha-nomi" --rpm 120
```

Kalit **faqat yaratilgan paytda bir marta** ko'rsatiladi — Redis'da uning sha256
dayjesti saqlanadi, kalitning o'zi emas.

---

## 2. Javob formati

Barcha javoblar — muvaffaqiyatli yoki xato — bir xil envelope'da qaytadi.

```json
{
  "success": true,
  "request_id": "r_9ec19bd4d29b3bcee62b",
  "took_ms": 34,
  "data": { },
  "error": null
}
```

`success` va `error` hech qachon birga to'ldirilmaydi, shuning uchun mijozga
faqat `success` ni tekshirish yetarli.

Xato holatida:
```json
{
  "success": false,
  "request_id": "r_...",
  "took_ms": 3,
  "data": null,
  "error": {
    "code": "IMAGE_TOO_LARGE",
    "message": "Rasm hajmi 20 MB dan oshmasligi kerak",
    "details": { "size_bytes": 31457280, "limit_bytes": 20971520 }
  }
}
```

---

## 3. `POST /v1/analyze`

Rasmni to'rt xil usulda yuborish mumkin. **Aniq bittasi** berilishi kerak.

### 3.1 URL orqali
```bash
curl -X POST https://api.qobilbek.dev/nsfw/v1/analyze \
  -H "X-API-Key: nsfw_..." \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/rasm.jpg"}'
```

### 3.2 Server yo'li orqali
```bash
curl -X POST https://api.qobilbek.dev/nsfw/v1/analyze \
  -H "X-API-Key: nsfw_..." \
  -H "Content-Type: application/json" \
  -d '{"path": "uploads/2026/08/rasm.jpg"}'
```
Faqat `.env` dagi `ALLOWED_PATH_ROOTS` oq ro'yxatidagi kataloglar ichidan
o'qiydi. Sozlanmagan bo'lsa bu rejim o'chiq (`403 PATH_NOT_ALLOWED`).

### 3.3 Base64 orqali
```bash
curl -X POST https://api.qobilbek.dev/nsfw/v1/analyze \
  -H "X-API-Key: nsfw_..." \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "data:image/jpeg;base64,/9j/4AAQ..."}'
```

### 3.4 Fayl yuklash (multipart)
```bash
curl -X POST https://api.qobilbek.dev/nsfw/v1/analyze \
  -H "X-API-Key: nsfw_..." \
  -F "file=@rasm.jpg" \
  -F 'options={"detect":true,"min_detection_score":25}'
```

### 3.5 Ixtiyoriy parametrlar

| Parametr | Turi | Default | Tavsif |
|----------|------|---------|--------|
| `detect` | bool | `true` | Tana qismlari detektorini ishlatish (bbox bilan) |
| `min_detection_score` | float | `25.0` | Bundan past topilmalar javobga kirmaydi (%) |
| `cache` | bool | `true` | sha256 bo'yicha natijani keshdan olish/keshga yozish |

### 3.6 GET varianti (tez sinov uchun)
```bash
curl "https://api.qobilbek.dev/nsfw/v1/analyze?url=https://example.com/a.jpg&detect=true" \
  -H "X-API-Key: nsfw_..."
```

---

## 4. Javob (`data` bloki)

```json
{
  "verdict": "suggestive",
  "is_nsfw": false,
  "is_safe": true,
  "needs_review": true,
  "confidence": 77.35,
  "scores": {
    "sfw": 19.48,
    "suggestive": 38.28,
    "nsfw": 77.35,
    "nsfl": 3.17
  },
  "detections": [
    {
      "label": "FACE_FEMALE",
      "score": 83.02,
      "box": { "x": 32, "y": 33, "width": 55, "height": 58 }
    }
  ],
  "reasons": [
    "classifier: sfw 19.48% / nsfw 77.35% / nsfl 3.17%",
    "detector: FACE_FEMALE 83.02%",
    "nsfw oraliq zonasi (77.35%), lekin detektor ochiq tana qismini topmadi -> suggestive"
  ],
  "image": {
    "width": 128, "height": 128, "format": "JPEG",
    "size_bytes": 5321,
    "sha256": "a3f1..."
  },
  "models": {
    "classifier": "image-safety-classifier-s",
    "detector": "nudenet-320n"
  },
  "timings_ms": { "fetch": 0, "decode": 0, "classify": 15, "detect": 12, "total": 29 },
  "cached": false
}
```

`models.detector` uchta qiymat oladi: `"nudenet-320n"` (ishladi),
`"disabled"` (`detect=false` berilgan) yoki `"failed"` (detektor xato berdi —
qaror faqat klassifikatorga tayangan, `needs_review` majburan `true`).

### 4.1 `verdict`

| Qiymat | Ma'nosi | `is_nsfw` |
|--------|---------|-----------|
| `safe` | Xavfsiz kontent | `false` |
| `suggestive` | Shahvoniy ishora — bikini, ich kiyim, ochiq yelka | `false` |
| `nsfw` | 18+ — yalang'ochlik / pornografiya | `true` |
| `nsfl` | Qon, zo'ravonlik, jirkanch kontent | `true` |

### 4.2 `needs_review` — chegaraviy holat bayrog'i

`true` bo'lsa, qaror ishonchsiz zonada chiqqan va **qo'lda ko'rib chiqishga
arziydi**. Bayroq `verdict` va `is_safe` ga **ta'sir qilmaydi** — bloklash
qarori sizniki. Qachon qo'yiladi:

| Holat | Sabab |
|-------|-------|
| `verdict = suggestive` | ta'rifiga ko'ra chegaraviy |
| `nsfw` 60–90% oralig'ida chiqqan | qaror detektor tasdig'iga tayangan |
| `nsfl` chegaraga 10 balldan yaqin | zo'rg'a o'tgan |
| `safe`, lekin kuchsiz ochiq-oydin topilma bor | oz farq bilan o'tgan |
| `safe`, lekin `nsfw` chegaraga 10 balldan yaqin | oz farq bilan o'tgan |

Tavsiya etilgan ishlatish:

```python
if data["is_nsfw"]:
    block()
elif data["needs_review"]:
    queue_for_moderator()     # ko'rsatiladi, lekin navbatga ham tushadi
else:
    publish()
```

Bu bayroq ostidagi rasmlar model aniqligini oshirish uchun eng qimmatli
o'qitish materiali — `training/README.md` ga qarang.

> ⚠️ Hozirgi chegaralar **o'lchanmagan**, qo'lda tanlangan. Ma'lum xato:
> ko'kragi ochiq bolakay suratiga klassifikator `nsfw 78%` beradi
> (`suggestive` deb qaytariladi, bloklanmaydi).

### 4.3 `scores` — hammasi foizda (0.00–100.00)

- `sfw + nsfw + nsfl = 100` — bu klassifikatorning softmax chiqishi.
- `suggestive` — **alohida hosila ko'rsatkich**, yuqoridagi yig'indiga kirmaydi.
  U asosan detektorning `COVERED` topilmalaridan hisoblanadi.

### 4.4 `reasons`

Qaror qanday chiqarilganini matn ko'rinishida tushuntiradi. Moderatorlar
paneli va nizoli holatlarni tekshirish uchun mo'ljallangan.

`diqqat:` bilan boshlangan qatorlar — verdictga **ta'sir qilmaydigan**, lekin
natijaning ishonchliligi haqidagi ogohlantirishlar:

| Qator | Ma'nosi | Nima qilish kerak |
|-------|---------|-------------------|
| `diqqat: rasm kichik (...)` | Rasmning kichik tomoni 320px dan past — bu NudeNet kirish o'lchami | Thumbnail emas, **asl rasmni** yuboring (4.4.1) |
| `diqqat: detektor ishlamadi...` | Detektor xato berdi, qaror faqat klassifikatorga tayandi | `models.detector` = `"failed"`, `needs_review` = `true`; loglarni tekshiring |

#### 4.4.1 Thumbnail yubormang — o'lchangan farq

Bitta rasm, ikki o'lcham:

| | 1600×1157 (asl) | 256×185 (thumbnail) |
|---|---|---|
| `FEMALE_BREAST_COVERED` | 56.05% | 51.41% |
| `ARMPITS_EXPOSED` | 46.83% | **topilmadi** |
| `BUTTOCKS_EXPOSED` | 37.48% | **topilmadi** |

Yo'qolgan `BUTTOCKS_EXPOSED` — *ochiq-oydin* sinf. Bu safar ball past bo'lgani
uchun verdict o'zgarmadi (ikkalasi ham `suggestive`), lekin 50% dan yuqori
bo'lganida thumbnail `suggestive`, asl rasm esa `nsfw` qaytargan bo'lardi.

Shu sababli: CDN'da `_thumb` varianti bo'lsa ham, tekshirishga **asl faylni**
bering. Bir postda bir nechta rasm bo'lsa — hammasini tekshiring, faqat
birinchisini emas (`/v1/analyze/batch` shuning uchun bor).

### 4.5 Verdict qanday chiqariladi

1. `nsfl ≥ 50%` → **nsfl**
2. `nsfw ≥ 90%` → **nsfw** (yuqori ishonch, detektor tasdig'i shart emas)
3. `60% ≤ nsfw < 90%` (oraliq zona):
   - detektor ochiq tana qismini topsa (≥50%) → **nsfw**
   - topmasa → **suggestive**
4. Detektorda ochiq tana qismi ≥50% → **nsfw**
5. `suggestive ≥ 25%` → **suggestive**
6. Aks holda → **safe**

> **3-qadam nima uchun kerak.** Klassifikator ataylab qattiqqo'l: mayka
> kiygan yoki yelkasi ochiq oddiy portretga ham 70–80% berishi mumkin
> (bu sinovda tasdiqlangan). Detektor tasdig'ini talab qilish shunday
> rasmlarning 18+ deb bloklanishining oldini oladi, ammo signal
> `suggestive` sifatida saqlanib qoladi.
>
> `detect=false` bilan yuborilganda tasdiqlash imkonsiz — bu holda faqat
> chegaraga tayaniladi va 3-qadam **nsfw** beradi.

### 4.6 Detektor sinflari (18 ta)

`FEMALE_GENITALIA_EXPOSED`, `MALE_GENITALIA_EXPOSED`, `ANUS_EXPOSED`,
`FEMALE_BREAST_EXPOSED`, `BUTTOCKS_EXPOSED`, `FEMALE_GENITALIA_COVERED`,
`FEMALE_BREAST_COVERED`, `BUTTOCKS_COVERED`, `ANUS_COVERED`, `BELLY_EXPOSED`,
`BELLY_COVERED`, `ARMPITS_EXPOSED`, `ARMPITS_COVERED`, `FEET_EXPOSED`,
`FEET_COVERED`, `FACE_FEMALE`, `FACE_MALE`, `MALE_BREAST_EXPOSED`

---

## 5. `POST /v1/analyze/batch`

Bir so'rovda 20 tagacha rasm, 8 tasi bir vaqtda qayta ishlanadi.

```bash
curl -X POST https://api.qobilbek.dev/nsfw/v1/analyze/batch \
  -H "X-API-Key: nsfw_..." \
  -H "Content-Type: application/json" \
  -d '{"items":[
        {"id":"a","url":"https://example.com/1.jpg"},
        {"id":"b","image_base64":"/9j/4AAQ..."},
        {"id":"c","path":"uploads/3.jpg"}
      ]}'
```

Javob:
```json
{
  "count": 3, "succeeded": 2, "failed": 1,
  "results": [
    {"id":"a","index":0,"success":true,"data":{ }},
    {"id":"c","index":2,"success":false,
     "error":{"code":"PATH_NOT_ALLOWED","message":"...","details":null}}
  ]
}
```

> Bitta element xato bo'lsa ham butun so'rov **HTTP 200** qaytaradi —
> xatolar element darajasida beriladi.

---

## 6. Xizmat endpointlari

### `GET /v1/health` (auth talab qilinmaydi)
```json
{"status":"ok","version":"1.0.0","models_loaded":true,"redis":"ok","uptime_s":3600}
```

### `GET /v1/models`
Yuklangan modellar, limitlar va joriy chegaralarni qaytaradi.

---

## 7. HTTP statuslar va xato kodlari

| HTTP | `error.code` | Sabab |
|------|-------------|-------|
| 400 | `INVALID_REQUEST` | Manba berilmagan / bir nechta manba / buzuq JSON |
| 400 | `INVALID_URL` | Faqat `http` va `https` qo'llab-quvvatlanadi |
| 401 | `UNAUTHORIZED` | Kalit yo'q, noto'g'ri yoki bekor qilingan |
| 403 | `FORBIDDEN_TARGET` | URL ichki/privat IP ga ishora qiladi (SSRF himoyasi) |
| 403 | `PATH_NOT_ALLOWED` | Yo'l oq ro'yxatdan tashqarida yoki rejim o'chiq |
| 404 | `NOT_FOUND` | Bunday endpoint yo'q |
| 413 | `IMAGE_TOO_LARGE` | Rasm 20 MB dan katta |
| 415 | `UNSUPPORTED_MEDIA_TYPE` | Format qo'llab-quvvatlanmaydi |
| 422 | `DECODE_FAILED` | Fayl buzilgan yoki rasm emas |
| 422 | `IMAGE_TOO_LARGE_PIXELS` | 50 megapikseldan katta (bomba himoyasi) |
| 429 | `RATE_LIMITED` | Limit oshdi — `Retry-After` sarlavhasiga qarang |
| 500 | `INTERNAL_ERROR` | Ichki xato |
| 502 | `FETCH_FAILED` | URL yuklanmadi (DNS, ulanish, upstream xatosi) |
| 504 | `FETCH_TIMEOUT` | URL yuklash vaqti tugadi (15 s) |

---

## 8. Sarlavhalar

| Sarlavha | Yo'nalish | Tavsif |
|----------|-----------|--------|
| `X-API-Key` | so'rov | Autentifikatsiya |
| `X-Request-ID` | ikkala | Mijoz bersa saqlanadi, aks holda generatsiya qilinadi |
| `X-RateLimit-Limit` | javob | Daqiqadagi limit |
| `X-RateLimit-Remaining` | javob | Qolgan so'rovlar |
| `X-RateLimit-Reset` | javob | Oyna necha soniyada yangilanadi |
| `Retry-After` | javob (429) | Necha soniyadan keyin urinish |

---

## 9. Limitlar

| Limit | Qiymat |
|-------|--------|
| Maksimal rasm hajmi | 20 MB |
| Maksimal piksel | 50 000 000 |
| Formatlar | JPEG, PNG, WEBP, GIF (1-kadr), BMP |
| Batch elementlari | 20 |
| URL yuklash timeouti | 15 s (connect 5 s, read 10 s) |
| Redirect'lar | 3 tagacha, har biri qayta tekshiriladi |
| Standart rate-limit | 60 so'rov/daqiqa (kalitga qarab sozlanadi) |
| Natija keshi | 7 kun (sha256 bo'yicha) |

---

## 10. Namuna kodlar

### Python
```python
import httpx

BASE = "https://api.qobilbek.dev/nsfw"
HEADERS = {"X-API-Key": "nsfw_..."}

def is_safe(url: str) -> bool:
    r = httpx.post(f"{BASE}/v1/analyze", headers=HEADERS, json={"url": url}, timeout=60)
    body = r.json()
    if not body["success"]:
        raise RuntimeError(body["error"]["code"] + ": " + body["error"]["message"])
    data = body["data"]
    print(f"{data['verdict']} — nsfw {data['scores']['nsfw']}%")
    return data["is_safe"]
```

### PHP
```php
$ch = curl_init("https://api.qobilbek.dev/nsfw/v1/analyze");
curl_setopt_array($ch, [
    CURLOPT_POST => true,
    CURLOPT_RETURNTRANSFER => true,
    CURLOPT_HTTPHEADER => ["X-API-Key: nsfw_...", "Content-Type: application/json"],
    CURLOPT_POSTFIELDS => json_encode(["url" => $imageUrl]),
    CURLOPT_TIMEOUT => 60,
]);
$body = json_decode(curl_exec($ch), true);
if (!$body["success"]) {
    throw new RuntimeException($body["error"]["code"]);
}
$isNsfw = $body["data"]["is_nsfw"];
$percent = $body["data"]["scores"]["nsfw"];
```

### Node.js
```js
const res = await fetch("https://api.qobilbek.dev/nsfw/v1/analyze", {
  method: "POST",
  headers: { "X-API-Key": "nsfw_...", "Content-Type": "application/json" },
  body: JSON.stringify({ url: imageUrl }),
});
const body = await res.json();
if (!body.success) throw new Error(body.error.code);
console.log(body.data.verdict, body.data.scores.nsfw + "%");
```

---

## 11. Maxfiylik

- Rasmlar **diskka yozilmaydi** — barcha ish xotirada bajariladi.
- Loglarda URL, fayl yo'li va rasm mazmuni **saqlanmaydi**; faqat
  `request_id`, verdict, davomiylik va kalit identifikatori qayd etiladi.
- Keshda faqat sha256 dayjest kalit sifatida va tahlil natijasi (7 kun)
  saqlanadi, rasmning o'zi emas.
