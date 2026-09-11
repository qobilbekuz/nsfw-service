# NSFW Detection API — To'liq Task Hujjati

**Loyiha:** 18+ (NSFW/NSFL) rasm aniqlovchi HTTP servis
**Domen:** `https://api.qobilbek.dev/nsfw/`
**Katalog:** `/www/wwwroot/api.qobilbek.dev/nsfw-service`
**Sana:** 2026-08-11

---

## 0. BO'LIM — Maqsad va qamrov

### 0.1 Nima quriladi
Rasmni qabul qilib, uning 18+ (kattalar uchun) kontent ekanligini **foiz (%)** ko'rinishida
baholab beradigan, o'z serverimizda ishlaydigan REST API.

### 0.2 Asosiy talablar (foydalanuvchi bergan)
| # | Talab | Yechim |
|---|-------|--------|
| T1 | Javoblar **foizda** (`nechi %`) | Barcha ballar `0.00–100.00` oralig'ida, 2 kasr raqam |
| T2 | **Standart JSON** javob formati | Yagona envelope: `success / data / error / request_id / took_ms` |
| T3 | Rasmni **URL** dan qabul qilish | `{"url": "https://..."}` — SSRF himoyasi bilan |
| T4 | Rasmni **path** dan qabul qilish | `{"path": "..."}` — faqat oq ro'yxatdagi kataloglardan |
| T5 | Standart HTTP status kodlari | 200/400/401/403/413/415/422/429/500/502/504 |
| T6 | `api.qobilbek.dev` da ishga tushirish | nginx `^~ /nsfw/` → uvicorn `127.0.0.1:8720` |

### 0.3 Qamrovdan tashqarida (v1 da yo'q)
- Video / GIF kadrlarini tahlil qilish (v2 rejasi — §9)
- Matn moderatsiyasi, OCR
- Yuz tanish / shaxsni aniqlash
- Rasmni saqlash yoki arxivlash (**ataylab**: maxfiylik uchun hech narsa diskka yozilmaydi)

---

## 1. BO'LIM — Model tadqiqi va tanlov

### 1.1 Ko'rib chiqilgan variantlar

| Model | Turi | Hajm | Litsenziya | Aniqlik | CPU tezligi | Qaror |
|-------|------|------|-----------|---------|-------------|-------|
| **OwenElliott/image-safety-classifier-s** | SwiftFormer-S klassifikator | 23 MB ONNX | MIT | 97.99% (~320k rasm) | **7 ms** | ✅ **Asosiy** |
| **NudeNet 3.4.2 (320n)** | YOLOv8 detektor | 12 MB ONNX | MIT | — (18 sinf, bbox) | **10 ms** | ✅ **Asosiy** |
| AdamCodd/vit-base-nsfw-detector | ViT-base-384 | 344 MB (int8: 88 MB) | Apache-2.0 | 96.54% | ~150 ms | ⚙️ Ixtiyoriy ansambl |
| Falconsai/nsfw_image_detection | ViT-base | ~350 MB | Apache-2.0 | 98% (o'z dataseti) | ~150 ms | ❌ ONNX yo'q, torch kerak |
| Marqo/nsfw-image-detection-384 | ViT-tiny-384 | ~22 MB | Apache-2.0 | 98.41% | ~15 ms | ❌ ONNX yo'q, timm+torch kerak |
| opennsfw2 / Yahoo OpenNSFW | ResNet-50 | ~23 MB | BSD | ~90% (eskirgan) | ~40 ms | ❌ Eskirgan (2016) |
| nsfwjs | TF.js | 4 MB | MIT | ~93% | — | ❌ Node stack, kerak emas |

### 1.2 Nima uchun shu ikkitasi

**`image-safety-classifier-s` (asosiy klassifikator)**
- **Uch sinf**: `NSFW` (pornografiya/shahvoniy), `NSFL` (qon, zo'ravonlik), `SFW`.
  Raqobatchilarning barchasi faqat 2 sinf beradi — NSFL alohida kategoriya katta ustunlik.
- Preprocessing (normalize + softmax) **ONNX ichiga pishirilgan** → Python tarafda xato qilish
  ehtimoli nolga tushadi, faqat resize + NCHW float32 (0–255) kerak.
- 6.1M parametr, torch kerak emas — sof `onnxruntime`.

**`NudeNet 320n` (detektor)**
- Klassifikator faqat "nechi %" beradi, lekin **nima uchun** ekanini tushuntirmaydi.
  NudeNet 18 ta tana qismini bbox bilan qaytaradi → javob **izohlanadigan** (explainable) bo'ladi.
- `EXPOSED` va `COVERED` farqi bor → `suggestive` (yarim ochiq) darajasini ajratishga imkon beradi.
- Kelajakda blur/senzura funksiyasi uchun bbox allaqachon tayyor.

**Ikkalasi birga = 17 ms/rasm, GPU'siz, 35 MB model.** Serverda GPU yo'q (tekshirildi),
shuning uchun yengil ONNX modellari yagona to'g'ri tanlov.

### 1.3 Tekshirilgan faktlar (bu serverda ishga tushirib sinaldi)
```
image-safety-classifier-s: IN  image [batch,3,224,224] float32 (0–255)
                           OUT probabilities [batch,3] → [NSFL, NSFW, SFW]
                           latency: 7.0 ms/rasm (CPU, 20 yadro)
NudeNet 320n:              IN  images [batch,3,H,W]
                           OUT output0 [batch,22,N]  (4 bbox + 18 sinf)
                           latency: 9.6 ms/rasm
```

---

## 2. BO'LIM — Arxitektura

```
        Internet
           │  HTTPS
     ┌─────▼──────────────────────────────┐
     │  nginx  api.qobilbek.dev           │
     │  location ^~ /nsfw/  → 127.0.0.1:8720
     │  client_max_body_size 20m          │
     └─────┬──────────────────────────────┘
           │
     ┌─────▼──────────────────────────────┐
     │  uvicorn (systemd: nsfw-api)       │
     │  --workers 3  --root-path /nsfw    │
     │  User=www                          │
     └─────┬──────────────────────────────┘
           │
     ┌─────▼───────────────┐   ┌──────────────┐
     │  FastAPI app        │──▶│  Redis :6379 │  auth kalitlari,
     │  ├ input layer      │   │  DB 4        │  rate-limit, natija keshi
     │  ├ guard layer      │   └──────────────┘
     │  ├ inference layer  │
     │  └ scoring layer    │
     └─────┬───────────────┘
           │
     ┌─────▼──────────────────────────────┐
     │  onnxruntime (CPU, 2 intra-thread) │
     │  ├ classifier: safety-s (23 MB)    │
     │  └ detector:   nudenet 320n (12 MB)│
     └────────────────────────────────────┘
```

### 2.1 Katalog tuzilishi
```
nsfw-service/
├── app/
│   ├── main.py              # FastAPI app, lifespan, middleware
│   ├── config.py            # pydantic-settings (.env)
│   ├── schemas.py           # so'rov/javob Pydantic modellari
│   ├── envelope.py          # standart JSON envelope + xato kodlari
│   ├── api/
│   │   ├── routes.py        # /v1/* endpointlar
│   │   └── deps.py          # auth, rate-limit dependency'lari
│   ├── core/
│   │   ├── fetcher.py       # URL yuklash + SSRF himoyasi
│   │   ├── loader.py        # path/base64/upload → bytes
│   │   └── imaging.py       # dekod, EXIF, bomba himoyasi
│   ├── services/
│   │   ├── classifier.py    # safety-classifier-s ONNX
│   │   ├── detector.py      # NudeNet ONNX
│   │   ├── scoring.py       # verdict/foiz mantiq
│   │   └── cache.py         # Redis sha256 keshi
│   └── models/              # (bo'sh — ONNX fayllar ../models/ da)
├── models/                  # .onnx fayllar (git'ga kirmaydi)
├── deploy/
│   ├── nsfw-api.service
│   └── nginx.conf
├── tests/
├── docs/
│   ├── TASKS.md             # ← shu fayl
│   └── API.md               # mijozlar uchun hujjat
├── .env / .env.example
└── requirements.txt
```

---

## 3. BO'LIM — API spetsifikatsiyasi

### 3.1 Standart javob envelope (BARCHA endpointlar)

**Muvaffaqiyat (HTTP 200):**
```json
{
  "success": true,
  "request_id": "r_01k2f8h9m3xq7v",
  "took_ms": 34,
  "data": { },
  "error": null
}
```

**Xato (HTTP 4xx/5xx):**
```json
{
  "success": false,
  "request_id": "r_01k2f8h9m3xq7v",
  "took_ms": 3,
  "data": null,
  "error": {
    "code": "IMAGE_TOO_LARGE",
    "message": "Rasm hajmi 20 MB dan oshmasligi kerak",
    "details": { "size_bytes": 31457280, "limit_bytes": 20971520 }
  }
}
```

> **Qoida:** `success` va `error` hech qachon bir vaqtda to'ldirilmaydi.
> Mijoz faqat `success` ni tekshirsa yetarli. `request_id` har bir javobda —
> log bilan solishtirish uchun (`X-Request-ID` sarlavhasida ham qaytadi).

### 3.2 `data` bloki (analyze javoblari uchun)

```json
{
  "verdict": "nsfw",
  "is_nsfw": true,
  "is_safe": false,
  "confidence": 97.45,
  "scores": {
    "sfw":        2.11,
    "suggestive": 0.00,
    "nsfw":      97.45,
    "nsfl":       0.44
  },
  "detections": [
    {
      "label": "FEMALE_BREAST_EXPOSED",
      "score": 91.20,
      "box": { "x": 210, "y": 88, "width": 164, "height": 152 }
    }
  ],
  "reasons": [
    "classifier: nsfw 97.45%",
    "detector: FEMALE_BREAST_EXPOSED 91.20%"
  ],
  "image": {
    "width": 1024, "height": 768, "format": "JPEG",
    "size_bytes": 204800,
    "sha256": "e3b0c44298fc1c149afbf4c8996fb924..."
  },
  "models": {
    "classifier": "image-safety-classifier-s",
    "detector": "nudenet-320n@3.4.2"
  },
  "timings_ms": { "fetch": 12, "decode": 3, "classify": 7, "detect": 10, "total": 34 },
  "cached": false
}
```

**`verdict` qiymatlari (4 daraja):**
| Qiymat | Ma'nosi | `is_nsfw` |
|--------|---------|-----------|
| `safe` | Xavfsiz kontent | `false` |
| `suggestive` | Shahvoniy ishora, yopiq/yarim ochiq (bikini, ich kiyim) | `false` |
| `nsfw` | 18+ — yalang'ochlik/pornografiya | `true` |
| `nsfl` | Qon, zo'ravonlik, jirkanch kontent | `true` |

### 3.3 Endpointlar

| Metod | Yo'l | Tavsif |
|-------|------|--------|
| `POST` | `/v1/analyze` | Bitta rasm tahlili (4 xil kirish rejimi) |
| `GET`  | `/v1/analyze` | `?url=...` — tez sinov uchun qulay |
| `POST` | `/v1/analyze/batch` | Ko'p rasm (maks. 20 ta), parallel |
| `GET`  | `/v1/health` | Liveness/readiness — auth talab qilinmaydi |
| `GET`  | `/v1/models` | Yuklangan modellar + chegaralar ro'yxati |
| `GET`  | `/docs` | Swagger UI (`--root-path /nsfw` bilan to'g'ri ishlaydi) |

### 3.4 Kirish rejimlari (`POST /v1/analyze`)

| # | Rejim | Content-Type | Namuna |
|---|-------|--------------|--------|
| 1 | **URL** | `application/json` | `{"url": "https://example.com/a.jpg"}` |
| 2 | **Path** | `application/json` | `{"path": "uploads/2026/08/a.jpg"}` |
| 3 | **Base64** | `application/json` | `{"image_base64": "/9j/4AAQ..."}` (data-URI ham) |
| 4 | **Fayl** | `multipart/form-data` | `file=@a.jpg` |

Ixtiyoriy parametrlar (barcha rejimlarda):
```json
{
  "detect": true,          // NudeNet detektorini ishlatish (default: true)
  "min_detection_score": 25.0,  // % — bundan pastini qaytarmaslik
  "cache": true            // sha256 bo'yicha keshdan o'qish/yozish
}
```

### 3.5 HTTP status kodlari va xato kodlari

| HTTP | `error.code` | Qachon |
|------|-------------|--------|
| 400 | `INVALID_REQUEST` | Kirish rejimi berilmagan yoki bir nechtasi birga berilgan |
| 400 | `INVALID_URL` | URL sxemasi noto'g'ri (faqat http/https) |
| 401 | `UNAUTHORIZED` | `X-API-Key` yo'q yoki noto'g'ri |
| 403 | `FORBIDDEN_TARGET` | SSRF: ichki/privat IP manzilga so'rov |
| 403 | `PATH_NOT_ALLOWED` | Path oq ro'yxatdagi katalogdan tashqarida |
| 404 | `FILE_NOT_FOUND` | Path bo'yicha fayl topilmadi |
| 413 | `IMAGE_TOO_LARGE` | Rasm hajmi limitdan katta |
| 415 | `UNSUPPORTED_MEDIA_TYPE` | Qo'llab-quvvatlanmaydigan format |
| 422 | `DECODE_FAILED` | Fayl rasm emas yoki buzilgan |
| 422 | `IMAGE_TOO_LARGE_PIXELS` | Dekompressiya bombasi (piksel limiti) |
| 429 | `RATE_LIMITED` | Limit oshdi (`Retry-After` sarlavhasi bilan) |
| 500 | `INTERNAL_ERROR` | Kutilmagan xato |
| 502 | `FETCH_FAILED` | URL yuklanmadi (DNS/ulanish/HTTP xatosi) |
| 504 | `FETCH_TIMEOUT` | URL yuklash vaqti tugadi |

### 3.6 Standart sarlavhalar
| Sarlavha | Yo'nalish | Tavsif |
|----------|-----------|--------|
| `X-API-Key` | so'rov | Autentifikatsiya kaliti |
| `X-Request-ID` | ikkala | Mijoz bersa saqlanadi, bermasa generatsiya qilinadi |
| `X-RateLimit-Limit` / `-Remaining` / `-Reset` | javob | Joriy limit holati |
| `Retry-After` | javob (429) | Necha soniyadan keyin urinish |

---

## 4. BO'LIM — Xavfsizlik

> Bu server 25+ sayt va ichki servisni (redis, postfix, aaPanel :8888, PowerDNS)
> host qiladi. URL-dan rasm yuklaydigan API — bu **SSRF vektori**. Quyidagilar
> ixtiyoriy emas, majburiy.

### 4.1 SSRF himoyasi (`core/fetcher.py`)
- [x] Faqat `http` / `https` sxemalari
- [x] DNS **oldindan** hal qilinadi (`getaddrinfo`), har bir IP tekshiriladi:
      privat (10/8, 172.16/12, 192.168/16), loopback, link-local (169.254 — **AWS/Hetzner
      metadata**), CGNAT (100.64/10), multicast, reserved, IPv6 ekvivalentlari — **rad etiladi**
- [x] **DNS rebinding** ga qarshi: tekshirilgan IP ga to'g'ridan-to'g'ri ulanish
      (`httpx` transport orqali IP-pinning), `Host` sarlavhasi asl domen bilan
- [x] Redirect'lar **qo'lda** kuzatiladi (maks. 3) — har bir qadam qayta tekshiriladi
      (`follow_redirects=False` + o'zimiz boshqaramiz)
- [x] Yuklash: `stream=True`, `Content-Length` va **haqiqiy o'qilgan baytlar** limitga qarab
      uziladi (yolg'on `Content-Length` ga ishonilmaydi)
- [x] Timeout: connect 5s, read 10s, umumiy 15s
- [x] Faqat `image/*` MIME + haqiqiy magic-bytes tekshiruvi

### 4.2 Path traversal himoyasi (`core/loader.py`)
- [x] `.env` dagi `ALLOWED_PATH_ROOTS` oq ro'yxati (default: bo'sh = path rejimi o'chiq)
- [x] `Path.resolve()` + `is_relative_to()` — symlink'dan keyin ham tekshiriladi
- [x] Symlink'lar rad etiladi (`lstat` bilan)
- [x] Faqat oddiy fayllar (`is_file()`), qurilma/FIFO emas

### 4.3 Rasm xavfsizligi (`core/imaging.py`)
- [x] `Image.MAX_IMAGE_PIXELS` = 50M (dekompressiya bombasi)
- [x] Bayt limiti: 20 MB (`MAX_IMAGE_BYTES`)
- [x] Ruxsat etilgan formatlar: JPEG, PNG, WEBP, GIF (1-kadr), BMP
- [x] Animatsiyali GIF/WEBP — faqat birinchi kadr
- [x] EXIF orientatsiyasi qo'llanadi, qolgan metadata tashlanadi
- [x] **Rasm hech qachon diskka yozilmaydi** — faqat xotirada

### 4.4 Autentifikatsiya va limitlar (`api/deps.py`)
- [x] `X-API-Key` → Redis `nsfw:key:<sha256>` (ochiq matnda saqlanmaydi)
- [x] Kalitlarni boshqarish: `scripts/apikey.py` (create / list / revoke)
- [x] Rate limit: kalit bo'yicha sliding-window (default 60 so'rov/daqiqa),
      Redis Lua skripti bilan atomar
- [x] `PUBLIC_MODE=false` — auth majburiy (default). Ichki foydalanish uchun
      IP oq ro'yxati (`TRUSTED_IPS`) ham bor
- [x] `/v1/health` — auth talab qilmaydi (monitoring uchun)

### 4.5 Servis darajasi
- [x] systemd: `User=www`, `NoNewPrivileges`, `ProtectSystem=full`, `PrivateTmp`
- [x] uvicorn `--forwarded-allow-ips=127.0.0.1` (X-Forwarded-For soxtalashtirishga qarshi)
- [x] `--no-server-header`
- [x] `.env` fayl `chmod 600`, `www:www`
- [x] nginx: `client_max_body_size 20m`, `/nsfw/` uchun `deny_sensitive.conf` amal qiladi

---

## 5. BO'LIM — Ishlash va kesh

- [x] ONNX sessiyalari **bir marta** yuklanadi (FastAPI `lifespan`), worker'lar orasida bo'lishilmaydi
- [x] `intra_op_num_threads=2`, `inter_op_num_threads=1` — 3 worker × 2 thread = 6 yadro
      (20 yadroli serverda boshqa servislarga joy qoladi)
- [x] Inference `run_in_threadpool` da (event loop bloklanmaydi)
- [x] Redis kesh: `nsfw:res:<sha256>` → natija, TTL 7 kun.
      Bir xil rasm ikkinchi marta kelsa **~1 ms**
- [x] Batch: `asyncio.Semaphore` bilan bir vaqtda maks. 8 ta
- [x] **O'lchangan natijalar** (jonli servisda, HTTPS orqali, 2026-08-11):

| Stsenariy | p50 | p95 | Maqsad | Izoh |
|-----------|-----|-----|--------|------|
| base64 / fayl yuklash (1024×768) | 42 ms | **63 ms** | <40 ms | Maqsaddan yuqori: TLS qo'l siqish + base64 dekod ham shu vaqt ichida |
| URL yuklash (tashqi sayt) | 170 ms | **251 ms** | <150 ms | Vaqtning ~150 ms i tashqi saytdan yuklashga ketadi, bu bizga bog'liq emas |
| Kesh urishi (bir xil rasm) | 4 ms | 52 ms | — | Model umuman ishlamaydi |

  Sof inference (tarmoqsiz, `timings_ms` bo'yicha): klassifikator ~15 ms +
  detektor ~12 ms = **~27 ms**. Ya'ni qolgan vaqt tarmoq va dekodga ketadi.
  Maqsadlar dastlab tarmoqsiz o'lchov uchun qo'yilgan edi; uchdan-uchga
  o'lchovda ular optimistik bo'lib chiqdi.

---

## 6. BO'LIM — Deploy

### 6.1 Port va marshrut
| Element | Qiymat |
|---------|--------|
| Ichki port | `127.0.0.1:8720` (bo'sh ekani tekshirildi) |
| Tashqi yo'l | `https://api.qobilbek.dev/nsfw/` |
| systemd unit | `nsfw-api.service` |
| Foydalanuvchi | `www:www` |

### 6.2 nginx bloki (mavjud `api.qobilbek.dev.conf` ga qo'shiladi)
```nginx
location = /nsfw { return 301 /nsfw/; }

location ^~ /nsfw/ {
    proxy_pass http://127.0.0.1:8720/;
    include proxy_api_params.conf;
    client_max_body_size 20m;
}
```
> `^~` majburiy — aks holda `\.(jpg|png)$` va `deny_sensitive.conf` regex'lari
> API yo'llarini o'g'irlab ketadi (`stars/` da xuddi shu muammo hal qilingan).

### 6.3 Deploy qadamlari
- [x] `.env` yaratish (`.env.example` dan), `chmod 600`
- [x] `chown -R www:www` butun katalog
- [x] `deploy/nsfw-api.service` → `/etc/systemd/system/`, `daemon-reload`, `enable --now`
- [x] nginx blokini qo'shish → `nginx -t` → `nginx -s reload`
- [x] Birinchi API kalitini yaratish (`scripts/apikey.py create`)
- [x] Tashqaridan smoke-test (health + real rasm)

---

## 7. BO'LIM — Kuzatuv va loglar

- [x] Loglar journald'ga (`journalctl -u nsfw-api -f`), `SyslogIdentifier=nsfw-api`
- [x] Har bir so'rov uchun bitta strukturaviy log qatori: `request_id`, `mode`,
      `verdict`, `took_ms`, `key_id`, `status` — **URL va rasm mazmuni loglanmaydi**
- [x] `/v1/health` → `{"status":"ok","models_loaded":true,"redis":"ok","uptime_s":...}`
- [x] Serverdagi mavjud Telegram health-check monitoringiga `/nsfw/v1/health` ni qo'shish
      (edugames hardening skripti bilan bir xil uslubda)

---

## 8. BO'LIM — Testlar

- [x] `tests/test_envelope.py` — javob formati barcha yo'llarda bir xilmi
- [x] `tests/test_ssrf.py` — `127.0.0.1`, `169.254.169.254`, `10.x`, `[::1]`,
      `localhost`, redirect→privat IP — hammasi 403
- [x] `tests/test_path.py` — `../../etc/passwd`, symlink, oq ro'yxatdan tashqari → 403
- [x] `tests/test_limits.py` — katta fayl → 413, rasm bo'lmagan fayl → 422, piksel bombasi → 422
- [x] `tests/test_scoring.py` — verdict mantiqi (chegaralar bo'yicha jadval testi)
- [x] `tests/test_api.py` — 4 ta kirish rejimi, batch, health, auth, rate-limit
- [x] Qo'lda: haqiqiy SFW rasmlar to'plamida false-positive tekshiruvi
      (13 ta rasm: manzara, natyurmort, shahar, 5 ta portret).
      **Topilgan muammo:** klassifikator ochiq yelkali oddiy portretga
      `nsfw 77%` bergan. Shu sababli §3.2 dagi "oraliq zona + detektor
      tasdig'i" mantiqi qo'shildi va portret `suggestive` ga tushdi.
      ⚠️ **Cheklov:** haqiqiy 18+ rasmlar bilan sinov O'TKAZILMAGAN —
      bunday to'plam serverga yuklanmadi. Ya'ni **recall** (18+ ni
      qo'ldan chiqarmaslik) faqat model mualliflarining ko'rsatkichlariga
      tayanadi, o'zimiz o'lchamadik. Ishlab chiqarishga to'liq ishonch
      uchun yopiq muhitda nazorat to'plami bilan sinov kerak.

---

## 9. BO'LIM — Kelajak bosqichlari (v1 dan keyin)

| Bosqich | Tavsif |
|---------|--------|
| v1.1 | `POST /v1/censor` — aniqlangan joylarni blur qilib, rasmni qaytarish (bbox tayyor) |
| v1.2 | Ansambl rejimi — `AdamCodd/vit-base` int8 ni ikkinchi fikr sifatida (`?strict=true`) |
| v1.3 | Video/GIF — kadrlarni namuna olib tahlil qilish |
| v1.4 | Webhook — asinxron rejim (`arq` worker, `cdn.edugames.uz` uslubida) |
| v1.5 | Statistika paneli — kunlik so'rov/verdict taqsimoti |

---

## 10. BO'LIM — Bajarilish tartibi (checklist)

**Bosqich 1 — Poydevor**
- [x] Serverni tekshirish (CPU/RAM/GPU/portlar) — GPU yo'q, 20 yadro, 62 GB
- [x] Model tadqiqi va tanlov
- [x] venv + kutubxonalar
- [x] Modellarni yuklab, ONNX imzosi va tezligini o'lchash

**Bosqich 2 — Yadro**
- [x] `config.py`, `envelope.py`, `schemas.py`
- [x] `core/imaging.py`, `core/fetcher.py` (SSRF), `core/loader.py` (path)
- [x] `services/classifier.py`, `services/detector.py`, `services/scoring.py`, `services/cache.py`

**Bosqich 3 — API**
- [x] `api/deps.py` (auth + rate limit), `api/routes.py`, `main.py`

**Bosqich 4 — Deploy**
- [x] systemd unit, nginx bloki, API kaliti, smoke-test

**Bosqich 5 — Sifat**
- [x] Testlar, `docs/API.md`, monitoringga ulash

---

## Manbalar
- NudeNet — https://github.com/notAI-tech/NudeNet (MIT, v3.4.2)
- image-safety-classifier-s — https://huggingface.co/OwenElliott/image-safety-classifier-s (MIT)
- Marqo nsfw-image-detection-384 — https://huggingface.co/Marqo/nsfw-image-detection-384
- AdamCodd/vit-base-nsfw-detector — https://huggingface.co/AdamCodd/vit-base-nsfw-detector
- Falconsai/nsfw_image_detection — https://huggingface.co/Falconsai/nsfw_image_detection
