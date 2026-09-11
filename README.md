# NSFW Detection API

18+ (NSFW/NSFL) rasm aniqlovchi servis. FastAPI + ONNX Runtime, GPU talab
qilmaydi, rasmlar diskka yozilmaydi.

**Jonli:** https://api.qobilbek.dev/nsfw/ · **Docs:** https://api.qobilbek.dev/nsfw/docs

| | |
|---|---|
| Katalog | `/www/wwwroot/api.qobilbek.dev/nsfw-service` |
| Servis | `nsfw-api.service` (systemd), `127.0.0.1:8720`, 3 worker |
| Foydalanuvchi | `www:www` |
| Redis | `redis://127.0.0.1:6379/4` |
| Loglar | `journalctl -u nsfw-api -f` |

## Modellar

| Model | Vazifa | Hajm | Litsenziya | CPU |
|-------|--------|------|-----------|-----|
| [OwenElliott/image-safety-classifier-s](https://huggingface.co/OwenElliott/image-safety-classifier-s) | Klassifikator: NSFW / NSFL / SFW | 23 MB | MIT | ~7 ms |
| [NudeNet 3.4.2](https://github.com/notAI-tech/NudeNet) (320n) | 18 ta tana qismi + bbox | 12 MB | MIT | ~10 ms |

Nima uchun aynan shular tanlangani — `docs/TASKS.md` §1.

## Hujjatlar

- **`docs/TASKS.md`** — to'liq task hujjati: model tadqiqi, arxitektura,
  API spetsifikatsiyasi, xavfsizlik talablari, bajarilish checklisti.
- **`docs/API.md`** — mijozlar uchun hujjat: endpointlar, javob formati,
  xato kodlari, namuna kodlar (Python / PHP / Node.js).
- **`training/README.md`** — modelni o'z ma'lumotingizda yaxshilash
  (chegaralarni kalibrlash va o'z boshingizni o'qitish; PyTorch kerak emas).

## Aniqlik haqida ochiq gap

Chegaralar (`THRESHOLD_*`) **o'lchanmagan** — bir nechta rasmga qarab qo'lda
tanlangan. Ma'lum xato: ko'kragi ochiq bolakay suratiga klassifikator
`nsfw 78%` beradi. Oraliq zona qoidasi uni bloklashdan saqlaydi
(`suggestive` qaytariladi), lekin zaxira atigi ~16 ball.

Javobdagi **`needs_review`** shunday chegaraviy holatlarni belgilaydi —
`is_safe` ni o'zgartirmaydi, faqat moderator navbatiga signal beradi.
Aniqlikni oshirish yo'li — `training/README.md`.

**Thumbnail yubormang.** O'lchangan: 1600×1157 rasmda topilgan
`BUTTOCKS_EXPOSED 37%` va `ARMPITS_EXPOSED 47%` o'sha rasmning 256×185
thumbnail'ida umuman topilmagan (NudeNet kirishi 320×320). Rasmning kichik
tomoni 320px dan past bo'lsa, javobning `reasons` ro'yxatida
`diqqat: rasm kichik` qatori paydo bo'ladi. Batafsil — `docs/API.md` §4.4.1.

## Tez boshlash

```bash
# Kalit yaratish
sudo -u www .venv/bin/python scripts/apikey.py create --name "loyiha" --rpm 120
sudo -u www .venv/bin/python scripts/apikey.py list
sudo -u www .venv/bin/python scripts/apikey.py revoke --id k_...

# Sinov
curl -X POST https://api.qobilbek.dev/nsfw/v1/analyze \
  -H "X-API-Key: nsfw_..." -H "Content-Type: application/json" \
  -d '{"url":"https://example.com/a.jpg"}'
```

## Ishlab chiqish

```bash
.venv/bin/python -m pytest              # 73 ta test
.venv/bin/uvicorn app.main:app --port 8799 --reload
systemctl restart nsfw-api              # kod o'zgargandan keyin
```

Testlar haqiqiy modellar bilan ishlaydi (mock yo'q) — asosiy xavf aynan
preprocessing va kanal tartibi kabi joylarda.

## Sozlash

Barcha sozlamalar `.env` da (`chmod 600`). Namuna — `.env.example`.
Eng muhimlari:

| O'zgaruvchi | Default | Tavsif |
|-------------|---------|--------|
| `PUBLIC_MODE` | `false` | `true` bo'lsa API kalitisiz ishlaydi |
| `ALLOWED_PATH_ROOTS` | bo'sh | `path` rejimi uchun oq ro'yxat; bo'sh = rejim o'chiq |
| `THRESHOLD_NSFW` | `60` | Oraliq zonaning quyi chegarasi (%) |
| `THRESHOLD_NSFW_CONFIDENT` | `90` | Bundan yuqorida detektor tasdig'i shart emas |
| `CUSTOM_HEAD_FILE` | bo'sh | O'z o'qitilgan boshingiz (`models/` ichida) |
| `RATE_LIMIT_PER_MINUTE` | `60` | Kalit uchun default limit |

`.env` o'zgargandan keyin `systemctl restart nsfw-api` kerak.

## Xavfsizlik

- **SSRF:** DNS o'zimiz hal qilinadi, har bir IP tekshiriladi, so'rov
  tekshirilgan IP ga yuboriladi (`Host` + SNI asl domen bilan) —
  DNS rebinding ishlamaydi. Redirect'lar qo'lda, har biri qayta tekshiriladi.
- **Path traversal:** `resolve()` + oq ro'yxat; symlink tashqariga chiqsa rad etiladi.
- **Rasm:** 20 MB / 50 MP chegarasi, magic-bytes tekshiruvi, EXIF tozalanadi.
- **Auth:** fail-closed (Redis yo'q bo'lsa ham kirishga ruxsat berilmaydi).
  Kesh va rate-limit esa fail-open.
