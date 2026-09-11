# Modelni o'z ma'lumotingizda yaxshilash

Servisdagi ikkala model ham **tayyor (pre-trained)**. Bu katalog ularni
sizning rasmlaringizga moslashtirish uchun. **PyTorch kerak emas** — hammasi
mavjud `onnxruntime` + `numpy` bilan ishlaydi.

## Nega kerak

Klassifikator umumiy internetda o'qitilgan va **ochiq teri maydoniga** qarab
qaror qiladi — kontekstga ham, yoshga ham emas. O'lchangan misol:

| Rasm | Asl model | To'g'risi |
|------|-----------|-----------|
| Ko'kragi ochiq bolakay, qo'lida baliq | `nsfw 78.30%` | `safe` |
| Plyajda ko'kragi ochiq yigit | `sfw 88.01%` + `BELLY_EXPOSED 77%` → `suggestive` | munozarali |
| Ko'zoynakli bola portreti | `sfw 93.08%` → `safe` | ✅ |

edugames.uz uchun bu eng yomon nuqta: trafikda bolalar suratlari ko'p
(sport, suzish, jismoniy tarbiya) va aynan shular yolg'on bloklanadi.

## Uch qadam

```bash
cd /www/wwwroot/api.qobilbek.dev/nsfw-service

# 1. Rasmlarni raqamli to'plamga aylantirish (bir marta, sekin)
.venv/bin/python training/embed.py --data-dir data --out dataset.npz

# 2. Daraja 0 — chegaralarni o'lchash (o'qitishsiz, sekundlar)
.venv/bin/python training/calibrate.py --data dataset.npz

# 3. Daraja 1 — o'z boshingizni o'qitish (sekundlar)
.venv/bin/python training/train_head.py --data dataset.npz
```

**Avval 2-qadamni bajaring.** Ko'pincha eng katta yutuq o'sha yerda va
hech narsa o'qitilmaydi. 3-qadam faqat 2-qadam yetmasa kerak — `calibrate.py`
buni o'zi aytadi.

## Ma'lumot tayyorlash

Yorliq = papka nomi:

```
data/
├── safe/          oddiy rasmlar (bolalar, sport, tabiat, hujjat...)
├── suggestive/    shahvoniy ishora bor, lekin 18+ emas
├── nsfw/          18+
└── nsfl/          qon, zo'ravonlik, jirkanch
```

CDN'dagi rasmlar uchun `url,label` ustunli CSV berish mumkin:

```csv
https://cdn.edugames.uz/f/xxx.webp,safe
https://cdn.edugames.uz/f/yyy.webp,suggestive
```

```bash
.venv/bin/python training/embed.py --urls-csv list.csv --data-dir data
```

### Nechta rasm kerak

| Maqsad | Minimum | Yaxshi |
|--------|---------|--------|
| `calibrate.py` (Daraja 0) | 500 jami | 1000+ |
| `train_head.py` (Daraja 1) | sinfiga 50 | sinfiga 300+ |

Muhimi — **sonidan ko'ra taqsimot**. 500 ta o'z trafikingizdan olingan rasm
5000 ta internetdan yuklangan rasmdan foydaliroq. Ayniqsa quyidagilar kerak:

- ko'kragi ochiq bolalar (suzish, sport) → `safe`
- yaqindan olingan teri kadrlari (yuz, qo'l) → `safe`
- sport gimnastikasi, suzish kostyumi → `safe` yoki `suggestive`
- haqiqiy `nsfw` misollar — busiz recall o'lchab bo'lmaydi

### Rasm saqlash haqida

Servis ishlab chiqarishda **hech qanday rasmni saqlamaydi**. Bu qoida
faqat ishlab chiqarish quvuriga tegishli; o'qitish to'plami esa diskda
yashaydi. `data/` va `*.npz` `.gitignore` da.

Chegaraviy holatlarni yig'ish uchun API javobidagi **`needs_review: true`**
bayrog'idan foydalaning — aynan shu rasmlar belgilashga eng arzigulik.

## `train_head.py` qanday ishlaydi

SwiftFormer-S grafigi ichida `mean_1` nomli **(1, 224)** tenzor bor — bu
backbone'ning rasm haqidagi butun "fikri", sinflarga bo'linishdan oldingi
holati. Ikkala klassifikatsiya boshi ham shundan oziqlanadi.

Biz uni chiqishga chiqaramiz, backbone'ni **qotirib qo'yamiz** va faqat
oxirgi chiziqli qatlamni sizning ma'lumotingizda qayta o'qitamiz.

| | O'lchangan |
|---|---|
| 1 rasmdan embedding | ~13 ms |
| 10 000 rasm | ~2.2 daqiqa |
| 10 000 embedding diskda | 8.5 MB |
| Boshni o'qitish (10 000×224) | ~8 s |

Embeddinglar **bir marta** hisoblanadi. Shundan keyin gipeparametrni
o'zgartirib qayta o'qitish sekundlar oladi.

### `--suggestive-as`

Klassifikatorda 3 ta sinf bor (`nsfl`, `nsfw`, `sfw`), sizda esa 4 ta yorliq.
`suggestive` klassifikator sinfi emas — u detektor topilmalaridan hosil
bo'ladi. Shuning uchun:

- `--suggestive-as sfw` (default) — klassifikator "ochiq teri = nsfw" deb
  o'rganmaydi. Aynan shu xato bolakayga 78% bergan edi.
- `--suggestive-as nsfw` — qattiqroq. Yolg'on bloklash ko'payadi.

## Ulash

```bash
# .env ga
CUSTOM_HEAD_FILE=custom_head.npz
systemctl restart nsfw-api

# tekshirish
curl -s -H "X-API-Key: $KEY" https://api.qobilbek.dev/nsfw/v1/models \
  | grep -o '"head":"[a-z]*"'      # -> "head":"custom"
```

Backbone o'zgarmaydi, `scoring.py` va `.env` chegaralari ham o'z kuchida
qoladi — faqat oxirgi qatlam almashadi.

**Nosozlikda servis tushmaydi:** bosh yuklanmasa (fayl yo'q, shakl noto'g'ri,
sinflar tartibi boshqa) log'ga xato yoziladi va **asl bosh** ishlatiladi.
`/v1/models` javobidagi `head` maydoni qaysi bosh ishlayotganini ko'rsatadi.

Orqaga qaytarish — `.env` dan `CUSTOM_HEAD_FILE` ni olib tashlash va restart.

## Daraja 2 — to'liq fine-tune (bu yerda yo'q)

Backbone'ni ham qayta o'qitish uchun `torch` + `timm` kerak (~2.5 GB) va
GPU'siz 20k rasm 8–20 soat oladi. Undan oldin Daraja 0 va 1 ni o'tkazing.

⚠️ Agar fine-tune qilsangiz: asl ONNX'da normalizatsiya (`Div 255` →
`Sub mean` → `Div std`) va `Softmax` **graf ichiga pishirilgan**. Oddiy
eksportda ular yo'qoladi va servis xato bermasdan noto'g'ri ishlay boshlaydi.
Eksportda o'sha wrapper'ni qayta qo'shish shart.
