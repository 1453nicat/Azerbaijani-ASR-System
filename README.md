# Azərbaycan dili üçün ASR sistemi
---

## Layihə Haqqında

Bu layihə aşağı resurslu, aqqlütinativ dil olan Azərbaycan dilində avtomatik nitq tanıma (ASR) sisteminin effektivliyini öyrənir. Əvvəlcə öncədən öyrədilmiş `whisper-small` modelinin zero-shot rejimində bazasını müəyyən edirik, sonra "common voice" datasetinin məhdud alt dəstində dekoderi fine-tuning edərək, data qıtlığına baxmayaraq, kiçik miqyaslı adaptasiyanın səhv dərəcələrini azalda biləcəyini qiymətləndiririk. Bununla yanaşı, Azərbaycan dili ASR üçün вэ unikal çətinliklər yaradır, məsələn, aqqlütinativ morfologiya çoxlu sayda şəkilçili söz formaları yaradır, fonetik olaraq oxşar samitlər asanlıqla qarışdırılır və açıq mənbəli audio data məhduddur.

---

## Repozitoriya Strukturu

```
az-stt-intern/
├── README.md                 # Bu fayl
├── asr.jpynb                 # Tam proyekt kodu: .jpynb formatında
├── asr.py                    # Tam proyekt kodu: .py formatında
├── requirements.txt          # Python
├── part_a/part_a.jpynb       # Hissə A: Base İnferens və qiymətləndirmə
├── part_b/part_b.jpynb       # Hissə B: Fine-Tuning pipeline
├── results/                  # WER/CER cədvəlləri, təlim əyriləri, müqayisə qrafikləri
│   ├── wer_cer_comparison.png
│   ├── training_progress.png
│   ├── wer_cer_last.png
└── report.pdf                # Hissə C: Analitik hesabat (Azərbaycan dilində)
```

---

## Model və Hiperparametrlər

| Parametr | Dəyər |
|----------|-------|
| **Base Model** | `openai/whisper-small`|
| **Tapşırıq** | Transkripsiya |
| **Dil** | Azərbaycan dili (`az`) |
| **Diskretləşdirmə tezliyi** | 16 kHz |
| **Fine-Tuning strategiyası** | Yalnız dekoder (encoder dondurulub) |
| **Təlim nümunələri** | 157 |
| **Validasiya nümunələri** | 50 |
| **Test nümunələri** | 100 |
| **Epoch sayı** | 5 |
| **Batch ölçüsü** | 8 per device (effektiv: 16 gradient toplama ilə) |
| **Öyrənmə sürəti** | 1×10⁻⁵ |
| **İsinmə addımları (warmup steps)** | 20 |
| **Dəqiqlik** | FP16 |
| **Gradient checkpointing** | Aktiv |
| **Hardware** | NVIDIA Tesla T4 (15 GB VRAM) |

---

## Nəticələr

### Kəmiyyat Müqayisəsi

| Model | WER (%) | CER (%) | Δ WER | Δ CER |
|-------|---------|---------|-------|-------|
| **Base** | 126,41 | 59,03 | — | — |
| **Fine-Tuned** (5 epoch) | 100,78 | 138,44 | **+25,62** | −79,41 |

*Aşağı dəyər daha yaxşıdır. Δ bazala nisbətən dəyişikliyi göstərir (müsbət = yaxşılaşma).*

### Əsas Nəticələr

- **Base performans:** Zero-shot model Azərbaycan dilində uğursuz olur, xarici dillərdə (ərəb, koreya, ingilis) hallüsinasiyalar və mənasız simvol ardıcıllıqları yaradır. WER 100%-dən yuxarıdır, çünki model referensdəkindən daha çox söz generasiya edir.
- **Fine-Tuning təsiri:** Yalnız 157 nümunədə dekoder fine-tuningi WER-i **25,62 faiz punkt** azaldır. Bu göstərir ki, hətta minimal həcmdə dilə xas data hallüsinasiyaları azalda və söz səviyyəsində tanımanı yaxşılaşdıra bilər.
- **CER regressiyası:** Simvol Səhv Dərəcəsi (CER) 59,03 %-dən 138,44 %-ə yüksəlir. Bu gözlənilməz nəticə fine-tuned modelin daha qısa, "azərbaycan dilinə daha oxşar" çıxışlar yaratdığını, lakin simvol səviyyəsində tez-tez səhv əvəzetmələr etdiyini göstərirş Bu, məhdud data üzərində həddindən artıq uyğunlaşma (overfitting) və aqqlütinativ şəkilçilərə görə ola bilər.
- **Validasiya dinamikası:** Ən yaxşı validasiya WER-i (~70,49 %) **2-ci epoch**-da əldə olunub, sonrakı epoch-larda overfitting əlamətləri müşahidə olunur. Bu, `load_best_model_at_end = True` parametri ilə modelin yalnız ən yaxşı epoch-un yadda saxlanmasını vacib qılır.

### Səhv Analizi Xülasəsi

| Səhv Tipi | Müşahidə |
|-----------|----------|
| **Hallüsinasiyalar** | Base model əlaqəsiz dillərdə ifadələr generasiya edir, fine-tuned model bunu azaldır, lakin tamamilə aradan qaldırmır. |
| **Fonetik qarışıqlıq** | Oxşar samitlər (*q* ↔ *ğ*, *x* ↔ *h*, *k* ↔ *c*) tez-tez yerinə yetirilir. |
| **Morfoloji** | Aqqlütinativ formalar (məsələn, *kitablarımdadır*) nadir hallarda düzgün transkripsiya olunur; model kök formalarına meyllənir. |

---

## Setup + run

### 1. Environment

```bash
git clone https://github.com/yourusername/az-stt-intern.git
cd az-stt-intern

# Virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Asılılıqları quraşdırın
pip install -r requirements.txt
```

### 2. Asılılıqlar

```
datasets==3.3.2
transformers
accelerate
jiwer
soundfile
librosa
evaluate
torch
torchaudio
numpy
matplotlib
```

> **Qeyd:** Layihə Google Colab-da CUDA 11.8 ilə hazırlanıb. GPU təlimi üçün PyTorch-u uyğun CUDA indeks URL-i ilə quraşdırın:
> ```bash
> pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
> ```

### 3. Pipeline-ı İşə Salmaq

| Addım | Notebook | Təsvir |
|-------|----------|--------|
| **A** | `part_a/part_a.ipynb` | Common Voice `az` test hissəsini yükləyin, zero-shot inferens aparın, WER/CER hesablayın, səhv paylanmalarını vizuallaşdırın. |
| **B** | `part_b/part_b.ipynb` | Təlim/validasiya hissələrini hazırlayın, xüsusiyyət çıxarın və tokenləşdirin, dekoderi fine-tune edin, bazal model ilə müqayisə edin. |
| **C** | `report.pdf` | Texniki çətinliklər, dilçi çətinliklər və yaxşılaşdırma yol xəritəsini əhatə edən tam analitik hesabat (Azərbaycan dilində). |

---

## Məhdudiyyətlər

- **Data Qıtlığı:** Common Voice 22.0-da təsdiqlənmiş Azərbaycan dili audio-su cəmi ~10 saat təşkil edir. Model bu həcmdə robust akustik və ya dilçi təmsil öyrənə bilmir.
- **Encoder Dondurulması:** Encoder-i dondurmaq VRAM istifadəsini azaltdı, lakin modelin akustik xüsusiyyətlərini Azərbaycan fonetikasına uyğunlaşdırmasının qarşısını aldı. Daha böyük datasetlərdə tam model fine-tuningi daha yaxşı nəticələr verə bilər.

---
