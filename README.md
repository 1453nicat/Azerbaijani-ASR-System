# Azərbaycan dili üçün ASR sistemi
---

## Layihə Haqqında

Bu layihə aşağı resurslu, aqqlütinativ dil olan Azərbaycan dilində avtomatik nitq tanıma (ASR) sisteminin effektivliyini öyrənir. Əvvəlcə öncədən öyrədilmiş `whisper-small` modelinin zero-shot rejimində bazasını müəyyən edirik, sonra "common voice" datasetinin məhdud alt dəstində dekoderi fine-tuning edərək, data qıtlığına baxmayaraq, kiçik miqyaslı adaptasiyanın səhv dərəcələrini azalda biləcəyini qiymətləndiririk. Bununla yanaşı, Azərbaycan dili ASR üçün вэ unikal çətinliklər yaradır, məsələn, aqqlütinativ morfologiya çoxlu sayda şəkilçili söz formaları yaradır, fonetik olaraq oxşar samitlər asanlıqla qarışdırılır və açıq mənbəli audio data məhduddur.

---

## Repozitoriya Strukturu

```
az-stt-intern/
├── README.md                 # Bu fayl
├── requirements.txt          # Python
├── part_a/                   # Hissə A: Base İnferens və qiymətləndirmə
│   └── part_a.ipynb
├── part_b/                   # Hissə B: Fine-Tuning pipeline
│   └── part_b.ipynb
├── results/                  # WER/CER cədvəlləri, təlim əyriləri, müqayisə qrafikləri
│   ├── wer_cer_comparison.png
│   ├── training_progress.png
│   ├── wer_distribution.png
└── report.pdf                # Hissə C: Analitik hesabat (Azərbaycan dilində)
```

---

## Model və Hiperparametrlər

| Parametr | Dəyər |
|----------|-------|
| **Bazal Model** | `openai/whisper-small` (244M parametr) |
| **Tapşırıq** | Şərtli Generasiya (Transkripsiya) |
| **Dil** | Azərbaycan dili (`az`) |
| **Diskretləşdirmə Tezliyi** | 16 kHz |
| **Fine-Tuning Strategiyası** | Yalnız dekoder (encoder dondurulub) |
| **Təlim Nümunələri** | 157 |
| **Validasiya Nümunələri** | 50 |
| **Test Nümunələri** | 100 |
| **Epoch Sayı** | 5 |
| **Batch Ölçüsü** | 8 per device (effektiv: 16 gradient toplama ilə) |
| **Öyrənmə Sürəti** | 1×10⁻⁵ |
| **Isınma Addımları** | 20 |
| **Dəqiqlik** | FP16 |
| **Gradient Checkpointing** | Aktiv |
| **Hardware** | NVIDIA Tesla T4 (15 GB VRAM) |

---

## Nəticələr

### Kəmiyyat Müqayisəsi

| Model | WER (%) | CER (%) | Δ WER | Δ CER |
|-------|---------|---------|-------|-------|
| **Bazal** (zero-shot) | 126,41 | 59,03 | — | — |
| **Fine-Tuned** (yalnız dekoder, 5 epoch) | 100,78 | 138,44 | **+25,62** | −79,41 |

*Aşağı dəyər daha yaxşıdır. Δ bazala nisbətən dəyişikliyi göstərir (müsbət = yaxşılaşma).*

### Əsas Nəticələr

- **Bazal Performans:** Zero-shot model Azərbaycan dilində fəlakətli şəkildə uğursuz olur, xarici dillərdə (ərəb, koreya, ingilis) hallüsinasiyalar və mənasız simvol ardıcıllıqları yaradır. WER 100 %-dən yuxarıdır, çünki model referensdəkindən daha çox söz generasiya edir.
- **Fine-Tuning Təsiri:** Yalnız 157 nümunədə dekoder fine-tuningi WER-i **25,62 faiz punkt** azaldır. Bu göstərir ki, hətta minimal həcmdə dilə xas data hallüsinasiyaları azalda və söz səviyyəsində tanımanı yaxşılaşdıra bilər.
- **CER Regressiyası:** Simvol Səhv Dərəcəsi (CER) 59,03 %-dən 138,44 %-ə yüksəlir. Bu gözlənilməz nəticə fine-tuned modelin daha qısa, "azərbaycan dilinə daha oxşar" çıxışlar yaratdığını, lakin simvol səviyyəsində tez-tez səhv əvəzetmələr etdiyini göstərir — bu, məhdud data üzərində həddindən artıq uyğunlaşma (overfitting) və aqqlütinativ şəkilçilərlə mübarizənin nəticəsidir.
- **Validasiya Dinamikası:** Ən yaxşı validasiya WER-i (~70,49 %) **2-ci epoch**-da əldə olunub; sonrakı epoch-larda overfitting əlamətləri müşahidə olunur. Bu, `load_best_model_at_end=True` parametri ilə modelin yalnız ən yaxşı epoch-un yadda saxlanmasını zəruri edir.

### Səhv Analizi Xülasəsi

| Səhv Tipi | Müşahidə |
|-----------|----------|
| **Hallüsinasiyalar** | Bazal model əlaqəsiz dillərdə bütün ifadələr generasiya edir; fine-tuned model azaldır, lakin tamamilə aradan qaldırmır. |
| **Fonetik Qarışıqlıq** | Oxşar samitlər (*q* ↔ *ğ*, *x* ↔ *h*, *k* ↔ *c*) tez-tez yerinə yetirilir. |
| **Morfoloji** | Aqqlütinativ formalar (məsələn, *kitablarımdadır*) nadir hallarda düzgün transkripsiya olunur; model kök formalarına meyllənir. |
| **OOV Sözlər** | Nadir və ya sahə-spesifik terminlər yüksək tezlikli yaxınlıqlarla əvəz olunur. |

---

## Quraşdırma və Yenidən İstehsal

### 1. Mühit

```bash
# Repozitoriyanı klonlayın
git clone https://github.com/yourusername/az-stt-intern.git
cd az-stt-intern

# Virtual mühit yaradın (tövsiyə olunur)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Asılılıqları quraşdırın
pip install -r requirements.txt
```

### 2. Asılılıqlar (`requirements.txt`)

```
datasets==2.18.0
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
| **A** | `part_a/baseline_asr.ipynb` | Common Voice `az` test hissəsini yükləyin, zero-shot inferens aparın, WER/CER hesablayın, səhv paylanmalarını vizuallaşdırın. |
| **B** | `part_b/finetune_whisper.ipynb` | Təlim/validasiya hissələrini hazırlayın, xüsusiyyət çıxarın və tokenləşdirin, dekoderi fine-tune edin, bazal model ilə müqayisə edin. |
| **C** | `report.pdf` | Texniki çətinliklər, dilçi çətinliklər və yaxşılaşdırma yol xəritəsini əhatə edən tam analitik hesabat (Azərbaycan dilində). |

---

## Müzakirə və Məhdudiyyətlər

- **Data Qıtlığı:** Common Voice 22.0-da təsdiqlənmiş Azərbaycan dili audio-su cəmi ~10 saat təşkil edir. Model bu həcmdə robust akustik və ya dilçi təmsil öyrənə bilmir. Bu əsas dar boğazdır.
- **Encoder Dondurulması:** Encoder-i dondurmaq VRAM istifadəsini azaltdı, lakin modelin akustik xüsusiyyətlərini Azərbaycan fonetikasına uyğunlaşdırmasının qarşısını aldı. Daha böyük datasetlərdə tam model fine-tuningi daha yaxşı nəticələr verə bilər.
- **Aqqlütinativ Mürəkkəblik:** Azərbaycan dilinin zəngin morfoloji paradigmı təlim zamanı görülən effektiv lüğəti mümkün söz formalarının çox kiçik bir hissəsini əhatə edir, bu da yüksək OOV dərəcələrinə səbəb olur.

---

## Gələcək İşlər

1. **Dataseti Genişləndirmək:** Azərbaycan dilinə xas, müxtəlif sahə, vurğu və səs-küy şəraitlərini əhatə edən əlavə audio data toplamaq və ya annotasiya etmək (hədəf: 50+ saat).
2. **Tam Model Fine-Tuningi:** Encoder-i açmaq və genişləndirilmiş datasetdə daha böyük variant (`whisper-medium` və ya `whisper-large`) üzərində tam fine-tuning aparmaq.
3. **Qabaqcıl Texnikalar:** Parametr-səmərəli adaptasiya üçün LoRA/PEFT, data artırımı üçün SpecAugment və yaxşılaşdırılmış generasiya keyfiyyəti üçün beam-search dekodlaşdırma tətbiq etmək.

---

## İstinadlar

- Radford, A., et al. (2022). *Robust Speech Recognition via Large-Scale Weak Supervision.* OpenAI.
- Mussakhojayeva, S., et al. (2023). *Multilingual Speech Recognition for Turkic Languages.* Information, 14(2), 74.
- Common Voice Dataset: [Mozilla Common Voice](https://commonvoice.mozilla.org/)

---

**Müəllif:** [Adınız Soyadınız]  
**Rəhbər:** [Rəhbərin Adı Soyadı]  
**Tarix:** May 2026
```

---

Bu versiyanı birbaşa `README.md` kimi istifadə edə bilərsiniz. Əgər istəsəniz, `requirements.txt` faylını da yaradıb sizə göndərə bilərəm.
