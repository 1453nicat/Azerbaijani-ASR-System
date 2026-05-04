# Azərbaycan dili üçün ASR sistemi
---

## Layihə Haqqında

Bu layihə aşağı resurslu, aqqlütinativ dil olan Azərbaycan dilində avtomatik nitq tanıma (ASR) sisteminin effektivliyini öyrənir. Əvvəlcə öncədən öyrədilmiş `whisper-small` modelinin zero-shot rejimində bazasını müəyyən edirik, sonra "common voice" datasetinin məhdud alt dəstində dekoderi fine-tuning edərək, data qıtlığına baxmayaraq, kiçik miqyaslı adaptasiyanın səhv dərəcələrini azalda biləcəyini qiymətləndiririk. Bununla yanaşı, Azərbaycan dili ASR üçün вэ unikal çətinliklər yaradır, məsələn, aqqlütinativ morfologiya çoxlu sayda şəkilçili söz formaları yaradır, fonetik olaraq oxşar samitlər asanlıqla qarışdırılır və açıq mənbəli audio data məhduddur.

---

## Repozitoriya Strukturu

az-stt-intern/
├── README.md                 # Bu fayl
├── requirements.txt          # Python
├── part_a/                   # Hissə A: Base inferens və qiymətləndirmə
│   └── baseline_asr.ipynb
├── part_b/                   # Hissə B: Fine-Tuning pipeline
│   └── finetune_whisper.ipynb
├── results/                  # WER/CER cədvəlləri, təlim əyriləri, müqayisə qrafikləri
│   ├── wer_cer_comparison.png
│   ├── training_progress.png
│   ├── wer_distribution.png
│   └── metrics.json
└── report.pdf                # Hissə C: Analitik hesabat (Azərbaycan dilində)

