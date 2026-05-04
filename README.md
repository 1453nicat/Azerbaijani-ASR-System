# Azerbaijani Speech-to-Text (ASR) with Whisper

&gt; **Internship Project** — Fine-tuning OpenAI Whisper for Low-Resource Azerbaijani ASR  
&gt; **Dataset:** Mozilla Common Voice 22.0 (Azerbaijani) | **Model:** `openai/whisper-small`

---

## Overview

This project investigates the effectiveness of transfer learning for Azerbaijani automatic speech recognition (ASR), a low-resource agglutinative language. We establish a baseline using the pre-trained `whisper-small` model in zero-shot mode, then fine-tune the decoder on a limited subset of Common Voice data to evaluate whether small-scale adaptation can reduce error rates despite severe data scarcity.

Azerbaijani presents unique challenges for ASR: agglutinative morphology generates a vast vocabulary of inflected forms, phonetically similar consonants (e.g., *q* / *ğ*, *x* / *h*) are easily confused, and publicly available transcribed audio remains critically limited [^4^].

---

## Repository Structure

