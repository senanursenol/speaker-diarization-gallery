# Speaker Diarization Gallery (Streamlit)

Bu repo, Pyannote tabanlı akışın öncelikli olduğu bir konuşmacı diyari̇zasyon demosu ve yardımcı scriptleri içerir. 
Ağır veri ve ağırlık dosyaları repoya dahil edilmez; yalnızca kod ve küçük örnekler yer alır.

## Özellikler
- Pyannote pipeline ile diyari̇zasyon
- (Opsiyonel) SpeechBrain ve Resemblyzer denemeleri `scripts/legacy/` altında
- Streamlit arayüzü (özel CSS)

## Kurulum
```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit/app.py