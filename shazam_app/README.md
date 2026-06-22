# Sonic Signatures — EE200 Q3B
**Course Project | IIT Kanpur | Dr. Tushar Sandhan**

A Shazam-style audio fingerprinting app built with Streamlit.

## How it works
1. Songs are pre-indexed into a fingerprint database using **STFT constellation maps** and **paired-hash fingerprints** `(f1, f2, Δt)`.
2. A query clip is fingerprinted the same way, and matched against the database via **offset-histogram voting**.
3. The song whose offset histogram has the sharpest peak wins.

## Two modes
- **Single-clip mode** — upload one `.wav`, see spectrogram + constellation + offset histograms + result
- **Batch mode** — upload multiple `.wav` files, download `results.csv` with `filename, prediction`

## Song database (8 songs)
- A Day In The Life
- A Hard Day's Night
- Back In The U.S.S.R.
- Blackbird
- Can't Buy Me Love
- Don't Stop Me Now
- Eleanor Rigby
- I'll Follow The Sun

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud
1. Push this folder to a GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set main file to `app.py` → Deploy
