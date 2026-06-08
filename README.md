# 🧬 AgeForge — Conditional GAN Face Aging

Progressive face aging and de-aging powered by a conditional GAN with age-conditioned ResBlocks.

## Architecture

```
Input (3×256×256)
    │
    ▼
┌─────────────────────────────┐
│  ENCODER (3 Conv layers)    │
│  ngf=32 → 64 → 128 channels │
└─────────────┬───────────────┘
              │
    ┌─────────▼──────────┐
    │  AGE CLASS EMBED   │  ← target age (0–9)
    └─────────┬──────────┘
              │ injected at each ResBlock
              ▼
┌─────────────────────────────┐
│  6× AgeConditionedResBlock  │
│  • Conv → InstanceNorm      │
│  • AdaIN via age embeddings │
│  • Residual skip connection │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  DECODER (2 ConvTranspose)  │
│  128 → 64 → 32 → 3 channels │
│  Tanh activation            │
└─────────────────────────────┘
```

## Age Classes

| Class | Age Range | Label        |
|-------|-----------|--------------|
| 0     | 0–10      | Child        |
| 1     | 11–20     | Teen         |
| 2     | 21–30     | Young Adult  |
| 3     | 31–40     | Adult        |
| 4     | 41–50     | Mid Adult    |
| 5     | 51–60     | Mature       |
| 6     | 61–70     | Senior       |
| 7     | 71–80     | Elder        |
| 8     | 81–90     | Late Elder   |
| 9     | 91–100    | Centenarian  |

## Features

- ✅ Conditional GAN architecture (Encoder → 6×ResBlocks → Decoder)
- ✅ Age-conditioned Adaptive Instance Normalization (AdaIN)
- ✅ Progressive aging/de-aging texture engine
- ✅ Forensic visualization panel (delta map + age classifier output)
- ✅ Download result as PNG
- ✅ Deployable to Streamlit Community Cloud

## Local Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push to a GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set `app.py` as main file
4. Deploy — no GPU needed (CPU inference)

## Notes

- This demo uses the cGAN **architecture** with random weights.  
- To use trained weights: load a `.pth` checkpoint in `load_model()`.
- For production training, use CACD, UTKFace, or FFHQ datasets with adversarial + cycle-consistency losses.
- **For research/educational use only.**
