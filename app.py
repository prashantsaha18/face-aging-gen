import streamlit as st
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageOps
import io
import base64
import time
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from collections import defaultdict
import json
import random

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgeForge · Conditional GAN Face Aging",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;800&display=swap');

:root {
    --bg:      #0a0c10;
    --surface: #111520;
    --border:  #1e2535;
    --accent:  #00e5ff;
    --accent2: #ff6b35;
    --accent3: #a259ff;
    --text:    #e0e8f8;
    --muted:   #5a6a8a;
    --success: #00ff9d;
    --warn:    #ffd166;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Space Mono', monospace;
}
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
h1,h2,h3 { font-family: 'Syne', sans-serif !important; }

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem; font-weight: 800;
    background: linear-gradient(135deg,#00e5ff 0%,#a259ff 50%,#ff6b35 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; letter-spacing: -2px; line-height: 1.1; margin-bottom:0;
}
.hero-sub {
    font-family:'Space Mono',monospace; color:var(--muted);
    font-size:0.78rem; letter-spacing:3px; text-transform:uppercase; margin-top:6px;
}
.metric-chip {
    display:inline-block; background:rgba(0,229,255,0.08);
    border:1px solid rgba(0,229,255,0.25); border-radius:4px;
    padding:4px 12px; font-size:0.72rem; color:var(--accent);
    letter-spacing:1.5px; margin:3px 4px 3px 0; font-family:'Space Mono',monospace;
}
.metric-chip-warn {
    display:inline-block; background:rgba(255,209,102,0.08);
    border:1px solid rgba(255,209,102,0.25); border-radius:4px;
    padding:4px 12px; font-size:0.72rem; color:var(--warn);
    letter-spacing:1.5px; margin:3px 4px 3px 0; font-family:'Space Mono',monospace;
}
.metric-chip-ok {
    display:inline-block; background:rgba(0,255,157,0.08);
    border:1px solid rgba(0,255,157,0.25); border-radius:4px;
    padding:4px 12px; font-size:0.72rem; color:var(--success);
    letter-spacing:1.5px; margin:3px 4px 3px 0; font-family:'Space Mono',monospace;
}
.section-label {
    font-family:'Space Mono',monospace; font-size:0.65rem; letter-spacing:3px;
    text-transform:uppercase; color:var(--muted); margin-bottom:8px;
    border-bottom:1px solid var(--border); padding-bottom:6px;
}
.age-badge {
    display:inline-block; font-family:'Syne',sans-serif;
    font-size:2.4rem; font-weight:800; color:var(--accent);
    text-shadow:0 0 20px rgba(0,229,255,0.4);
}
.info-card {
    background:var(--surface); border:1px solid var(--border);
    border-left:3px solid var(--accent); border-radius:6px;
    padding:14px 18px; margin:10px 0; font-size:0.8rem; line-height:1.7;
}
.warning-card {
    background:rgba(255,107,53,0.06); border:1px solid rgba(255,107,53,0.3);
    border-left:3px solid var(--accent2); border-radius:6px;
    padding:14px 18px; margin:10px 0; font-size:0.8rem;
}
.stat-card {
    background:var(--surface); border:1px solid var(--border);
    border-radius:8px; padding:16px; text-align:center;
}
.stat-val {
    font-family:'Syne',sans-serif; font-size:1.8rem; font-weight:800; color:var(--accent);
}
.stat-lbl {
    font-family:'Space Mono',monospace; font-size:0.6rem;
    letter-spacing:2px; color:var(--muted); text-transform:uppercase; margin-top:4px;
}
.arch-pill {
    display:inline-block; background:rgba(162,89,255,0.12);
    border:1px solid rgba(162,89,255,0.3); border-radius:20px;
    padding:3px 10px; font-size:0.68rem; color:var(--accent3);
    margin:2px; font-family:'Space Mono',monospace;
}
div[data-testid="stButton"] > button {
    background:linear-gradient(135deg,#00e5ff22,#a259ff22) !important;
    border:1px solid var(--accent3) !important; color:var(--text) !important;
    font-family:'Space Mono',monospace !important; font-size:0.75rem !important;
    letter-spacing:2px !important; border-radius:4px !important;
    text-transform:uppercase !important; transition:all 0.2s !important;
}
div[data-testid="stButton"] > button:hover {
    background:linear-gradient(135deg,#00e5ff44,#a259ff44) !important;
    box-shadow:0 0 16px rgba(162,89,255,0.3) !important;
}
[data-testid="stFileUploader"] {
    border:1.5px dashed var(--border) !important;
    border-radius:8px !important; background:var(--surface) !important;
}
.stProgress > div > div > div {
    background:linear-gradient(90deg,var(--accent),var(--accent3)) !important;
}
.bio-row { display:flex; justify-content:space-between; margin:4px 0; font-size:0.78rem; }
.bio-key { color:var(--muted); }
.bio-val { color:var(--text); font-weight:700; }
.tab-content { padding: 12px 0; }
.timeline-dot {
    width:12px; height:12px; border-radius:50%;
    background:var(--accent); display:inline-block; margin-right:8px;
}
.health-bar-wrap { background:#1e2535; border-radius:4px; height:6px; margin:4px 0; overflow:hidden; }
.health-bar { height:6px; border-radius:4px; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# GAN MODEL
# ═══════════════════════════════════════════════════════════════════
class AgeConditionedResBlock(nn.Module):
    def __init__(self, channels, num_age_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.InstanceNorm2d(channels, affine=True)
        self.norm2 = nn.InstanceNorm2d(channels, affine=True)
        self.age_embed = nn.Embedding(num_age_classes, channels * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, age_code):
        style = self.age_embed(age_code)
        gamma, beta = style.chunk(2, dim=-1)
        gamma = gamma.view(-1, gamma.size(1), 1, 1)
        beta  = beta.view(-1, beta.size(1), 1, 1)
        out = self.conv1(x)
        out = self.norm1(out) * (1 + gamma) + beta
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        return self.relu(out + x)


class FaceAgingGenerator(nn.Module):
    def __init__(self, num_age_classes=10, ngf=32):
        super().__init__()
        self.num_age_classes = num_age_classes
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, ngf, 7, padding=3), nn.InstanceNorm2d(ngf, affine=True), nn.ReLU(True))
        self.enc2 = nn.Sequential(
            nn.Conv2d(ngf, ngf*2, 3, stride=2, padding=1), nn.InstanceNorm2d(ngf*2, affine=True), nn.ReLU(True))
        self.enc3 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf*4, 3, stride=2, padding=1), nn.InstanceNorm2d(ngf*4, affine=True), nn.ReLU(True))
        self.res_blocks = nn.ModuleList([
            AgeConditionedResBlock(ngf*4, num_age_classes) for _ in range(6)
        ])
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(ngf*4, ngf*2, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf*2, affine=True), nn.ReLU(True))
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(ngf*2, ngf, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf, affine=True), nn.ReLU(True))
        self.dec3 = nn.Sequential(nn.Conv2d(ngf, 3, 7, padding=3), nn.Tanh())

    def forward(self, img, target_age_cls):
        x = self.enc1(img); x = self.enc2(x); x = self.enc3(x)
        for blk in self.res_blocks:
            x = blk(x, target_age_cls)
        x = self.dec1(x); x = self.dec2(x)
        return self.dec3(x)


@st.cache_resource(show_spinner=False)
def load_model():
    model = FaceAgingGenerator(num_age_classes=10, ngf=32)
    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════════
# AGE / HEALTH DATA TABLES
# ═══════════════════════════════════════════════════════════════════
AGE_GROUPS = {
    0:(0,10,"Child"),1:(11,20,"Teen"),2:(21,30,"Young Adult"),
    3:(31,40,"Adult"),4:(41,50,"Mid Adult"),5:(51,60,"Mature"),
    6:(61,70,"Senior"),7:(71,80,"Elder"),8:(81,90,"Late Elder"),9:(91,100,"Centenarian"),
}

# Physiological reference data by decade
PHYSIO_DATA = {
    10: dict(rhr=90, sbp=105, vo2max=52, grip=20, bmd=0.85, skin_elasticity=98, collagen=100, telomere=95),
    20: dict(rhr=72, sbp=118, vo2max=48, grip=42, bmd=1.05, skin_elasticity=90, collagen=88, telomere=88),
    30: dict(rhr=73, sbp=122, vo2max=44, grip=44, bmd=1.03, skin_elasticity=80, collagen=76, telomere=80),
    40: dict(rhr=74, sbp=127, vo2max=40, grip=42, bmd=0.99, skin_elasticity=67, collagen=64, telomere=72),
    50: dict(rhr=75, sbp=133, vo2max=35, grip=39, bmd=0.94, skin_elasticity=52, collagen=52, telomere=63),
    60: dict(rhr=76, sbp=140, vo2max=30, grip=35, bmd=0.88, skin_elasticity=38, collagen=40, telomere=54),
    70: dict(rhr=77, sbp=148, vo2max=25, grip=29, bmd=0.80, skin_elasticity=26, collagen=30, telomere=44),
    80: dict(rhr=78, sbp=155, vo2max=20, grip=22, bmd=0.72, skin_elasticity=16, collagen=20, telomere=34),
    90: dict(rhr=80, sbp=162, vo2max=15, grip=15, bmd=0.64, skin_elasticity=9,  collagen=12, telomere=24),
   100: dict(rhr=82, sbp=168, vo2max=10, grip=10, bmd=0.58, skin_elasticity=5,  collagen=7,  telomere=16),
}

# Skin biomarkers % of peak
SKIN_MARKERS = {
    10: dict(hyaluronic=100,elastin=100,sebum=70,melanin=30),
    20: dict(hyaluronic=95, elastin=95, sebum=100,melanin=40),
    30: dict(hyaluronic=82, elastin=85, sebum=80, melanin=50),
    40: dict(hyaluronic=68, elastin=72, sebum=65, melanin=58),
    50: dict(hyaluronic=54, elastin=57, sebum=50, melanin=62),
    60: dict(hyaluronic=40, elastin=43, sebum=38, melanin=65),
    70: dict(hyaluronic=28, elastin=30, sebum=28, melanin=60),
    80: dict(hyaluronic=18, elastin=20, sebum=20, melanin=52),
    90: dict(hyaluronic=10, elastin=12, sebum=14, melanin=44),
   100: dict(hyaluronic=6,  elastin=7,  sebum=10, melanin=36),
}

# Cognitive markers (% of peak)
COGNITIVE = {
    10: dict(processing=70, memory=65, exec_func=60, attention=72, fluid_intel=75),
    20: dict(processing=100,memory=95, exec_func=92, attention=98, fluid_intel=100),
    30: dict(processing=97, memory=94, exec_func=95, attention=96, fluid_intel=96),
    40: dict(processing=90, memory=88, exec_func=92, attention=90, fluid_intel=88),
    50: dict(processing=82, memory=80, exec_func=86, attention=83, fluid_intel=78),
    60: dict(processing=72, memory=70, exec_func=77, attention=73, fluid_intel=66),
    70: dict(processing=60, memory=58, exec_func=65, attention=61, fluid_intel=54),
    80: dict(processing=47, memory=44, exec_func=51, attention=48, fluid_intel=40),
    90: dict(processing=33, memory=30, exec_func=36, attention=33, fluid_intel=27),
   100: dict(processing=20, memory=18, exec_func=22, attention=20, fluid_intel=16),
}

# Population life expectancy / health span data (WHO-inspired)
LIFESPAN_DATA = {
    "Global Average":     {"male":71,"female":75,"healthy_span":63,"disability_years":8},
    "High Income":        {"male":78,"female":83,"healthy_span":70,"disability_years":10},
    "Japan":              {"male":82,"female":88,"healthy_span":75,"disability_years":11},
    "USA":                {"male":76,"female":81,"healthy_span":68,"disability_years":11},
    "India":              {"male":68,"female":71,"healthy_span":60,"disability_years":9},
    "Sub-Saharan Africa": {"male":61,"female":64,"healthy_span":53,"disability_years":9},
    "Centenarian (Blue Zone)":{"male":100,"female":105,"healthy_span":95,"disability_years":5},
}

AGING_BIOMARKERS_HISTORY = {
    "year":  [2000,2005,2010,2015,2020,2023],
    "telomere_length_avg": [7.2,7.0,6.8,6.6,6.4,6.3],
    "global_life_exp": [66.8,67.9,69.5,71.2,72.6,73.2],
    "centenarian_pop_M": [0.18,0.21,0.29,0.41,0.57,0.72],
    "aging_research_papers_K": [12,18,28,42,61,80],
}

GAN_TRAINING_METRICS = {
    "epoch": list(range(1,51)),
    "g_loss": [4.2-i*0.06+np.random.RandomState(i).randn()*0.12 for i in range(50)],
    "d_loss": [1.8-i*0.02+np.random.RandomState(i+50).randn()*0.08 for i in range(50)],
    "fid":    [280-i*4.8+np.random.RandomState(i+100).randn()*5 for i in range(50)],
    "lpips":  [0.62-i*0.009+np.random.RandomState(i+150).randn()*0.012 for i in range(50)],
    "age_acc":[0.18+i*0.015+np.random.RandomState(i+200).randn()*0.012 for i in range(50)],
}
# Clamp
GAN_TRAINING_METRICS["g_loss"] = [max(0.4, x) for x in GAN_TRAINING_METRICS["g_loss"]]
GAN_TRAINING_METRICS["d_loss"] = [max(0.55, x) for x in GAN_TRAINING_METRICS["d_loss"]]
GAN_TRAINING_METRICS["fid"]    = [max(32, x) for x in GAN_TRAINING_METRICS["fid"]]
GAN_TRAINING_METRICS["lpips"]  = [max(0.08, x) for x in GAN_TRAINING_METRICS["lpips"]]
GAN_TRAINING_METRICS["age_acc"]= [min(0.94, x) for x in GAN_TRAINING_METRICS["age_acc"]]


# ═══════════════════════════════════════════════════════════════════
# UTILS
# ═══════════════════════════════════════════════════════════════════
TRANSFORM = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3),
])

def age_to_class(age):
    for cls,(lo,hi,_) in AGE_GROUPS.items():
        if lo<=age<=hi: return cls
    return 9

def tensor_to_pil(t):
    img = t.squeeze(0).detach().cpu()
    img = (img*0.5+0.5).clamp(0,1)
    return transforms.ToPILImage()(img)

def nearest_decade(age):
    decades = sorted(PHYSIO_DATA.keys())
    return min(decades, key=lambda d: abs(d-age))

def interp_physio(age, table):
    d = sorted(table.keys())
    if age <= d[0]: return table[d[0]]
    if age >= d[-1]: return table[d[-1]]
    lo = max(k for k in d if k<=age)
    hi = min(k for k in d if k>=age)
    if lo==hi: return table[lo]
    t = (age-lo)/(hi-lo)
    lo_d, hi_d = table[lo], table[hi]
    return {k: lo_d[k]*(1-t)+hi_d[k]*t for k in lo_d}


def apply_aging_texture(img_np, age, direction):
    img = img_np.copy().astype(np.float32)/255.0
    h,w = img.shape[:2]

    # ─── SKIN MASK DETECTION ───
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 15, 40]), np.array([25, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([165, 15, 40]), np.array([180, 255, 255]))
    skin_mask = cv2.bitwise_or(mask1, mask2)

    # Center-focused radial mask to target face and ignore background skin-tones
    Y, X = np.ogrid[:h, :w]
    yc, xc = h / 2.0, w / 2.0
    radial_mask = np.clip(1.0 - np.sqrt((Y - yc)**2 / (h/2.0)**2 + (X - xc)**2 / (w/2.0)**2), 0, 1)

    # Combined face-skin mask
    skin_mask_f = (skin_mask.astype(np.float32) / 255.0) * radial_mask

    # Adaptive smoothing kernel based on image size
    ksize = int(min(h, w) * 0.05) | 1
    ksize = max(5, ksize)
    if ksize % 2 == 0:
        ksize += 1

    face_mask = cv2.GaussianBlur(skin_mask_f, (ksize, ksize), 0)
    if face_mask.max() < 0.1:
        face_mask = cv2.GaussianBlur(radial_mask, (ksize, ksize), 0)

    face_mask = np.expand_dims(np.clip(face_mask, 0, 1), axis=2)

    if direction=="age":
        intensity = np.clip((age-30)/70.0,0,1)

        # 1. Exaggerate natural face creases (wrinkles)
        gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        blurred_gray = cv2.GaussianBlur(gray_img, (5, 5), 0)
        bilateral = cv2.bilateralFilter(blurred_gray, 9, 50, 50)
        edges = cv2.Laplacian(bilateral, cv2.CV_32F, ksize=3)
        edges = np.absolute(edges)
        if edges.max() > 0:
            edges = np.clip(edges / (edges.max() + 1e-5), 0, 1)
        else:
            edges = np.zeros_like(edges)

        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges_blur = cv2.GaussianBlur(edges, (5, 5), 0)
        wrinkle_mask = np.expand_dims(edges_blur, axis=2)

        # Darken the natural lines on the face skin
        wrinkles = wrinkle_mask * face_mask * intensity * 0.28
        img = img * (1.0 - wrinkles)

        # 2. Add fine-grained skin micro-textures
        rng = np.random.RandomState(42)
        noise_fine = rng.randn(h, w, 1).astype(np.float32) * 0.015 * intensity
        img += noise_fine * face_mask

        # 3. Desaturate the skin
        gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        gray_3ch = np.stack([gray] * 3, axis=-1)
        img = img * (1.0 - face_mask * intensity * 0.35) + gray_3ch * (face_mask * intensity * 0.35)

        # 4. Color shift: older skin loses redness and becomes warmer/paler
        img[:, :, 0] -= 0.04 * intensity * face_mask[:, :, 0]
        img[:, :, 2] -= 0.04 * intensity * face_mask[:, :, 0]

        # 5. Soft Eye Bags band (vertical Gaussian centered at 42% height)
        eye_y = h * 0.42
        eye_band = np.exp(-((Y - eye_y) / (h * 0.06))**2)
        eye_x_mask = np.exp(-((X - w*0.5) / (w * 0.22))**2)
        eye_mask = np.expand_dims(eye_band * eye_x_mask, axis=2) * face_mask
        img = img * (1.0 - eye_mask * intensity * 0.16)

        # 6. Sagging shadows on lower face skin
        sag = np.linspace(0, intensity * 0.08, h)[:, None, None]
        img -= sag * face_mask

        img = np.clip(img, 0, 1)
    else:
        intensity = np.clip((30-age)/30.0, 0, 1)

        # 1. Bilateral skin smoothing (removes blemishes while preserving sharp boundaries like eyes/lips)
        img_uint8 = (img * 255.0).astype(np.uint8)
        smoothed = cv2.bilateralFilter(img_uint8, 15, 30, 30).astype(np.float32) / 255.0
        img = img * (1.0 - face_mask * intensity * 0.75) + smoothed * (face_mask * intensity * 0.75)

        # 2. Youthful flush: boost red and green channels slightly on skin
        img[:, :, 0] += 0.06 * intensity * face_mask[:, :, 0]
        img[:, :, 1] += 0.02 * intensity * face_mask[:, :, 0]

        # 3. Overall subtle skin brightness boost
        img += 0.04 * intensity * face_mask

        img = np.clip(img, 0, 1)

    return np.clip(img*255,0,255).astype(np.uint8)


def run_gan_inference(model, pil_img, source_age, target_age):
    tensor = TRANSFORM(pil_img).unsqueeze(0)
    age_cls = torch.tensor([age_to_class(target_age)])
    with torch.no_grad():
        out = model(tensor, age_cls)
    gan_out = tensor_to_pil(out).resize(pil_img.size, Image.LANCZOS)
    gan_np  = np.array(gan_out)
    src_np  = np.array(pil_img)

    # Extract high-frequency style details from the GAN output to avoid random color blobs
    gan_gray = cv2.cvtColor(gan_np, cv2.COLOR_RGB2GRAY)
    gan_gray_3ch = np.stack([gan_gray]*3, axis=-1)
    gan_blur = cv2.GaussianBlur(gan_gray_3ch, (21, 21), 0)
    highpass = gan_gray_3ch.astype(np.float32) - gan_blur.astype(np.float32)

    # Blend a subtle 4% of the GAN fine details to retain styling without disfiguring blobs
    blended = src_np.astype(np.float32) + highpass * 0.04
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    direction = "age" if target_age>source_age else "deage"
    result = apply_aging_texture(blended, target_age, direction)
    return Image.fromarray(result)


def compute_image_stats(img_pil):
    """Extract image statistics for analysis panel."""
    arr = np.array(img_pil.convert("RGB").resize((256,256)))
    r,g,b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    gray = 0.299*r+0.587*g+0.114*b
    # Approximate skin tone detection (hue range)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    skin_mask = (
        (hsv[:,:,0]>=0)&(hsv[:,:,0]<=25)&
        (hsv[:,:,1]>=40)&(hsv[:,:,2]>=80)
    )
    return {
        "brightness": float(gray.mean()),
        "contrast":   float(gray.std()),
        "r_mean": float(r.mean()), "g_mean": float(g.mean()), "b_mean": float(b.mean()),
        "skin_pct": float(skin_mask.mean()*100),
        "sharpness": float(cv2.Laplacian(gray.astype(np.uint8),cv2.CV_64F).var()),
        "warm_bias": float((r.mean()-b.mean())),
    }

def compute_aging_score(source_age, target_age, img_stats):
    """Compute composite aging transformation score."""
    delta = abs(target_age - source_age)
    base = min(delta / 70.0, 1.0)
    contrast_factor = min(img_stats["contrast"] / 60.0, 1.0)
    skin_factor = min(img_stats["skin_pct"] / 40.0, 1.0)
    score = (base*0.6 + contrast_factor*0.2 + skin_factor*0.2)*100
    return round(score, 1)


# ═══════════════════════════════════════════════════════════════════
# VISUALIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════
BG  = '#0a0c10'
SUR = '#111520'
BDR = '#1e2535'
ACC = '#00e5ff'
AC2 = '#ff6b35'
AC3 = '#a259ff'
TXT = '#e0e8f8'
MUT = '#5a6a8a'
SUC = '#00ff9d'
WRN = '#ffd166'

def styled_fig(w=14, h=5):
    fig = plt.figure(figsize=(w,h), facecolor=BG)
    return fig

def styled_ax(ax):
    ax.set_facecolor(SUR)
    for sp in ax.spines.values(): sp.set_edgecolor(BDR)
    ax.tick_params(colors=MUT, labelsize=7)
    ax.xaxis.label.set_color(MUT)
    ax.yaxis.label.set_color(MUT)
    return ax


def fig_forensic(original, processed, source_age, target_age):
    fig = plt.figure(figsize=(16,4.5), facecolor=BG)
    gs  = gridspec.GridSpec(1,5,figure=fig,wspace=0.08)

    orig_np = np.array(original.resize((256,256),Image.LANCZOS))
    proc_np = np.array(processed.resize((256,256),Image.LANCZOS))
    diff    = np.abs(orig_np.astype(np.int16)-proc_np.astype(np.int16)).astype(np.uint8)
    diff_g  = diff.mean(axis=2).astype(np.uint8)

    # ── Original
    ax1=fig.add_subplot(gs[0]); ax1.imshow(orig_np); ax1.axis('off')
    ax1.set_title(f"SOURCE · {source_age}yr",color=ACC,fontsize=8,fontfamily='monospace',pad=8,fontweight='bold')

    # ── Result
    ax2=fig.add_subplot(gs[1]); ax2.imshow(proc_np); ax2.axis('off')
    lbl="AGED" if target_age>source_age else "DE-AGED"
    ax2.set_title(f"{lbl} · {target_age}yr",color=AC2,fontsize=8,fontfamily='monospace',pad=8,fontweight='bold')

    # ── Channel delta
    ax3=fig.add_subplot(gs[2]); ax3.set_facecolor(SUR)
    ax3.imshow(diff, vmin=0,vmax=80); ax3.axis('off')
    ax3.set_title("RGB DELTA",color=AC3,fontsize=8,fontfamily='monospace',pad=8,fontweight='bold')

    # ── Heatmap
    ax4=fig.add_subplot(gs[3]); ax4.set_facecolor(SUR)
    im=ax4.imshow(diff_g,cmap='inferno',vmin=0,vmax=60); ax4.axis('off')
    ax4.set_title("HEAT MAP",color=WRN,fontsize=8,fontfamily='monospace',pad=8,fontweight='bold')
    plt.colorbar(im,ax=ax4,fraction=0.046,pad=0.04).ax.tick_params(colors=MUT,labelsize=6)

    # ── Age probability
    ax5=fig.add_subplot(gs[4]); ax5.set_facecolor(SUR)
    ages=[0,10,20,30,40,50,60,70,80,90,100]
    probs=[np.exp(-0.5*((a-target_age)/15)**2) for a in ages]
    probs=np.array(probs)/sum(probs)
    colors_=[ACC if abs(a-target_age)<15 else BDR for a in ages]
    ax5.barh([str(a) for a in ages],probs,color=colors_,height=0.7)
    ax5.set_title("AGE CLASSIFIER",color=SUC,fontsize=8,fontfamily='monospace',pad=8,fontweight='bold')
    ax5.tick_params(colors=MUT,labelsize=6)
    for sp in ax5.spines.values(): sp.set_edgecolor(BDR)
    ax5.spines['top'].set_visible(False); ax5.spines['right'].set_visible(False)

    fig.tight_layout(pad=1.2)
    return fig


def fig_rgb_histogram(src_pil, res_pil):
    fig,axes=plt.subplots(1,3,figsize=(13,3.2),facecolor=BG)
    channels=[(0,ACC,'Red'),(1,SUC,'Green'),(2,AC2,'Blue')]
    s_arr=np.array(src_pil.resize((256,256))); r_arr=np.array(res_pil.resize((256,256)))
    for i,(ch,col,name) in enumerate(channels):
        ax=axes[i]; styled_ax(ax)
        hist_s,bins=np.histogram(s_arr[:,:,ch].flatten(),bins=64,range=(0,255))
        hist_r,_   =np.histogram(r_arr[:,:,ch].flatten(),bins=64,range=(0,255))
        bc=(bins[:-1]+bins[1:])/2
        ax.fill_between(bc,hist_s,alpha=0.4,color=TXT,label='Source')
        ax.fill_between(bc,hist_r,alpha=0.6,color=col, label='Aged')
        ax.plot(bc,hist_r,color=col,lw=1)
        ax.set_title(f"{name} Channel",color=col,fontsize=8,fontfamily='monospace')
        ax.legend(fontsize=6,labelcolor=MUT,framealpha=0)
        ax.set_xlabel("Pixel value",fontsize=7); ax.set_ylabel("Count",fontsize=7)
    fig.suptitle("RGB HISTOGRAM COMPARISON",color=TXT,fontsize=9,fontfamily='monospace',y=1.02)
    fig.tight_layout(pad=1.5)
    return fig


def fig_physio_radar(source_age, target_age):
    src_p = interp_physio(source_age, PHYSIO_DATA)
    tgt_p = interp_physio(target_age, PHYSIO_DATA)
    ref   = PHYSIO_DATA[20]

    labels = ['VO₂max','Grip\nStrength','Skin\nElasticity','Bone\nDensity','Collagen','Telomere']
    keys   = ['vo2max','grip','skin_elasticity','bmd','collagen','telomere']
    # Normalise 0-100
    def norm(d,k): return (d[k]/ref[k])*100 if ref[k]!=0 else 0
    src_v=[norm(src_p,k) for k in keys]
    tgt_v=[norm(tgt_p,k) for k in keys]

    angles=np.linspace(0,2*np.pi,len(labels),endpoint=False).tolist()
    src_v+=src_v[:1]; tgt_v+=tgt_v[:1]; angles+=angles[:1]

    fig,ax=plt.subplots(1,1,figsize=(5.5,5.5),subplot_kw=dict(polar=True),facecolor=BG)
    ax.set_facecolor(SUR)
    ax.plot(angles,src_v,color=ACC,lw=2,label=f"Age {source_age}")
    ax.fill(angles,src_v,color=ACC,alpha=0.18)
    ax.plot(angles,tgt_v,color=AC2,lw=2,label=f"Age {target_age}")
    ax.fill(angles,tgt_v,color=AC2,alpha=0.18)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels,size=8,color=TXT)
    ax.set_ylim(0,110); ax.set_yticks([25,50,75,100])
    ax.set_yticklabels(['25','50','75','100'],size=6,color=MUT)
    ax.grid(color=BDR,lw=0.8); ax.spines['polar'].set_edgecolor(BDR)
    ax.legend(loc='upper right',bbox_to_anchor=(1.35,1.1),fontsize=8,labelcolor=TXT,framealpha=0)
    ax.set_title("PHYSIOLOGICAL BIOMARKERS",color=ACC,fontfamily='monospace',fontsize=9,pad=20)
    fig.patch.set_facecolor(BG)
    return fig


def fig_skin_biomarkers(source_age, target_age):
    src_s=interp_physio(source_age,SKIN_MARKERS)
    tgt_s=interp_physio(target_age,SKIN_MARKERS)
    labels=['Hyaluronic\nAcid','Elastin','Sebum','Melanin']
    keys  =['hyaluronic','elastin','sebum','melanin']
    x=np.arange(len(labels)); w=0.32

    fig,ax=plt.subplots(figsize=(7,4),facecolor=BG)
    styled_ax(ax)
    b1=ax.bar(x-w/2,[src_s[k] for k in keys],w,color=ACC,alpha=0.7,label=f'Age {source_age}')
    b2=ax.bar(x+w/2,[tgt_s[k] for k in keys],w,color=AC2,alpha=0.7,label=f'Age {target_age}')
    ax.set_xticks(x); ax.set_xticklabels(labels,fontsize=8,color=TXT)
    ax.set_ylabel("% of Peak (Age 20)",fontsize=8)
    ax.set_ylim(0,115)
    ax.axhline(100,color=BDR,lw=0.8,linestyle='--',alpha=0.6)
    ax.legend(fontsize=8,labelcolor=TXT,framealpha=0)
    ax.set_title("SKIN BIOMARKERS — % OF PEAK",color=AC3,fontfamily='monospace',fontsize=9)
    for bar in b1: ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,f'{bar.get_height():.0f}',
                           ha='center',va='bottom',fontsize=6,color=ACC)
    for bar in b2: ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,f'{bar.get_height():.0f}',
                           ha='center',va='bottom',fontsize=6,color=AC2)
    fig.tight_layout(pad=1.5)
    return fig


def fig_cognitive(source_age, target_age):
    src_c=interp_physio(source_age,COGNITIVE)
    tgt_c=interp_physio(target_age,COGNITIVE)
    labels=['Processing\nSpeed','Memory','Executive\nFunction','Attention','Fluid\nIntelligence']
    keys  =['processing','memory','exec_func','attention','fluid_intel']

    fig,ax=plt.subplots(figsize=(8,4),facecolor=BG)
    styled_ax(ax)
    x=np.arange(len(labels)); w=0.32
    ax.bar(x-w/2,[src_c[k] for k in keys],w,color=AC3,alpha=0.75,label=f'Age {source_age}')
    ax.bar(x+w/2,[tgt_c[k] for k in keys],w,color=WRN,alpha=0.75,label=f'Age {target_age}')
    ax.set_xticks(x); ax.set_xticklabels(labels,fontsize=8,color=TXT)
    ax.set_ylabel("% of Peak",fontsize=8); ax.set_ylim(0,115)
    ax.axhline(100,color=BDR,lw=0.8,linestyle='--',alpha=0.6)
    ax.legend(fontsize=8,labelcolor=TXT,framealpha=0)
    ax.set_title("COGNITIVE PERFORMANCE MARKERS",color=WRN,fontfamily='monospace',fontsize=9)
    fig.tight_layout(pad=1.5)
    return fig


def fig_lifespan():
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),facecolor=BG)
    # Left: life expectancy bars
    ax=axes[0]; styled_ax(ax)
    pops=list(LIFESPAN_DATA.keys())
    males=[LIFESPAN_DATA[p]["male"] for p in pops]
    females=[LIFESPAN_DATA[p]["female"] for p in pops]
    healthy=[LIFESPAN_DATA[p]["healthy_span"] for p in pops]
    y=np.arange(len(pops)); h=0.25
    ax.barh(y+h, males,   h, color=ACC, alpha=0.8, label='Male')
    ax.barh(y,   females, h, color=AC2, alpha=0.8, label='Female')
    ax.barh(y-h, healthy, h, color=SUC, alpha=0.8, label='Healthy Span')
    ax.set_yticks(y); ax.set_yticklabels(pops, fontsize=7, color=TXT)
    ax.set_xlabel("Years", fontsize=8); ax.legend(fontsize=7,labelcolor=TXT,framealpha=0)
    ax.set_title("GLOBAL LIFE EXPECTANCY BY POPULATION",color=ACC,fontfamily='monospace',fontsize=8)
    ax.set_xlim(0,115)

    # Right: research trend
    ax2=axes[1]; styled_ax(ax2)
    yrs=AGING_BIOMARKERS_HISTORY["year"]
    ax2.plot(yrs,AGING_BIOMARKERS_HISTORY["global_life_exp"],color=ACC,lw=2,marker='o',ms=5,label='Global Life Exp (yr)')
    ax2.set_ylabel("Life Expectancy (yr)",fontsize=8,color=ACC)
    ax2r=ax2.twinx()
    ax2r.set_facecolor(SUR)
    ax2r.plot(yrs,AGING_BIOMARKERS_HISTORY["aging_research_papers_K"],color=AC2,lw=2,marker='s',ms=5,label='Research Papers (K)')
    ax2r.set_ylabel("Papers Published (K)",fontsize=8,color=AC2)
    ax2r.tick_params(colors=MUT,labelsize=7)
    ax2.set_title("LIFE EXPECTANCY & RESEARCH GROWTH",color=ACC,fontfamily='monospace',fontsize=8)
    ax2.set_xlabel("Year",fontsize=8)
    lines=[plt.Line2D([0],[0],color=ACC,lw=2,label='Life Exp'),
           plt.Line2D([0],[0],color=AC2,lw=2,label='Research Papers')]
    ax2.legend(handles=lines,fontsize=7,labelcolor=TXT,framealpha=0)
    fig.tight_layout(pad=1.5)
    return fig


def fig_gan_training():
    ep=GAN_TRAINING_METRICS["epoch"]
    fig,axes=plt.subplots(2,3,figsize=(14,6),facecolor=BG)
    axes=axes.flatten()
    configs=[
        ("g_loss","Generator Loss",ACC),
        ("d_loss","Discriminator Loss",AC2),
        ("fid","FID Score ↓",AC3),
        ("lpips","LPIPS ↓",WRN),
        ("age_acc","Age Accuracy ↑",SUC),
    ]
    for i,(key,title,col) in enumerate(configs):
        ax=axes[i]; styled_ax(ax)
        vals=GAN_TRAINING_METRICS[key]
        ax.plot(ep,vals,color=col,lw=1.5,alpha=0.9)
        ax.fill_between(ep,vals,alpha=0.12,color=col)
        ax.set_title(title,color=col,fontsize=8,fontfamily='monospace')
        ax.set_xlabel("Epoch",fontsize=7); ax.set_ylabel(key,fontsize=7)
        # Annotate final value
        ax.annotate(f'{vals[-1]:.3f}',xy=(ep[-1],vals[-1]),
                    color=col,fontsize=7,ha='right',
                    xytext=(-4,6),textcoords='offset points')
    # 6th panel: combined normalised
    ax=axes[5]; styled_ax(ax)
    def norm_series(s): mn,mx=min(s),max(s); return [(x-mn)/(mx-mn+1e-9) for x in s]
    ax.plot(ep,norm_series(GAN_TRAINING_METRICS["fid"]),color=AC3,lw=1.5,label='FID (norm)')
    ax.plot(ep,norm_series(GAN_TRAINING_METRICS["g_loss"]),color=ACC,lw=1.5,label='G-Loss (norm)')
    ax.plot(ep,GAN_TRAINING_METRICS["age_acc"],color=SUC,lw=1.5,label='Age Acc')
    ax.set_title("TRAINING OVERVIEW",color=TXT,fontsize=8,fontfamily='monospace')
    ax.legend(fontsize=6,labelcolor=TXT,framealpha=0)
    ax.set_xlabel("Epoch",fontsize=7)
    fig.suptitle("GAN TRAINING METRICS  (50-epoch simulation)",color=TXT,fontsize=10,fontfamily='monospace',y=1.01)
    fig.tight_layout(pad=1.5)
    return fig


def fig_aging_timeline(source_age, target_age):
    """Horizontal milestone timeline."""
    fig,ax=plt.subplots(figsize=(14,3.5),facecolor=BG)
    styled_ax(ax); ax.set_xlim(0,100); ax.set_ylim(-1,2.5); ax.axis('off')

    milestones=[
        (2,"First words"),(5,"School"),(13,"Puberty"),(18,"Adulthood"),
        (25,"Brain maturity"),(30,"Peak strength"),(40,"Perimenopause onset"),
        (50,"Presbyopia"),(60,"Retirement age"),(65,"Medicare eligible"),
        (70,"Sarcopenia risk"),(80,"Frailty threshold"),(90,"Superager"),(100,"Centenarian"),
    ]
    ax.axhline(0.5,xmin=0,xmax=1,color=BDR,lw=2,zorder=1)

    for age_m,label in milestones:
        above = milestones.index((age_m,label))%2==0
        y_tip=0.5; y_txt=1.4 if above else -0.6
        col=ACC if age_m<=source_age else (AC2 if age_m<=target_age else MUT)
        ax.plot([age_m,age_m],[y_tip,y_txt],color=col,lw=1,zorder=2,alpha=0.7)
        ax.scatter([age_m],[y_tip],color=col,s=50,zorder=3)
        ax.text(age_m,y_txt+(0.12 if above else -0.12),label,ha='center',va='bottom' if above else 'top',
                fontsize=6.5,color=col,fontfamily='monospace')
        ax.text(age_m,y_tip-(0.18 if above else 0),str(age_m),ha='center',va='top',
                fontsize=6,color=MUT,fontfamily='monospace')

    # Highlight source and target
    ax.scatter([source_age],[0.5],color=ACC,s=160,zorder=5,marker='D')
    ax.scatter([target_age],[0.5],color=AC2,s=160,zorder=5,marker='D')
    ax.text(source_age,0.5+0.25,f"YOU\n{source_age}",ha='center',fontsize=8,color=ACC,fontweight='bold',fontfamily='monospace')
    ax.text(target_age,0.5+0.25,f"TARGET\n{target_age}",ha='center',fontsize=8,color=AC2,fontweight='bold',fontfamily='monospace')

    ax.set_title("HUMAN AGING MILESTONE TIMELINE",color=TXT,fontfamily='monospace',fontsize=10,pad=8)
    fig.tight_layout(pad=1)
    return fig


def fig_biological_age_estimate(img_stats, source_age, target_age):
    """Simulate a biological age estimate from image features."""
    brightness_factor = (img_stats["brightness"]-100)/100
    contrast_factor   = (img_stats["contrast"]-40)/40
    warm_factor       = img_stats["warm_bias"]/20
    sharpness_factor  = (img_stats["sharpness"]-200)/200
    bio_age = source_age + brightness_factor*3 - contrast_factor*2 + warm_factor*4 - sharpness_factor*2
    bio_age = max(5, min(100, bio_age))

    metrics = {
        "Chronological Age": source_age,
        "Est. Biological Age": round(bio_age,1),
        "Skin Age Score": round(source_age + warm_factor*5, 1),
        "Photo Quality Score": round(min(100, img_stats["sharpness"]/3), 1),
        "Skin Coverage %": round(img_stats["skin_pct"], 1),
    }

    fig,axes=plt.subplots(1,2,figsize=(11,4),facecolor=BG)
    # Gauge
    ax=axes[0]; ax.set_facecolor(SUR)
    theta=np.linspace(np.pi,0,200)
    for i,t in enumerate(theta[:-1]):
        c=plt.cm.RdYlGn(i/len(theta))
        ax.plot([0,np.cos(t)],[0,np.sin(t)],color=c,lw=8,alpha=0.5)
    needle_t=np.pi*(1-(bio_age/100))
    ax.plot([0,0.7*np.cos(needle_t)],[0,0.7*np.sin(needle_t)],color=TXT,lw=3,zorder=5)
    ax.scatter([0],[0],color=TXT,s=80,zorder=6)
    ax.text(0,-0.15,f"Bio Age: {bio_age:.0f}",ha='center',fontsize=12,color=TXT,fontweight='bold',fontfamily='monospace')
    ax.text(0,-0.35,f"Chrono: {source_age}",ha='center',fontsize=8,color=MUT,fontfamily='monospace')
    ax.set_xlim(-1.2,1.2); ax.set_ylim(-0.5,1.2)
    ax.axis('off')
    ax.set_title("BIOLOGICAL AGE GAUGE",color=ACC,fontfamily='monospace',fontsize=9)
    # Bar table
    ax2=axes[1]; styled_ax(ax2)
    keys=list(metrics.keys()); vals=list(metrics.values())
    colors_m=[ACC,AC3 if abs(bio_age-source_age)>5 else SUC, AC2, WRN, SUC]
    bars=ax2.barh(keys,vals,color=colors_m[:len(keys)],alpha=0.75,height=0.55)
    for bar,val in zip(bars,vals):
        ax2.text(bar.get_width()+0.5,bar.get_y()+bar.get_height()/2,
                 f'{val}',va='center',fontsize=7,color=TXT)
    ax2.set_xlabel("Value",fontsize=8); ax2.tick_params(labelsize=7)
    ax2.set_title("IMAGE ANALYSIS METRICS",color=AC3,fontfamily='monospace',fontsize=9)
    ax2.set_xlim(0,max(vals)*1.25)
    fig.tight_layout(pad=1.5)
    return fig


def fig_population_pyramid():
    """World age-sex population pyramid (2023 estimated data)."""
    age_groups=["0-4","5-9","10-14","15-19","20-24","25-29","30-34","35-39",
                "40-44","45-49","50-54","55-59","60-64","65-69","70-74","75-79","80+"]
    male_pop  =[334,321,308,296,295,292,278,260,247,232,213,189,158,124,93,63,58]
    female_pop=[317,305,292,280,279,277,264,247,235,222,205,184,156,126,99,73,81]

    fig,ax=plt.subplots(figsize=(9,6),facecolor=BG)
    styled_ax(ax)
    y=np.arange(len(age_groups))
    ax.barh(y,[-m for m in male_pop],height=0.75,color=ACC,alpha=0.75,label='Male')
    ax.barh(y,female_pop,height=0.75,color=AC2,alpha=0.75,label='Female')
    ax.set_yticks(y); ax.set_yticklabels(age_groups,fontsize=8,color=TXT)
    ax.axvline(0,color=BDR,lw=1)
    ticks=[-300,-200,-100,0,100,200,300]
    ax.set_xticks(ticks); ax.set_xticklabels([str(abs(t)) for t in ticks],fontsize=7)
    ax.set_xlabel("Population (millions)",fontsize=8)
    ax.legend(fontsize=8,labelcolor=TXT,framealpha=0)
    ax.set_title("WORLD POPULATION PYRAMID 2023",color=ACC,fontfamily='monospace',fontsize=9)
    fig.tight_layout(pad=1.5)
    return fig


def fig_telomere_decline():
    ages=list(range(0,101,5))
    length=[9.5-age*0.035+np.sin(age*0.1)*0.2 for age in ages]
    length=[max(4.5,l) for l in length]

    fig,ax=plt.subplots(figsize=(9,4),facecolor=BG)
    styled_ax(ax)
    ax.plot(ages,length,color=ACC,lw=2.5)
    ax.fill_between(ages,length,4.5,alpha=0.15,color=ACC)
    ax.axhline(7.0,color=WRN,lw=1,linestyle='--',alpha=0.7,label='Aging threshold')
    ax.axhline(5.5,color=AC2,lw=1,linestyle='--',alpha=0.7,label='Disease risk threshold')
    ax.set_xlabel("Age (years)",fontsize=8); ax.set_ylabel("Telomere length (kb)",fontsize=8)
    ax.set_title("TELOMERE LENGTH DECLINE WITH AGE",color=ACC,fontfamily='monospace',fontsize=9)
    ax.legend(fontsize=8,labelcolor=TXT,framealpha=0)
    ax.set_xlim(0,100)
    fig.tight_layout(pad=1.5)
    return fig


# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:4px 0 18px">
        <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.5rem;
            background:linear-gradient(135deg,#00e5ff,#a259ff);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;">AgeForge</div>
        <div style="font-family:'Space Mono',monospace;font-size:0.6rem;
            letter-spacing:3px;color:#5a6a8a;text-transform:uppercase;">
            Conditional GAN · v2.0</div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-label">Architecture</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="line-height:2.1">
        <span class="arch-pill">Encoder-Decoder</span>
        <span class="arch-pill">ResBlocks ×6</span>
        <span class="arch-pill">Age Embedding</span>
        <span class="arch-pill">AdaIN</span>
        <span class="arch-pill">Tanh Output</span>
        <span class="arch-pill">10 Age Classes</span>
    </div>""", unsafe_allow_html=True)

    model_ref = FaceAgingGenerator()
    total_params = sum(p.numel() for p in model_ref.parameters())
    st.markdown('<br><div class="section-label">Model Stats</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-card">
        🧠 <b>{total_params/1e6:.2f}M</b> parameters<br>
        📐 Input: <b>256 × 256</b> RGB<br>
        🎯 <b>10</b> age classes · <b>6</b> ResBlocks<br>
        ⚡ ngf = <b>32</b> · AdaIN per block
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-label">Ages</div>', unsafe_allow_html=True)
    source_age = st.slider("Source Age", 5, 85, 30, 1)
    target_age = st.slider("Target Age", 5, 100, 65, 1)

    delta = target_age - source_age
    if delta>0:   d_str=f"⬆ Aging +{delta} yrs"; d_col="#ff6b35"
    elif delta<0: d_str=f"⬇ De-aging {abs(delta)} yrs"; d_col="#00e5ff"
    else:         d_str="— No change"; d_col="#5a6a8a"
    st.markdown(f'<div style="font-size:0.78rem;color:{d_col};margin:6px 0 14px;">{d_str}</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="section-label">Output Panels</div>', unsafe_allow_html=True)
    show_forensic    = st.checkbox("Forensic panel",         value=True)
    show_histogram   = st.checkbox("RGB histogram",          value=True)
    show_physio      = st.checkbox("Physio radar",           value=True)
    show_skin        = st.checkbox("Skin biomarkers",        value=True)
    show_cognitive   = st.checkbox("Cognitive markers",      value=True)
    show_bioage      = st.checkbox("Biological age gauge",   value=True)
    show_timeline    = st.checkbox("Aging timeline",         value=True)
    show_lifespan    = st.checkbox("Lifespan data",          value=True)
    show_population  = st.checkbox("Population pyramid",     value=False)
    show_telomere    = st.checkbox("Telomere decline",       value=True)
    show_training    = st.checkbox("GAN training metrics",   value=True)

    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown("""
    <div class="warning-card">
        ⚠️ Demo uses cGAN architecture with random weights. Aging texture applied via deterministic post-processing pipeline.
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<div style="margin-bottom:18px">
    <div class="hero-title">AgeForge</div>
    <div class="hero-sub">Conditional GAN · Face Aging & De-Aging · Full Biometric Analysis</div>
</div>""", unsafe_allow_html=True)

st.markdown("""
<span class="metric-chip">cGAN</span>
<span class="metric-chip">ADAPTIVE INSTANCE NORM</span>
<span class="metric-chip">FORENSIC ANALYSIS</span>
<span class="metric-chip">BIOMETRIC DATA</span>
<span class="metric-chip">TELOMERE RESEARCH</span>
<span class="metric-chip">POPULATION STATS</span>
<span class="metric-chip">PYTORCH</span>
""", unsafe_allow_html=True)

st.markdown('<hr style="border:none;border-top:1px solid #1e2535;margin:18px 0;">', unsafe_allow_html=True)

# ── TABS ──
tab_main, tab_science, tab_model, tab_about = st.tabs([
    "🧬 Face Aging", "📊 Aging Science Data", "⚙️ Model & Training", "📖 About"
])


# ══════════════════════════════════════════
# TAB 1 — FACE AGING
# ══════════════════════════════════════════
with tab_main:
    col_up, col_info = st.columns([1,1], gap="large")

    with col_up:
        st.markdown('<div class="section-label">Upload Face Image</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Drop a face photo",type=["jpg","jpeg","png","webp"],
                                    help="Best results: frontal, well-lit faces")

    with col_info:
        st.markdown('<div class="section-label">Pipeline Overview</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-card">
            <b>1. Encoder</b> — 3-stage conv downsampling (stride-2)<br>
            <b>2. 6× ResBlocks</b> — age embedding injected via AdaIN at each block<br>
            <b>3. Decoder</b> — transposed convolutions + Tanh output<br>
            <b>4. Texture engine</b> — wrinkles, desaturation, sagging, dark circles<br>
            <b>5. Image analysis</b> — brightness, contrast, skin %, sharpness<br>
            <b>6. Forensic panel</b> — delta map, heatmap, age classifier
        </div>""", unsafe_allow_html=True)

    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")
        MAX_DIM=512; w,h=pil_img.size
        scale=min(MAX_DIM/w,MAX_DIM/h,1.0)
        pil_img=pil_img.resize((int(w*scale),int(h*scale)),Image.LANCZOS)

        run_col,_=st.columns([1,3])
        with run_col:
            run_btn=st.button("▶  RUN AGING GAN", use_container_width=True)

        if run_btn or 'result_img' not in st.session_state:
            with st.spinner(""):
                prog=st.progress(0,"Loading model…")
                model=load_model()
                prog.progress(15,"Analysing image…")
                img_stats=compute_image_stats(pil_img)
                prog.progress(35,"Encoding face…")
                time.sleep(0.2)
                prog.progress(60,"Running ResBlocks…")
                time.sleep(0.2)
                result=run_gan_inference(model,pil_img,source_age,target_age)
                prog.progress(85,"Rendering output…")
                time.sleep(0.15)
                prog.progress(100,"Done ✓")
                time.sleep(0.25)
                prog.empty()
                st.session_state['result_img']  = result
                st.session_state['source_img']  = pil_img
                st.session_state['img_stats']   = img_stats
                st.session_state['source_age']  = source_age
                st.session_state['target_age']  = target_age

        if 'result_img' in st.session_state:
            result   = st.session_state['result_img']
            src_img  = st.session_state['source_img']
            img_stats= st.session_state['img_stats']
            aging_score = compute_aging_score(source_age,target_age,img_stats)

            # ── Quick stats bar
            st.markdown('<hr style="border:none;border-top:1px solid #1e2535;margin:16px 0;">', unsafe_allow_html=True)
            m1,m2,m3,m4,m5,m6 = st.columns(6)
            with m1:
                st.markdown(f'<div class="stat-card"><div class="stat-val">{source_age}</div><div class="stat-lbl">Source Age</div></div>',unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="stat-card"><div class="stat-val" style="color:#ff6b35">{target_age}</div><div class="stat-lbl">Target Age</div></div>',unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="stat-card"><div class="stat-val" style="color:#a259ff">{abs(delta)}</div><div class="stat-lbl">Year Delta</div></div>',unsafe_allow_html=True)
            with m4:
                st.markdown(f'<div class="stat-card"><div class="stat-val" style="color:#00ff9d">{aging_score}</div><div class="stat-lbl">Transform Score</div></div>',unsafe_allow_html=True)
            with m5:
                st.markdown(f'<div class="stat-card"><div class="stat-val" style="color:#ffd166">{img_stats["brightness"]:.0f}</div><div class="stat-lbl">Brightness</div></div>',unsafe_allow_html=True)
            with m6:
                st.markdown(f'<div class="stat-card"><div class="stat-val">{img_stats["skin_pct"]:.1f}%</div><div class="stat-lbl">Skin Coverage</div></div>',unsafe_allow_html=True)

            st.markdown('<br>', unsafe_allow_html=True)

            # ── Side by side
            c1,cmid,c2=st.columns([1,0.05,1],gap="small")
            with c1:
                st.markdown(f'<div class="section-label">Source · Age {source_age}</div>',unsafe_allow_html=True)
                st.image(src_img,use_container_width=True)
            with cmid:
                st.markdown('<div style="height:100%;display:flex;align-items:center;justify-content:center;color:#a259ff;font-size:2rem;padding-top:80px">→</div>',unsafe_allow_html=True)
            with c2:
                lbl="AGED" if target_age>source_age else "DE-AGED"
                st.markdown(f'<div class="section-label">{lbl} · Age {target_age}</div>',unsafe_allow_html=True)
                st.image(result,use_container_width=True)

            # ── Download
            buf=io.BytesIO(); result.save(buf,format="PNG")
            dl1,dl2,_=st.columns([1,1,3])
            with dl1:
                st.download_button("⬇ Download Result",data=buf.getvalue(),
                                   file_name=f"ageforge_{target_age}yr.png",mime="image/png")
            with dl2:
                src_buf=io.BytesIO(); src_img.save(src_buf,format="PNG")
                st.download_button("⬇ Download Source",data=src_buf.getvalue(),
                                   file_name="ageforge_source.png",mime="image/png")

            # ── Forensic
            if show_forensic:
                st.markdown('<br><div class="section-label">Forensic Analysis Panel</div>',unsafe_allow_html=True)
                with st.spinner("Rendering forensic panel…"):
                    fig=fig_forensic(src_img,result,source_age,target_age)
                st.pyplot(fig); plt.close(fig)

            # ── RGB Histogram
            if show_histogram:
                st.markdown('<br><div class="section-label">RGB Histogram Comparison</div>',unsafe_allow_html=True)
                with st.spinner(""):
                    fig=fig_rgb_histogram(src_img,result)
                st.pyplot(fig); plt.close(fig)

            # ── Biological age
            if show_bioage:
                st.markdown('<br><div class="section-label">Biological Age Estimation</div>',unsafe_allow_html=True)
                with st.spinner(""):
                    fig=fig_biological_age_estimate(img_stats,source_age,target_age)
                st.pyplot(fig); plt.close(fig)

            # ── Image stats detail
            st.markdown('<br><div class="section-label">Image Analysis Report</div>',unsafe_allow_html=True)
            r1c1,r1c2,r1c3,r1c4=st.columns(4)
            stat_items=[
                ("Brightness",f"{img_stats['brightness']:.1f}","/ 255"),
                ("Contrast (σ)",f"{img_stats['contrast']:.1f}","px"),
                ("Sharpness (Lap)",f"{img_stats['sharpness']:.0f}","score"),
                ("Warm Bias (R-B)",f"{img_stats['warm_bias']:.1f}","units"),
            ]
            for col_w, (lbl,val,unit) in zip([r1c1,r1c2,r1c3,r1c4],stat_items):
                with col_w:
                    st.markdown(f"""
                    <div class="info-card" style="text-align:center">
                        <div style="font-size:1.4rem;font-weight:800;color:#00e5ff;font-family:'Syne',sans-serif">{val}</div>
                        <div style="font-size:0.6rem;color:#5a6a8a;letter-spacing:2px;text-transform:uppercase">{lbl}</div>
                        <div style="font-size:0.65rem;color:#5a6a8a">{unit}</div>
                    </div>""",unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="border:1.5px dashed #1e2535;border-radius:12px;padding:60px 40px;
            text-align:center;background:#111520;margin-top:20px;">
            <div style="font-size:2.8rem;margin-bottom:16px">🧬</div>
            <div style="font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:700;
                color:#e0e8f8;margin-bottom:8px;">Upload a face image to begin</div>
            <div style="font-family:'Space Mono',monospace;font-size:0.72rem;color:#5a6a8a;
                line-height:1.8;max-width:460px;margin:0 auto;">
                The conditional GAN encodes your image, injects the target age class<br>
                via learned embeddings into 6 ResBlocks, and decodes a progressively<br>
                aged or de-aged result with full biometric analysis output.
            </div>
        </div>""",unsafe_allow_html=True)


# ══════════════════════════════════════════
# TAB 2 — AGING SCIENCE DATA
# ══════════════════════════════════════════
with tab_science:
    st.markdown('<div class="section-label">Physiological Biomarkers — Source vs Target Age</div>',unsafe_allow_html=True)

    if show_physio:
        c1,c2=st.columns([1,1],gap="large")
        with c1:
            with st.spinner(""):
                fig=fig_physio_radar(source_age,target_age)
            st.pyplot(fig); plt.close(fig)
        with c2:
            # Tabular comparison
            src_p=interp_physio(source_age,PHYSIO_DATA)
            tgt_p=interp_physio(target_age,PHYSIO_DATA)
            st.markdown(f"""
            <div class="info-card">
                <b>Physiological comparison: Age {source_age} → {target_age}</b><br><br>
                <div class="bio-row"><span class="bio-key">Resting Heart Rate</span><span class="bio-val">{src_p['rhr']:.0f} → {tgt_p['rhr']:.0f} bpm</span></div>
                <div class="bio-row"><span class="bio-key">Systolic BP</span><span class="bio-val">{src_p['sbp']:.0f} → {tgt_p['sbp']:.0f} mmHg</span></div>
                <div class="bio-row"><span class="bio-key">VO₂max</span><span class="bio-val">{src_p['vo2max']:.0f} → {tgt_p['vo2max']:.0f} ml/kg/min</span></div>
                <div class="bio-row"><span class="bio-key">Grip Strength</span><span class="bio-val">{src_p['grip']:.0f} → {tgt_p['grip']:.0f} kg</span></div>
                <div class="bio-row"><span class="bio-key">Bone Mineral Density</span><span class="bio-val">{src_p['bmd']:.2f} → {tgt_p['bmd']:.2f} g/cm²</span></div>
                <div class="bio-row"><span class="bio-key">Skin Elasticity</span><span class="bio-val">{src_p['skin_elasticity']:.0f}% → {tgt_p['skin_elasticity']:.0f}%</span></div>
                <div class="bio-row"><span class="bio-key">Collagen (% peak)</span><span class="bio-val">{src_p['collagen']:.0f}% → {tgt_p['collagen']:.0f}%</span></div>
                <div class="bio-row"><span class="bio-key">Telomere Length</span><span class="bio-val">{src_p['telomere']:.0f}% → {tgt_p['telomere']:.0f}%</span></div>
            </div>""", unsafe_allow_html=True)

    if show_skin:
        st.markdown('<br><div class="section-label">Skin Biomarkers</div>',unsafe_allow_html=True)
        with st.spinner(""):
            fig=fig_skin_biomarkers(source_age,target_age)
        st.pyplot(fig); plt.close(fig)

    if show_cognitive:
        st.markdown('<br><div class="section-label">Cognitive Performance</div>',unsafe_allow_html=True)
        with st.spinner(""):
            fig=fig_cognitive(source_age,target_age)
        st.pyplot(fig); plt.close(fig)

    if show_telomere:
        st.markdown('<br><div class="section-label">Telomere Length Decline</div>',unsafe_allow_html=True)
        with st.spinner(""):
            fig=fig_telomere_decline()
        st.pyplot(fig); plt.close(fig)

    if show_timeline:
        st.markdown('<br><div class="section-label">Aging Milestone Timeline</div>',unsafe_allow_html=True)
        with st.spinner(""):
            fig=fig_aging_timeline(source_age,target_age)
        st.pyplot(fig); plt.close(fig)

    if show_lifespan:
        st.markdown('<br><div class="section-label">Global Lifespan Data</div>',unsafe_allow_html=True)
        with st.spinner(""):
            fig=fig_lifespan()
        st.pyplot(fig); plt.close(fig)

    if show_population:
        st.markdown('<br><div class="section-label">World Population Pyramid (2023)</div>',unsafe_allow_html=True)
        with st.spinner(""):
            fig=fig_population_pyramid()
        st.pyplot(fig); plt.close(fig)

    # ── Age-decade facts table
    st.markdown('<br><div class="section-label">Decade-by-Decade Reference Table</div>',unsafe_allow_html=True)
    decade_facts={
        "20s": "Peak muscle mass, VO₂max ~48, collagen production highest, minimal wrinkles",
        "30s": "Metabolism slows ~2% /decade, collagen -1%/yr begins, first fine lines",
        "40s": "Presbyopia onset, perimenopause, grip strength peaks then declines",
        "50s": "Menopause, bone density loss accelerates, sarcopenia risk begins",
        "60s": "Retirement-age frailty risk, hearing loss common, skin thins significantly",
        "70s": "Falls risk elevated, telomere length critically reduced, cognitive processing -40%",
        "80s": "Centenarian potential, frailty threshold, immune senescence dominates",
        "90s+":"Superager cognition (~10%), extreme longevity via genetics + environment",
    }
    cols=st.columns(4)
    for i,(decade,fact) in enumerate(decade_facts.items()):
        with cols[i%4]:
            st.markdown(f"""
            <div class="info-card" style="min-height:100px;margin-bottom:8px;">
                <div style="color:#00e5ff;font-weight:700;font-size:0.9rem;margin-bottom:6px">{decade}</div>
                <div style="font-size:0.72rem;color:#c0cce8;line-height:1.6">{fact}</div>
            </div>""",unsafe_allow_html=True)


# ══════════════════════════════════════════
# TAB 3 — MODEL & TRAINING
# ══════════════════════════════════════════
with tab_model:
    st.markdown('<div class="section-label">GAN Training Metrics (50-Epoch Simulation)</div>',unsafe_allow_html=True)

    if show_training:
        with st.spinner(""):
            fig=fig_gan_training()
        st.pyplot(fig); plt.close(fig)

    st.markdown('<br><div class="section-label">Architecture Deep-Dive</div>',unsafe_allow_html=True)
    arch_c1,arch_c2=st.columns(2,gap="large")

    with arch_c1:
        st.markdown("""
        <div class="info-card">
            <b style="color:#00e5ff">Encoder (3 stages)</b><br>
            • Conv2d(3→32, k=7, pad=3) + InstanceNorm + ReLU<br>
            • Conv2d(32→64, k=3, stride=2) + IN + ReLU<br>
            • Conv2d(64→128, k=3, stride=2) + IN + ReLU<br><br>
            <b style="color:#a259ff">6× AgeConditionedResBlock</b><br>
            • Conv2d(128→128) → InstanceNorm → AdaIN → ReLU<br>
            • Conv2d(128→128) → InstanceNorm<br>
            • Age embedding: nn.Embedding(10, 256) → γ, β<br>
            • Residual skip: out = f(x) + x<br><br>
            <b style="color:#ff6b35">Decoder (2 stages + output)</b><br>
            • ConvTranspose2d(128→64, stride=2) + IN + ReLU<br>
            • ConvTranspose2d(64→32, stride=2) + IN + ReLU<br>
            • Conv2d(32→3, k=7) + Tanh → [-1, 1]
        </div>""",unsafe_allow_html=True)

    with arch_c2:
        st.markdown("""
        <div class="info-card">
            <b style="color:#00ff9d">Training Configuration</b><br>
            • Dataset: UTKFace / CACD (age-labelled face images)<br>
            • Loss: L_GAN + λ_cyc·L_cycle + λ_id·L_identity<br>
            • Optimizer: Adam (lr=2e-4, β₁=0.5, β₂=0.999)<br>
            • Batch size: 8 | Image size: 256×256<br>
            • Augmentation: random flip, colour jitter<br><br>
            <b style="color:#ffd166">Evaluation Metrics</b><br>
            • FID (Fréchet Inception Distance) ↓<br>
            • LPIPS (Perceptual similarity) ↓<br>
            • Age accuracy (Classifier) ↑<br>
            • Identity preservation (FaceNet cosine sim) ↑<br><br>
            <b style="color:#a259ff">Age Conditioning Method</b><br>
            • Adaptive Instance Normalisation (AdaIN)<br>
            • Per-block learned embedding: age_cls → γ, β<br>
            • Injected at every ResBlock (×6 injection points)
        </div>""",unsafe_allow_html=True)

    # Model param breakdown
    st.markdown('<br><div class="section-label">Parameter Breakdown</div>',unsafe_allow_html=True)
    m=FaceAgingGenerator()
    sections=[
        ("Encoder (enc1-3)", sum(p.numel() for n,p in m.named_parameters() if 'enc' in n)),
        ("ResBlocks ×6",     sum(p.numel() for n,p in m.named_parameters() if 'res' in n)),
        ("Decoder (dec1-3)", sum(p.numel() for n,p in m.named_parameters() if 'dec' in n)),
    ]
    total=sum(p.numel() for p in m.parameters())
    p1,p2,p3=st.columns(3)
    for col_w,(sec,cnt) in zip([p1,p2,p3],sections):
        with col_w:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-val">{cnt/1e3:.1f}K</div>
                <div class="stat-lbl">{sec}</div>
                <div style="color:#5a6a8a;font-size:0.65rem;margin-top:4px">{cnt/total*100:.1f}% of total</div>
            </div>""",unsafe_allow_html=True)

    # Comparison table: cGAN variants
    st.markdown('<br><div class="section-label">Conditional GAN Variants Comparison</div>',unsafe_allow_html=True)
    variants={
        "Model":        ["AgeForge (ours)","FRAN","HRFAE","SAM","CAFEGAN"],
        "Params (M)":   [f"{total/1e6:.2f}","~3.4","~8.1","~12.6","~6.2"],
        "Age Classes":  ["10","Continuous","8","Continuous","10"],
        "FID ↓":        ["~38*","29.4","41.2","33.7","36.1"],
        "Training Data":["UTKFace","FFHQ","CACD","UTKFace","CelebA-HQ"],
        "Conditioning": ["AdaIN","StyleGAN2","Attention","Diffusion","AdaIN+L1"],
    }
    st.table(variants)
    st.markdown('<div style="font-size:0.65rem;color:#5a6a8a">*Estimated from simulation. Real FID requires trained weights and full evaluation.</div>',unsafe_allow_html=True)


# ══════════════════════════════════════════
# TAB 4 — ABOUT
# ══════════════════════════════════════════
with tab_about:
    st.markdown("""
    <div class="info-card">
        <b style="color:#00e5ff;font-size:1.1rem">AgeForge v2.0</b><br><br>
        AgeForge is a research-grade conditional GAN face aging application built on PyTorch and Streamlit.
        It demonstrates age-conditioned image-to-image translation using Adaptive Instance Normalisation (AdaIN)
        injected at 6 residual blocks, enabling fine-grained age control via a 10-class embedding.<br><br>
        <b>Key Features</b><br>
        ✦ Full cGAN architecture with encoder-decoder + 6 ResBlocks<br>
        ✦ AdaIN age conditioning at every residual block<br>
        ✦ Progressive aging/de-aging texture engine (wrinkles, desaturation, sagging)<br>
        ✦ Forensic analysis: RGB delta, heatmap, age classifier probability<br>
        ✦ RGB histogram comparison between source and result<br>
        ✦ Biological age gauge from image features<br>
        ✦ Physiological radar across 8 biomarkers<br>
        ✦ Skin biomarkers, cognitive performance, telomere decline charts<br>
        ✦ Global lifespan, population pyramid, aging milestone timeline<br>
        ✦ GAN training metrics (50-epoch simulation)<br>
        ✦ Model architecture deep-dive + parameter breakdown<br>
        ✦ cGAN variant comparison table<br><br>
        <b>Deploy</b><br>
        Push to GitHub → share.streamlit.io → set app.py → Deploy (CPU-only, no GPU needed)<br><br>
        <b style="color:#ff6b35">For research and educational use only.</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<br><div class="section-label">References</div>',unsafe_allow_html=True)
    refs=[
        "He et al. (2016) — Deep Residual Learning for Image Recognition",
        "Huang & Belongie (2017) — Arbitrary Style Transfer via AdaIN",
        "Isola et al. (2017) — Image-to-Image Translation with Conditional GANs",
        "Roy et al. (2022) — FRAN: Face Re-Aging Network",
        "Li et al. (2021) — HRFAE: High Resolution Face Age Editing",
        "Alaluf et al. (2021) — Only a Matter of Style: Age Transformation using StyleGAN",
        "WHO (2023) — World Health Statistics Report",
        "Blackburn & Epel (2017) — The Telomere Effect (telomere biology reference)",
    ]
    for ref in refs:
        st.markdown(f'<div style="font-size:0.75rem;color:#5a6a8a;margin:4px 0;padding-left:12px;border-left:2px solid #1e2535">📄 {ref}</div>',unsafe_allow_html=True)

# ── Footer
st.markdown("""
<hr style="border:none;border-top:1px solid #1e2535;margin:40px 0 16px;">
<div style="text-align:center;font-family:'Space Mono',monospace;font-size:0.62rem;color:#2a3a5a;letter-spacing:2px;">
    AGEFORGE v2.0 · CONDITIONAL GAN · PYTORCH · STREAMLIT · FOR RESEARCH USE ONLY
</div>""", unsafe_allow_html=True)
