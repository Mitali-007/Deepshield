import streamlit as st
import numpy as np
import tempfile, os, sys, time
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.applications.efficientnet import preprocess_input
except Exception:
    import tensorflow as tf
    import keras
    from keras.applications.efficientnet import preprocess_input

# ── PAGE CONFIG ─────────────────────────────────────────
st.set_page_config(
    page_title="DeepShield — Deepfake Detector",
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.stApp { background: #080b12; color: #e8eaf0; }
.stApp::before {
    content: '';
    position: fixed; top:0; left:0; right:0; bottom:0;
    background-image:
        linear-gradient(rgba(0,245,160,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,245,160,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none; z-index: 0;
}
.hero-title {
    font-size: 3.5rem; font-weight: 800; letter-spacing: -2px;
    background: linear-gradient(135deg, #00f5a0 0%, #00d9f5 50%, #7b61ff 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.result-real {
    background: linear-gradient(135deg, #00f5a0, #00d9f5);
    color: #080b12; padding: 1.2rem 2rem; border-radius: 12px;
    text-align: center; font-size: 2rem; font-weight: 800;
    box-shadow: 0 0 40px rgba(0,245,160,0.4);
}
.result-fake {
    background: linear-gradient(135deg, #ff416c, #ff4b2b);
    color: white; padding: 1.2rem 2rem; border-radius: 12px;
    text-align: center; font-size: 2rem; font-weight: 800;
    box-shadow: 0 0 40px rgba(255,65,108,0.5);
}
.metric-tile {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px; padding: 1rem; text-align: center;
}
.metric-value { font-size: 1.8rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; }
.metric-label { font-size: 0.7rem; letter-spacing: 3px; text-transform: uppercase; color: #6b7280; }
.glass-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px; padding: 1.2rem; margin-bottom: 1rem;
}
[data-testid="stFileUploader"] {
    background: rgba(0,245,160,0.03) !important;
    border: 2px dashed rgba(0,245,160,0.25) !important;
    border-radius: 16px !important;
}
[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
.stButton > button {
    background: linear-gradient(135deg, #00f5a0, #00d9f5) !important;
    color: #080b12 !important; font-weight: 700 !important;
    border: none !important; border-radius: 8px !important;
}
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,245,160,0.3), transparent);
    margin: 1rem 0;
}
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── MODEL LOADING ────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load model once and cache it."""
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')

        model = keras.models.load_model("best_model_v2(1).keras")
        return model, None

    except Exception as e:
        return None, str(e)


def preprocess_face(img_array, img_size=224):
    img = cv2.resize(img_array, (img_size, img_size))
    img = img.astype(np.float32)
    img = preprocess_input(img)
    return np.expand_dims(img, axis=0)

def detect_face(img_array):
    try:
        from mtcnn import MTCNN
        detector = MTCNN()
        rgb      = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB) if img_array.shape[-1] == 3 else img_array
        results  = detector.detect_faces(rgb)
        if not results:
            return img_array, False
        best    = max(results, key=lambda x: x['confidence'])
        x, y, w, h = best['box']
        x, y    = max(0, x), max(0, y)
        pad     = int(0.2 * max(w, h))
        x1, y1  = max(0, x-pad), max(0, y-pad)
        x2, y2  = min(img_array.shape[1], x+w+pad), min(img_array.shape[0], y+h+pad)
        return rgb[y1:y2, x1:x2], True
    except Exception:
        return img_array, False


def predict(model, face):
    prob_fake = float(model.predict(preprocess_face(face), verbose=0)[0][0])
    prob_real = 1.0 - prob_fake
    return {
        "label":      "FAKE" if prob_fake >= 0.5 else "REAL",
        "confidence": round(max(prob_fake, prob_real) * 100, 1),
        "fake_prob":  round(prob_fake * 100, 1),
        "real_prob":  round(prob_real * 100, 1),
    }


def predict_video(model, video_path, n=15):
    cap    = cv2.VideoCapture(video_path)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    idxs   = np.linspace(0, total-1, n, dtype=int)
    frames = []
    bar    = st.progress(0)
    txt    = st.empty()

    for i, idx in enumerate(idxs):
        bar.progress((i+1)/len(idxs))
        txt.markdown(f'<p style="color:#00f5a0;font-family:JetBrains Mono;font-size:0.8rem;">⚡ Frame {i+1}/{len(idxs)}</p>', unsafe_allow_html=True)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret: continue
        face, found = detect_face(frame)
        if face.size == 0: continue
        r = predict(model, face)
        r["timestamp"] = round(idx/fps, 2)
        frames.append(r)

    cap.release(); bar.empty(); txt.empty()
    if not frames: return None

    avg_fake = np.mean([f["fake_prob"] for f in frames])
    avg_real = 100 - avg_fake
    return {
        "label":         "FAKE" if avg_fake >= 50 else "REAL",
        "avg_fake_prob": round(avg_fake, 1),
        "avg_real_prob": round(avg_real, 1),
        "confidence":    round(max(avg_fake, avg_real), 1),
        "frames":        frames,
    }


def gauge(value, title, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number={"suffix":"%","font":{"size":26,"color":color,"family":"JetBrains Mono"}},
        title={"text":title,"font":{"size":12,"color":"#6b7280"}},
        gauge={
            "axis":{"range":[0,100],"tickcolor":"#374151"},
            "bar":{"color":color,"thickness":0.25},
            "bgcolor":"rgba(0,0,0,0)", "borderwidth":0,
            "steps":[
                {"range":[0,50],"color":"rgba(0,245,160,0.05)"},
                {"range":[50,100],"color":"rgba(255,65,108,0.05)"}
            ],
        }
    ))
    fig.update_layout(
        height=190, margin=dict(l=15,r=15,t=35,b=5),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def timeline(frames):
    ts = [f["timestamp"] for f in frames]
    fp = [f["fake_prob"]  for f in frames]
    colors = ["#ff416c" if f["label"]=="FAKE" else "#00f5a0" for f in frames]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts, y=fp, fill="tozeroy",
        fillcolor="rgba(255,65,108,0.08)",
        line=dict(color="rgba(0,0,0,0)"), showlegend=False))
    fig.add_trace(go.Scatter(x=ts, y=fp, mode="lines+markers",
        line=dict(color="#ff416c",width=2),
        marker=dict(color=colors,size=8),
        hovertemplate="<b>%{x:.1f}s</b><br>Fake: %{y:.1f}%<extra></extra>"))
    fig.add_hline(y=50, line_dash="dot", line_color="rgba(255,255,255,0.2)",
        annotation_text="Decision Boundary",
        annotation_font=dict(color="rgba(255,255,255,0.3)",size=10))
    fig.update_layout(
        xaxis_title="Timestamp (s)", yaxis_title="Fake Probability (%)",
        yaxis_range=[0,100], height=280,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,17,23,0.8)",
        font={"family":"JetBrains Mono","color":"#6b7280","size":11},
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
        showlegend=False, margin=dict(l=10,r=10,t=10,b=10)
    )
    return fig


# ── SIDEBAR ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0;">
        <div style="font-size:2.5rem;">🛡️</div>
        <div style="font-family:Syne;font-weight:800;font-size:1.2rem;color:#e8eaf0;">DeepShield</div>
        <div style="font-family:'JetBrains Mono';font-size:0.65rem;color:#00f5a0;letter-spacing:3px;">v1.0 · FF++ C23</div>
    </div>
    <div class="divider"></div>
    """, unsafe_allow_html=True)

    mode = st.radio("Detection Mode", ["🖼️ Image", "🎥 Video"])
    n_frames = 15
    if "Video" in mode:
        n_frames = st.slider("Frames to Analyze", 5, 30, 15)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
        <p style="color:#6b7280;font-size:0.65rem;letter-spacing:3px;text-transform:uppercase;">Model Info</p>
        <div style="display:flex;justify-content:space-between;padding:0.3rem 0;">
            <span style="color:#9ca3af;font-size:0.8rem;">Architecture</span>
            <span style="color:#00f5a0;font-family:'JetBrains Mono';font-size:0.75rem;">EfficientNetB0</span>
        </div>
        <div style="display:flex;justify-content:space-between;padding:0.3rem 0;">
            <span style="color:#9ca3af;font-size:0.8rem;">Dataset</span>
            <span style="color:#00f5a0;font-family:'JetBrains Mono';font-size:0.75rem;">FF++ C23</span>
        </div>
        <div style="display:flex;justify-content:space-between;padding:0.3rem 0;">
            <span style="color:#9ca3af;font-size:0.8rem;">Accuracy</span>
            <span style="color:#00f5a0;font-family:'JetBrains Mono';font-size:0.75rem;">92%</span>
        </div>
        <div style="display:flex;justify-content:space-between;padding:0.3rem 0;">
            <span style="color:#9ca3af;font-size:0.8rem;">AUC Score</span>
            <span style="color:#00f5a0;font-family:'JetBrains Mono';font-size:0.75rem;">0.977</span>
        </div>
    </div>
    <p style="color:#374151;font-size:0.65rem;font-family:'JetBrains Mono';text-align:center;letter-spacing:1px;">
        Batch 24 · CSE AI-ML · VJIT 2026
    </p>
    """, unsafe_allow_html=True)


# ── HEADER ───────────────────────────────────────────────
st.markdown("""
<div style="padding:1.5rem 0 0.5rem 0;">
    <p style="color:#00f5a0;font-family:'JetBrains Mono';font-size:0.7rem;letter-spacing:4px;text-transform:uppercase;">
        ⬡ AI-Powered Forensic Analysis
    </p>
    <h1 class="hero-title">DeepShield</h1>
    <p style="color:#6b7280;font-size:0.9rem;">
        Detect AI-generated & manipulated media · EfficientNetB0 · FF++ C23 · 92% Accuracy
    </p>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

# Load model
model, err = load_model()

if err:
    st.markdown(f"""
    <div style="background:rgba(255,65,108,0.1);border:1px solid rgba(255,65,108,0.3);
                border-radius:12px;padding:1rem;">
        <p style="color:#ff416c;font-family:'JetBrains Mono';font-size:0.85rem;margin:0;">
            ⚠️ {err}
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

st.markdown("""
<div style="display:inline-flex;align-items:center;gap:0.5rem;
            background:rgba(0,245,160,0.08);border:1px solid rgba(0,245,160,0.2);
            border-radius:20px;padding:0.3rem 0.8rem;margin-bottom:1rem;">
    <span style="width:8px;height:8px;background:#00f5a0;border-radius:50%;display:inline-block;"></span>
    <span style="color:#00f5a0;font-family:'JetBrains Mono';font-size:0.75rem;letter-spacing:1px;">
        MODEL LOADED · READY
    </span>
</div>
""", unsafe_allow_html=True)


# ── IMAGE MODE ───────────────────────────────────────────
if "Image" in mode:
    uploaded = st.file_uploader("Upload a face image", type=["jpg","jpeg","png"])

    if uploaded:
        image     = Image.open(uploaded).convert("RGB")
        img_array = np.array(image)

        col1, col2 = st.columns([1, 1.4], gap="large")

        with col1:
            st.markdown('<p style="color:#6b7280;font-size:0.7rem;letter-spacing:3px;">INPUT IMAGE</p>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            with st.spinner("Detecting face..."):
                face, found = detect_face(img_array)
            if found:
                st.markdown('<p style="color:#6b7280;font-size:0.7rem;letter-spacing:3px;margin-top:0.8rem;">DETECTED FACE</p>', unsafe_allow_html=True)
                st.image(Image.fromarray(face), width=160)
                st.success("✓ Face detected")
            else:
                st.warning("⚠ No face found — using full image")

        with col2:
            with st.spinner("Analyzing..."):
                time.sleep(0.2)
                result = predict(model, face)

            css   = "result-real" if result["label"] == "REAL" else "result-fake"
            emoji = "✓ AUTHENTIC" if result["label"] == "REAL" else "⚠ MANIPULATED"
            st.markdown(f'<div class="{css}">{emoji}</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            color = "#00f5a0" if result["label"] == "REAL" else "#ff416c"
            for col, val, lbl, clr in zip(
                [c1, c2, c3],
                [result["confidence"], result["real_prob"], result["fake_prob"]],
                ["Confidence", "Real Prob", "Fake Prob"],
                [color, "#00f5a0", "#ff416c"]
            ):
                with col:
                    st.markdown(f"""
                    <div class="metric-tile">
                        <div class="metric-value" style="color:{clr}">{val}%</div>
                        <div class="metric-label">{lbl}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            g1, g2 = st.columns(2)
            with g1:
                st.plotly_chart(gauge(result["real_prob"], "REAL", "#00f5a0"),
                    use_container_width=True, config={"displayModeBar":False})
            with g2:
                st.plotly_chart(gauge(result["fake_prob"], "FAKE", "#ff416c"),
                    use_container_width=True, config={"displayModeBar":False})

            # Probability bar
            st.markdown(f"""
            <div style="margin-top:0.5rem;">
                <div style="display:flex;justify-content:space-between;margin-bottom:0.3rem;">
                    <span style="color:#00f5a0;font-family:'JetBrains Mono';font-size:0.75rem;">REAL {result['real_prob']}%</span>
                    <span style="color:#ff416c;font-family:'JetBrains Mono';font-size:0.75rem;">FAKE {result['fake_prob']}%</span>
                </div>
                <div style="height:8px;background:rgba(255,255,255,0.05);border-radius:4px;overflow:hidden;">
                    <div style="height:100%;width:{result['real_prob']}%;
                                background:linear-gradient(90deg,#00f5a0,#00d9f5);
                                border-radius:4px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("⚠️ Interpretation Guide"):
                st.write("This system gives a **probability score**, not absolute truth. Scores above 85% confidence are considered high confidence. Always apply human judgment.")


# ── VIDEO MODE ───────────────────────────────────────────
elif "Video" in mode:
    uploaded_video = st.file_uploader("Upload a video", type=["mp4","avi","mov"])

    if uploaded_video:
        col1, col2 = st.columns([1, 1.4], gap="large")

        with col1:
            st.markdown(
                '<p style="color:#6b7280;font-size:0.7rem;letter-spacing:3px;">INPUT VIDEO</p>',
                unsafe_allow_html=True
            )
            st.video(uploaded_video)

        with col2:
            # Save uploaded video temporarily
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_video.read())
            tfile.close()

            # Analyze video
            result = predict_video(model, tfile.name, n=n_frames)

            # Remove temp file safely
            try:
                os.remove(tfile.name)
            except:
                pass

            # Display result
            if not result:
                st.error("No faces detected in video.")

            else:
                css = "result-real" if result["label"] == "REAL" else "result-fake"
                emoji = "✓ AUTHENTIC" if result["label"] == "REAL" else "⚠ MANIPULATED"

                st.markdown(
                    f'<div class="{css}">{emoji}</div>',
                unsafe_allow_html=True
            )

            st.markdown("<br>", unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)

            color = "#00f5a0" if result["label"] == "REAL" else "#ff416c"

            data = [
                (result["confidence"], "Confidence", color),
                (result["avg_real_prob"], "Real Prob", "#00f5a0"),
                (result["avg_fake_prob"], "Fake Prob", "#ff416c"),
                (len(result["frames"]), "Frames", "#7b61ff"),
            ]

            for col, (val, lbl, clr) in zip([m1, m2, m3, m4], data):
                with col:
                    st.markdown(f"""
                    <div class="metric-tile">
                        <div class="metric-value" style="color:{clr}">
                            {val:.0f}{'%' if lbl != 'Frames' else ''}
                        </div>
                        <div class="metric-label">{lbl}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Timeline graph
            if result["frames"]:
                st.markdown(
                    '<br><p style="color:#6b7280;font-size:0.7rem;letter-spacing:3px;">FRAME TIMELINE</p>',
                    unsafe_allow_html=True
                )

                st.plotly_chart(
                    timeline(result["frames"]),
                    use_container_width=True,
                    config={"displayModeBar": False}
                )

# ── FOOTER ───────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;color:#374151;font-family:'JetBrains Mono';
            font-size:0.65rem;letter-spacing:2px;margin-top:3rem;
            padding-top:1rem;border-top:1px solid rgba(255,255,255,0.05);">
    DEEPSHIELD · BATCH 14 · CSE AI-ML · VJIT 2026
</div>
""", unsafe_allow_html=True)