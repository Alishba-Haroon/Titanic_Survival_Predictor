"""Titanic Survival Predictor - Streamlit demo (with 3D ocean background).

Run with:  streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

BASE = Path(__file__).parent
MODEL_PATH = BASE / "models" / "titanic_model.joblib"
METRICS_PATH = BASE / "models" / "metrics.json"

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

# ---------------------------------------------------------------------------
# 3D BACKGROUND  (three.js: moonlit night sky, animated ocean, icebergs, ship)
# ---------------------------------------------------------------------------
SCENE_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>html,body{margin:0;height:100%;overflow:hidden;background:#050b18}canvas{display:block}</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
</head>
<body>
<script>
// Turn this iframe into a fixed full-screen layer behind the Streamlit app
try {
  const f = window.frameElement;
  if (f) {
    Object.assign(f.style, {position:'fixed', top:'0', left:'0', width:'100vw',
      height:'100vh', zIndex:'-1', border:'0', pointerEvents:'none'});
    if (f.parentElement) { f.parentElement.style.height = '0'; f.parentElement.style.margin = '0'; }
  }
} catch (e) {}

const renderer = new THREE.WebGLRenderer({antialias: true});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x0b1d33, 0.0055);
const camera = new THREE.PerspectiveCamera(58, innerWidth / innerHeight, 0.1, 2500);
camera.position.set(0, 10, 70);

// ---- Sky (vertical gradient dome) ----
const skyGeo = new THREE.SphereGeometry(1200, 32, 16);
const pos = skyGeo.attributes.position, cols = [];
const cHorizon = new THREE.Color(0x244a73), cZenith = new THREE.Color(0x02050e);
for (let i = 0; i < pos.count; i++) {
  const h = Math.max(0, pos.getY(i) / 1200);
  const c = cHorizon.clone().lerp(cZenith, Math.pow(h, 0.55));
  cols.push(c.r, c.g, c.b);
}
skyGeo.setAttribute('color', new THREE.Float32BufferAttribute(cols, 3));
scene.add(new THREE.Mesh(skyGeo, new THREE.MeshBasicMaterial({vertexColors: true, side: THREE.BackSide, fog: false})));

// ---- Stars ----
const sp = [];
for (let i = 0; i < 1800; i++) {
  const th = Math.random() * Math.PI * 2, ph = Math.acos(Math.random() * 0.95 + 0.05);
  sp.push(1100 * Math.sin(ph) * Math.cos(th), 1100 * Math.cos(ph), 1100 * Math.sin(ph) * Math.sin(th));
}
const starGeo = new THREE.BufferGeometry();
starGeo.setAttribute('position', new THREE.Float32BufferAttribute(sp, 3));
const stars = new THREE.Points(starGeo, new THREE.PointsMaterial({color: 0xdfe9ff, size: 1.5, sizeAttenuation: false, fog: false, transparent: true, opacity: 0.85}));
scene.add(stars);

// ---- Moon (glow sprite) ----
const mc = document.createElement('canvas'); mc.width = mc.height = 256;
const mx = mc.getContext('2d');
const g = mx.createRadialGradient(128, 128, 0, 128, 128, 128);
g.addColorStop(0, 'rgba(255,255,245,1)'); g.addColorStop(0.16, 'rgba(240,244,255,0.95)');
g.addColorStop(0.22, 'rgba(180,205,255,0.35)'); g.addColorStop(1, 'rgba(120,160,255,0)');
mx.fillStyle = g; mx.fillRect(0, 0, 256, 256);
const moon = new THREE.Sprite(new THREE.SpriteMaterial({map: new THREE.CanvasTexture(mc), fog: false, transparent: true, depthWrite: false}));
moon.position.set(-260, 210, -900); moon.scale.set(380, 380, 1);
scene.add(moon);

// ---- Lights ----
const moonLight = new THREE.DirectionalLight(0xbcd4ff, 1.25);
moonLight.position.set(-260, 210, -900);
scene.add(moonLight);
scene.add(new THREE.HemisphereLight(0x3a5a8a, 0x05101f, 0.55));

// ---- Ocean ----
const SEG = 150, SIZE = 1400;
const oceanGeo = new THREE.PlaneGeometry(SIZE, SIZE, SEG, SEG);
const ocean = new THREE.Mesh(oceanGeo, new THREE.MeshStandardMaterial({color: 0x07284a, roughness: 0.22, metalness: 0.65}));
ocean.rotation.x = -Math.PI / 2;
scene.add(ocean);
const oPos = oceanGeo.attributes.position;
const baseXY = [];
for (let i = 0; i < oPos.count; i++) baseXY.push(oPos.getX(i), oPos.getY(i));

function waveAt(x, y, t) {
  return Math.sin(x * 0.045 + t * 0.9) * 1.1 + Math.sin(y * 0.07 + t * 1.2) * 0.8 +
         Math.sin((x + y) * 0.028 + t * 0.6) * 1.6 + Math.sin((x - y) * 0.11 + t * 1.7) * 0.35;
}

// ---- Icebergs ----
function makeIceberg(r, x, z, seed) {
  const geo = new THREE.IcosahedronGeometry(r, 2);
  const p = geo.attributes.position;
  for (let i = 0; i < p.count; i++) {
    const vx = p.getX(i), vy = p.getY(i), vz = p.getZ(i);
    const n = 1 + 0.28 * Math.sin(vx * 0.35 + seed) * Math.cos(vz * 0.31 + seed * 1.7) + 0.16 * Math.sin(vy * 0.6 + seed * 2.3);
    p.setXYZ(i, vx * n, vy * n * 1.25, vz * n);
  }
  geo.computeVertexNormals();
  const m = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({color: 0xd6e9f7, roughness: 0.55, metalness: 0.05, flatShading: true, emissive: 0x0b1c2e}));
  m.position.set(x, r * 0.15, z);
  scene.add(m);
  return m;
}
const bergs = [makeIceberg(15, -75, -150, 1.3), makeIceberg(9, 95, -230, 4.1), makeIceberg(22, -190, -330, 7.7)];

// ---- Ship silhouette ----
const ship = new THREE.Group();
const dark = new THREE.MeshStandardMaterial({color: 0x0a0e15, roughness: 0.8});
const hull = new THREE.Mesh(new THREE.BoxGeometry(62, 6, 9), dark); hull.position.y = 3; ship.add(hull);
const bow = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 4.5, 8, 4), dark);
bow.rotation.z = -Math.PI / 2; bow.rotation.x = Math.PI / 4; bow.position.set(35, 3, 0); ship.add(bow);
const deck = new THREE.Mesh(new THREE.BoxGeometry(44, 4, 7), new THREE.MeshStandardMaterial({color: 0x161c27, roughness: 0.7})); deck.position.y = 8; ship.add(deck);
const funnelMat = new THREE.MeshStandardMaterial({color: 0x9a6a33, roughness: 0.6});
const topMat = new THREE.MeshStandardMaterial({color: 0x050505});
for (let i = 0; i < 4; i++) {
  const fx = -12 + i * 8;
  const fu = new THREE.Mesh(new THREE.CylinderGeometry(1.7, 1.9, 9, 16), funnelMat); fu.position.set(fx, 15, 0); ship.add(fu);
  const tp = new THREE.Mesh(new THREE.CylinderGeometry(1.75, 1.75, 1.6, 16), topMat); tp.position.set(fx, 19.8, 0); ship.add(tp);
}
const wp = [];
for (let i = 0; i < 44; i++) { wp.push(-30 + i * 1.4, 4.6 + (i % 2) * 0.1, 4.55); wp.push(-20 + i * 0.9, 8.2, 3.6); }
const winGeo = new THREE.BufferGeometry(); winGeo.setAttribute('position', new THREE.Float32BufferAttribute(wp, 3));
ship.add(new THREE.Points(winGeo, new THREE.PointsMaterial({color: 0xffd58a, size: 2.4, sizeAttenuation: false, fog: false})));
ship.position.set(65, 0, -135); ship.rotation.y = -0.25;
scene.add(ship);

// ---- Subtle mouse parallax (listens on the parent page) ----
const mouse = {x: 0, y: 0};
try {
  window.parent.document.addEventListener('mousemove', e => {
    mouse.x = (e.clientX / window.parent.innerWidth - 0.5) * 2;
    mouse.y = (e.clientY / window.parent.innerHeight - 0.5) * 2;
  });
} catch (e) {}

addEventListener('resize', () => {
  renderer.setSize(innerWidth, innerHeight);
  camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
});

const clock = new THREE.Clock();
function animate() {
  requestAnimationFrame(animate);
  const t = clock.getElapsedTime();

  for (let i = 0; i < oPos.count; i++) oPos.setZ(i, waveAt(baseXY[i * 2], baseXY[i * 2 + 1], t));
  oPos.needsUpdate = true; oceanGeo.computeVertexNormals();

  bergs.forEach((b, i) => { b.position.y = 2 + Math.sin(t * 0.5 + i) * 0.5; b.rotation.y += 0.0005; });
  ship.position.y = 1.5 + Math.sin(t * 0.9) * 0.45;
  ship.rotation.z = Math.sin(t * 0.7) * 0.018;
  stars.rotation.y = t * 0.002;

  camera.position.x += ((Math.sin(t * 0.08) * 8 + mouse.x * 10) - camera.position.x) * 0.03;
  camera.position.y += ((10 - mouse.y * 3) - camera.position.y) * 0.03;
  camera.lookAt(0, 14, -120);
  renderer.render(scene, camera);
}
animate();
</script>
</body>
</html>
"""

components.html(SCENE_HTML, height=0)

# ---------------------------------------------------------------------------
# GLASS-STYLE UI  (keeps the form readable on top of the 3D scene)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    html, body { background: #050b18; }
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stHeader"] { background: transparent !important; }

    .block-container {
        background: rgba(7, 15, 30, 0.66);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 18px;
        padding: 2.2rem 2.4rem 2.6rem !important;
        margin-top: 2rem;
        box-shadow: 0 24px 70px rgba(0, 0, 0, 0.55);
    }
    [data-testid="stSidebar"] {
        background: rgba(6, 12, 26, 0.75) !important;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border-right: 1px solid rgba(255, 255, 255, 0.10);
    }
    h1, h2, h3, label, p,
    [data-testid="stMarkdownContainer"],
    [data-testid="stWidgetLabel"] p,
    [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"],
    [data-testid="stCaptionContainer"] { color: #eaf2ff !important; }
    h1 { text-shadow: 0 2px 24px rgba(120, 170, 255, 0.35); }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-testid="stNumberInputContainer"] {
        background: rgba(255, 255, 255, 0.08) !important;
        border-color: rgba(255, 255, 255, 0.18) !important;
    }
    input { color: #ffffff !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# APP LOGIC (unchanged)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    return json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}


st.title("🚢 Titanic Survival Predictor")
st.write("Enter a passenger's details and the trained machine-learning model will estimate "
         "the chance that they would have survived the Titanic disaster.")

if not MODEL_PATH.exists():
    st.error("Model file not found. Run `notebooks/Titanic_Capstone.ipynb` first to create "
             "`models/titanic_model.joblib`.")
    st.stop()

model = load_model()
metrics = load_metrics()

with st.sidebar:
    st.header("About the model")
    if metrics:
        st.write(f"**Algorithm:** {metrics['model']}")
        st.metric("Test accuracy", f"{metrics['test_accuracy']:.1%}")
        st.metric("ROC-AUC", f"{metrics['test_roc_auc']:.2f}")
    st.caption("Trained on 891 passengers from the Kaggle Titanic dataset.")

col1, col2 = st.columns(2)
with col1:
    pclass = st.selectbox("Ticket class", [1, 2, 3], index=2,
                          format_func=lambda x: {1: "1st (upper)", 2: "2nd (middle)", 3: "3rd (lower)"}[x])
    sex = st.radio("Sex", ["male", "female"], horizontal=True)
    age = st.slider("Age", 0, 80, 28)
    title = st.selectbox("Title", ["Mr", "Mrs", "Miss", "Master", "Rare"],
                         help="Master = young boy. Rare = Dr, Rev, Col, Countess, etc.")
with col2:
    sibsp = st.number_input("Siblings / spouses aboard", 0, 8, 0)
    parch = st.number_input("Parents / children aboard", 0, 6, 0)
    fare = st.number_input("Ticket fare (£)", 0.0, 520.0, 15.0, step=1.0)
    embarked = st.selectbox("Port of embarkation", ["S", "C", "Q"],
                            format_func=lambda x: {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}[x])

if st.button("Predict survival", type="primary", use_container_width=True):
    family_size = int(sibsp + parch + 1)
    passenger = pd.DataFrame([{
        "Pclass": pclass, "Age": age, "Fare": fare,
        "FamilySize": family_size, "IsAlone": int(family_size == 1),
        "Sex": sex, "Embarked": embarked, "Title": title,
    }])

    prob = float(model.predict_proba(passenger)[0, 1])
    st.divider()
    if prob >= 0.5:
        st.success(f"✅ Likely to SURVIVE — estimated probability {prob:.0%}")
    else:
        st.error(f"❌ Unlikely to survive — estimated probability {prob:.0%}")
    st.progress(prob)
    st.caption("This is a statistical estimate based on historical data, not a certainty.")