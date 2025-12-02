# app.py
# CASPER v3 — High-Speed UAV AO Governance Console
#
# Deterministic, human-gated autonomy demo.
# All telemetry is synthetic and non-operational.
# No weapon control, no targeting, no kinetic decision logic.

import time
import math
import json
import io
import hashlib
from datetime import datetime
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Page Config
# ------------------------------------------------------------
st.set_page_config(
page_title="CASPER v3 — High-Speed UAV Governance Console",

layout="wide",
)

# ------------------------------------------------------------
# Constants & Presets
# ------------------------------------------------------------

DT_SECONDS = 1.0
MAX_HISTORY = 600

# High-speed UAV envelope presets (no hypersonics)
ENVELOPE_PRESETS: Dict[str, Dict[str, float | str]] = {
"Conservative Test Profile": {
"max_mach": 1.2,
"max_q_kpa": 450.0,
"max_g": 3.5,
"max_thermal_index": 0.65,
"max_latency_ms": 250.0,
"description": "Tight margins for early high-speed envelope expansion flights.",
},
"Nominal Demo Flight": {
"max_mach": 1.8,
"max_q_kpa": 650.0,
"max_g": 4.5,
"max_thermal_index": 0.78,
"max_latency_ms": 300.0,

"description": "Balanced high-speed demonstration profile with headroom.",
},
"Aggressive Envelope Probe": {
"max_mach": 2.3,
"max_q_kpa": 800.0,
"max_g": 5.5,
"max_thermal_index": 0.90,
"max_latency_ms": 350.0,
"description": "Pushes up against high-speed envelope limits for test-only runs.",
},
}

# Flight environment profiles (link + sensor stress)
ENVIRONMENTS: Dict[str, Dict[str, float]] = {
"Clear Skies / Clean Link": {
"latency_base": 120.0,
"latency_jitter": 40.0,
"thermal_bias": 0.0,
"imu_drift_bias": 0.0,
},
"High Latency Link": {
"latency_base": 260.0,
"latency_jitter": 80.0,
"thermal_bias": 0.05,
"imu_drift_bias": 0.02,
},

"Thermal Stress Test": {
"latency_base": 180.0,
"latency_jitter": 60.0,
"thermal_bias": 0.18,
"imu_drift_bias": 0.01,
},
"Sensor Degradation": {
"latency_base": 220.0,
"latency_jitter": 90.0,
"thermal_bias": 0.10,
"imu_drift_bias": 0.04,
},
}

# AO-level governance presets (Casper-style)
THRESHOLD_PRESETS = {
"Defensive": {
"clarity_threshold": 85.0,
"threat_threshold": 40.0,
"description": "High caution, minimal risk tolerance.",
},
"Balanced": {
"clarity_threshold": 75.0,
"threat_threshold": 65.0,
"description": "Standard operational envelope.",
},

"Aggressive": {
"clarity_threshold": 65.0,
"threat_threshold": 80.0,
"description": "Accept higher risk for mission objectives.",
},
}

# Mission stages (shared flight + AO phases)
MISSION_STAGES = [
{
"code": "STAGE_1_BOOST",
"label": "Boost & Form Up",
"duration": 40,
"description": "Takeoff, climb, and formation lock-in.",
},
{
"code": "STAGE_2_GRID_SEARCH",
"label": "Grid Search",
"duration": 70,
"description": "Structured search pattern over synthetic terrain.",
},
{
"code": "STAGE_3_MARK_RELAY",
"label": "Mark & Relay",
"duration": 70,
"description": "Signal marking, relay behavior, comms load rising.",

},
{
"code": "STAGE_4_COLLAPSE",
"label": "Collapse & Regroup",
"duration": 50,
"description": "Perimeter collapse, re-cohesion, path to RTB.",
},
{
"code": "STAGE_5_RTB",
"label": "Return to Base",
"duration": 9999,
"description": "Returning to home corridor.",
},
]

# ------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------

def clamp(v: float, lo: float, hi: float) -> float:
return max(lo, min(hi, v))

def sha256_hash(data: Dict[str, Any]) -> str:
s = json.dumps(data, sort_keys=True)
return hashlib.sha256(s.encode("utf-8")).hexdigest()

# ------------------------------------------------------------
# Deterministic RNG helpers
# ------------------------------------------------------------

def rng_for_tick(tick: int) -> np.random.Generator:
"""
Deterministic per-tick RNG.
Same ss.rng_seed + tick => same Generator sequence.
"""
ss = st.session_state
base = ss.rng_seed
return np.random.default_rng(base + tick)

def tick_rng() -> np.random.Generator:
"""
Access the per-tick generator (set in the simulation step).
"""
ss = st.session_state
if "current_rng" not in ss:
ss.current_rng = rng_for_tick(ss.tick)
return ss.current_rng

# ------------------------------------------------------------
# Session State Init
# ------------------------------------------------------------

def init_state():
ss = st.session_state
ss.tick = 0
ss.mission_time = 0.0
ss.running = False
ss.last_update = time.time()
ss.history: List[Dict[str, Any]] = []
ss.events: List[Dict[str, Any]] = []
ss.proposals: List[Dict[str, Any]] = []
ss.audit_log: List[Dict[str, Any]] = []
ss.prev_hash = "0" * 64
ss.gate_open = False
ss.preset = "Nominal Demo Flight"
ss.environment = "Clear Skies / Clean Link"
ss.threshold_preset = "Balanced"
ss.cam_frame_ix = 0
ss.mission_stage_index = 0
ss.mission_stage_tick = 0
ss.rple_insights: List[Dict[str, Any]] = []
ss.rple_memory: List[str] = []

# Deterministic run metadata

ss.run_id = int(time.time() * 1000)
ss.rng_seed = ss.run_id % (2**32)
ss.current_rng = rng_for_tick(0)

# Terrain for synthetic vision (fixed, non-tick RNG)
if "terrain" not in ss:
rng = np.random.RandomState(42)
base = rng.normal(0.0, 1.0, size=(80, 80))
for i in range(0, 80, 8):
base[i:i+1, :] += 1.5
base[:, i:i+1] += 1.5
ss.terrain = base

# Swarm agents
ss.swarm_agents: List[Dict[str, Any]] = []

# HyperClarity engine
class HyperClarity:
def __init__(self):
self.ema_clarity = 0.9
self.alpha = 0.15

def compute(self, raw_signal: float, noise: float) -> float:
raw_signal_clamped = np.clip(raw_signal, 0, 1)
noise_clamped = np.clip(noise, 0, 1)
clarity = raw_signal_clamped * (1 - 0.4 * noise_clamped)

clarity = max(clarity, 0.60)
clarity = clarity * 0.92 + 0.08
self.ema_clarity = (
self.alpha * clarity + (1 - self.alpha) * self.ema_clarity
)
return round(float(self.ema_clarity * 100), 2)

ss.clarity_engine = HyperClarity()

if "tick" not in st.session_state:
init_state()

ss = st.session_state

# ------------------------------------------------------------
# AO Configuration Sidebar
# ------------------------------------------------------------

st.sidebar.markdown("### AO Configuration")

use_custom_ao = st.sidebar.checkbox(
"Use Custom AO",
value=False,
help="Enable a custom Area of Operations instead of presets.",
)

if use_custom_ao:
AO_LABEL = st.sidebar.text_input("AO Label", value="Custom AO")
BASE_LAT = st.sidebar.number_input(
"Center Latitude", value=49.9935, format="%.4f"
)
BASE_LON = st.sidebar.number_input(
"Center Longitude", value=36.2304, format="%.4f"
)
LAT_DELTA = st.sidebar.number_input(
"Lat Range (±)", value=0.08, format="%.3f"
)
LON_DELTA = st.sidebar.number_input(
"Lon Range (±)", value=0.12, format="%.3f"
)
else:
AO_MODE = st.sidebar.selectbox(
"AO Mode (Synthetic)",
["Kharkiv (synthetic)", "Generic AO", "Training Range Sandbox"],
)
if AO_MODE == "Kharkiv (synthetic)":
AO_LABEL = "Kharkiv Region"
BASE_LAT, BASE_LON = 49.9935, 36.2304
LAT_DELTA, LON_DELTA = 0.08, 0.12
elif AO_MODE == "Generic AO":
AO_LABEL = "Generic Contested AO"

BASE_LAT, BASE_LON = 35.0, 40.0
LAT_DELTA, LON_DELTA = 0.15, 0.25
else:
AO_LABEL = "Training Range Sandbox"
BASE_LAT, BASE_LON = 39.0, -104.8
LAT_DELTA, LON_DELTA = 0.10, 0.20

# ------------------------------------------------------------
# Sidebar Presets & Environment
# ------------------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.markdown("### Flight Envelope Preset")
preset_names = list(ENVELOPE_PRESETS.keys())
preset_choice = st.sidebar.radio(
"Select flight envelope",
options=preset_names,
index=preset_names.index(ss.preset),
)
ss.preset = preset_choice
st.sidebar.info(
f"**{ss.preset}** — {ENVELOPE_PRESETS[ss.preset]['description']}"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Environment Profile")

env_names = list(ENVIRONMENTS.keys())
env_choice = st.sidebar.radio(
"Select environment",
options=env_names,
index=env_names.index(ss.environment),
)
ss.environment = env_choice

st.sidebar.markdown("---")
st.sidebar.markdown("### Tactical Threshold Preset")

tp_col1, tp_col2, tp_col3 = st.sidebar.columns(3)
with tp_col1:
if st.button("Defensive", use_container_width=True):
ss.threshold_preset = "Defensive"
with tp_col2:
if st.button("Balanced", use_container_width=True):
ss.threshold_preset = "Balanced"
with tp_col3:
if st.button("Aggressive", use_container_width=True):
ss.threshold_preset = "Aggressive"

current_threshold = THRESHOLD_PRESETS[ss.threshold_preset]
st.sidebar.info(
f"**{ss.threshold_preset}**: {current_threshold['description']}"
)

st.sidebar.markdown("---")
st.sidebar.caption(
"All telemetry is synthetic. Governance & safety oversight demo only."
)

# ------------------------------------------------------------
# Governance Banner
# ------------------------------------------------------------

st.info(
"""
### GOVERNANCE-ONLY AUTONOMY DEMO
**No weapon control · No fire authority · No targeting logic · No kinetic decision loops**
This console models ISR / corridor management, AO clarity, and human-gated safety
proposals only.
"""
)

# ------------------------------------------------------------
# HyperClarity / State Classification
# ------------------------------------------------------------

def get_current_mission_stage() -> Dict[str, Any]:
return MISSION_STAGES[ss.mission_stage_index]

def advance_mission_stage():
stage = get_current_mission_stage()
if ss.mission_stage_tick >= stage["duration"]:
if ss.mission_stage_index < len(MISSION_STAGES) - 1:
ss.mission_stage_index += 1
ss.mission_stage_tick = 0
new_stage = get_current_mission_stage()
log_event("INFO", f"Mission stage → {new_stage['label']}")
record_audit_event(
"mission_stage_changed",
{"code": new_stage["code"], "label": new_stage["label"]},
)
else:
ss.mission_stage_tick = stage["duration"]

def compute_clarity_and_risk(row: Dict[str, Any]) -> Dict[str, Any]:
"""
Unified clarity/risk engine that blends:
- Flight envelope pressure (Mach, q, g, thermal, latency, IMU)
- AO conditions (threat index, civilian density, nav drift, comms loss)
"""
preset = ENVELOPE_PRESETS[ss.preset]

mach = row["mach"]
q_kpa = row["q_kpa"]

g_load = row["g_load"]
thermal = row["thermal_index"]
latency = row["link_latency_ms"]
imu_drift = row["imu_drift_deg_s"]
threat = row["threat_index"]
civ = row["civ_density"]
nav = row["nav_drift"]
comms = row["comms_loss"]

mach_util = clamp(mach / preset["max_mach"], 0.0, 1.6)
q_util = clamp(q_kpa / preset["max_q_kpa"], 0.0, 1.6)
g_util = clamp(abs(g_load) / preset["max_g"], 0.0, 1.8)
thermal_util = clamp(thermal / preset["max_thermal_index"], 0.0, 1.8)
latency_util = clamp(latency / preset["max_latency_ms"], 0.0, 2.0)
imu_util = clamp(imu_drift / 0.12, 0.0, 2.0)

threat_util = clamp(threat / 100.0, 0.0, 1.2)
civ_util = clamp(civ / 1.0, 0.0, 1.0)
nav_util = clamp(nav / 30.0, 0.0, 1.0)
comms_util = clamp(comms / 1.0, 0.0, 1.0)

envelope_pressure = (
0.22 * q_util
+ 0.18 * thermal_util
+ 0.14 * mach_util
+ 0.10 * g_util

+ 0.06 * latency_util
+ 0.05 * imu_util
+ 0.10 * threat_util
+ 0.08 * civ_util
+ 0.04 * nav_util
+ 0.03 * comms_util
)

stage_code = row["mission_stage"]
stage_signal_factor = {
"STAGE_1_BOOST": 0.95,
"STAGE_2_GRID_SEARCH": 0.90,
"STAGE_3_MARK_RELAY": 0.85,
"STAGE_4_COLLAPSE": 0.80,
"STAGE_5_RTB": 0.92,
}.get(stage_code, 0.90)
stage_noise_floor = {
"STAGE_1_BOOST": 0.05,
"STAGE_2_GRID_SEARCH": 0.10,
"STAGE_3_MARK_RELAY": 0.15,
"STAGE_4_COLLAPSE": 0.20,
"STAGE_5_RTB": 0.08,
}.get(stage_code, 0.10)

signal_degradation = 0.40 * nav_util + 0.40 * comms_util + 0.20 * civ_util
raw_signal = stage_signal_factor * (1.0 - signal_degradation)

environmental_noise = (
0.55 * threat_util + 0.25 * civ_util + 0.20 * nav_util
)
noise = stage_noise_floor + (1.0 - stage_noise_floor) * environmental_noise

clarity = ss.clarity_engine.compute(raw_signal, noise)

risk = clamp(
(envelope_pressure * 60.0) + (100.0 - clarity) * 0.40,
0.0,
100.0,
)

predicted_risk = clamp(
risk
+ 7.0 * (q_util - 0.8)
+ 6.0 * (thermal_util - 0.85)
+ 5.0 * (threat_util - 0.7)
+ 4.0 * (comms_util - 0.6),
0.0,
100.0,
)

if clarity >= 90 and threat < 25 and risk < 30:
state = "STABLE"

elif clarity >= 80 and risk < 50:
state = "TENSE"
elif clarity >= 65 and risk < 75:
state = "HIGH_RISK"
else:
state = "CRITICAL"

return {
"clarity": round(clarity, 1),
"risk": round(risk, 1),
"predicted_risk": round(predicted_risk, 1),
"state": state,
"envelope_pressure": round(envelope_pressure, 3),
}

# ------------------------------------------------------------
# Audit & Event logging
# ------------------------------------------------------------

def record_audit_event(event_type: str, payload: Dict[str, Any]):
prev_hash = ss.audit_log[-1]["hash"] if ss.audit_log else "0" * 64
entry = {
"tick": ss.tick,
"timestamp": datetime.utcnow().isoformat() + "Z",
"event_type": event_type,
"payload": payload,

"prev_hash": prev_hash,
}
entry["hash"] = sha256_hash(entry)
ss.audit_log.append(entry)
ss.audit_log = ss.audit_log[-120:]

def log_event(level: str, msg: str, extras: Dict[str, Any] | None = None):
payload = {
"timestamp": datetime.utcnow().strftime("%H:%M:%S"),
"level": level,
"msg": msg,
"ao_label": AO_LABEL,
}
if extras:
payload.update(extras)
ss.events.append(payload)
ss.events = ss.events[-220:]

# ------------------------------------------------------------
# Swarm helpers
# ------------------------------------------------------------

def init_swarm_agents(n: int = 9):
rng = tick_rng()
agents = []

for i in range(n):
angle = 2 * math.pi * i / n
agents.append(
{
"id": f"A{i+1:02d}",
"lat": BASE_LAT + (0.02 * math.cos(angle)),
"lon": BASE_LON + (0.03 * math.sin(angle)),
"heading": float(rng.uniform(260, 280)),
"role": rng.choice(["SCOUT", "RELAY", "MARKER"]),
}
)
ss.swarm_agents = agents
record_audit_event("swarm_initialized", {"num_agents": n})

def move_swarm_agents():
if not ss.swarm_agents:
return

rng = tick_rng()
stage_code = get_current_mission_stage()["code"]
center_lat, center_lon = BASE_LAT, BASE_LON

for agent in ss.swarm_agents:
if stage_code == "STAGE_1_BOOST":
agent["heading"] += float(rng.uniform(-1.2, 1.2))

speed_lat, speed_lon = 0.0004, 0.0007
elif stage_code == "STAGE_2_GRID_SEARCH":
agent["heading"] += float(rng.uniform(-2.0, 2.0))
speed_lat, speed_lon = 0.0006, 0.0010
elif stage_code == "STAGE_3_MARK_RELAY":
agent["heading"] += float(rng.uniform(-3.0, 3.0))
speed_lat, speed_lon = 0.0007, 0.0011
elif stage_code == "STAGE_4_COLLAPSE":
dx = center_lat - agent["lat"]
dy = center_lon - agent["lon"]
target_angle = math.degrees(math.atan2(dy, dx))
agent["heading"] = (
0.8 * agent["heading"]
+ 0.2 * target_angle
+ float(rng.uniform(-4.0, 4.0))
)
speed_lat, speed_lon = 0.0006, 0.0009
else: # RTB
dx = center_lat - agent["lat"]
dy = (BASE_LON + 0.04) - agent["lon"]
target_angle = math.degrees(math.atan2(dy, dx))
agent["heading"] = (
0.9 * agent["heading"]
+ 0.1 * target_angle
+ float(rng.uniform(-2.0, 2.0))
)

speed_lat, speed_lon = 0.0005, 0.0008

rad = math.radians(agent["heading"])
agent["lat"] += math.cos(rad) * speed_lat
agent["lon"] += math.sin(rad) * speed_lon
agent["lat"] = clamp(
agent["lat"], BASE_LAT - LAT_DELTA, BASE_LAT + LAT_DELTA
)
agent["lon"] = clamp(
agent["lon"], BASE_LON - LON_DELTA, BASE_LON + LON_DELTA
)

def kharkiv_random_step(prev_lat: float, prev_lon: float) -> tuple[float, float]:
rng = tick_rng()
lat = prev_lat + float(rng.uniform(-0.0015, 0.0015))
lon = prev_lon + float(rng.uniform(-0.0025, 0.0025))
lat = clamp(lat, BASE_LAT - LAT_DELTA, BASE_LAT + LAT_DELTA)
lon = clamp(lon, BASE_LON - LON_DELTA, BASE_LON + LON_DELTA)
return float(lat), float(lon)

# ------------------------------------------------------------
# RPLE — recursive predictive logic insights
# ------------------------------------------------------------

def generate_rple_insights():

rng = tick_rng()
base_insights = [
{
"domain": "Primary",
"insight": "Primary clarity metric trending with mild volatility.",
"confidence": 0.84,
"novelty": 0.77,
},
{
"domain": "Correlation",
"insight": "Threat index and civilian density show inverse drift over last window.",
"confidence": 0.73,
"novelty": 0.71,
},
{
"domain": "Anomaly",
"insight": "Short anomaly burst detected in nav drift and comms interplay.",
"confidence": 0.91,
"novelty": 0.86,
},
{
"domain": "Risk",
"insight": "Risk and clarity remain aligned within one sigma across stages.",
"confidence": 0.68,
"novelty": 0.64,
},

]

for insight in base_insights:
insight["confidence"] = round(
min(
0.99,
max(
0.4,
insight["confidence"] + float(rng.normal(0, 0.03)),
),
),
2,
)
insight["novelty"] = round(
min(
0.99,
max(
0.3,
insight["novelty"] + float(rng.normal(0, 0.04)),
),
),
2,
)
key = sha256_hash(
{"domain": insight["domain"], "insight": insight["insight"]}
)

if key in ss.rple_memory:
insight["status"] = "Persisting"
else:
insight["status"] = "New"
ss.rple_memory.append(key)
insight["tick"] = ss.tick

ss.rple_insights = base_insights
ss.rple_memory = ss.rple_memory[-50:]

# ------------------------------------------------------------
# Synthetic vision + camera
# ------------------------------------------------------------

def generate_synthetic_vision(
threat_index: float, civ_density: float
) -> np.ndarray:
rng = tick_rng()
grid = np.zeros((40, 40), dtype=float)
grid += rng.normal(0.05, 0.03, size=grid.shape)

# Threat clusters
for _ in range(int(2 + threat_index / 20)):
cx = rng.integers(8, 32)
cy = rng.integers(8, 32)
radius = rng.integers(3, 7)

amp = 0.4 + threat_index / 200.0
for x in range(grid.shape[0]):
for y in range(grid.shape[1]):
dist = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
if dist < radius:
grid[x, y] += amp * (1 - dist / radius)

# Civilian belt
band_center = rng.integers(10, 30)
civ_amp = 0.25 + civ_density * 0.3
for x in range(grid.shape[0]):
for y in range(grid.shape[1]):
dist = abs(y - band_center)
if dist < 4:
grid[x, y] += civ_amp * (1 - dist / 4)

grid = np.clip(grid, 0.0, 1.0)
return grid

def generate_synthetic_camera_frame(
vision_grid: np.ndarray, frame_ix: int, mode_label: str
):
h, w = vision_grid.shape
base = np.zeros((h, w, 3), dtype=np.float32)

base[:, :, 0] = vision_grid
base[:, :, 1] = 0.4 * (1.0 - vision_grid)

if "Kharkiv" in mode_label:
base[:, :, 2] = 0.25
elif "Training" in mode_label:
base[:, :, 2] = 0.1
else:
base[:, :, 2] = 0.18

# Scan line
line_y = frame_ix % h
base[line_y : line_y + 1, :, :] += 0.25

# HUD reticle
cx, cy = h // 2, w // 2
radius = 6
base[cx - radius : cx + radius, cy - radius, :] += 0.3
base[cx - radius : cx + radius, cy + radius, :] += 0.3
base[cx - radius, cy - radius : cy + radius, :] += 0.3
base[cx + radius, cy - radius : cy + radius, :] += 0.3

Y, X = np.ogrid[:h, :w]
dist = np.sqrt((X - w / 2) ** 2 + (Y - h / 2) ** 2)
vignette = 1.0 - (dist / dist.max()) * 0.35
base *= vignette[:, :, None]

base = np.clip(base, 0.0, 1.0)
return (base * 255).astype("uint8")

def draw_synthetic_vision(
lat: float, lon: float, clarity: float, threat: float
):
terrain = ss.terrain.copy()
x = int(40 + (lon - BASE_LON) * 600)
y = int(40 - (lat - BASE_LAT) * 600)
x = max(2, min(77, x))
y = max(2, min(77, y))

terrain[y - 1 : y + 2, x - 1 : x + 2] += 4.0

radius = 8 + int(threat * 0.12)
yy, xx = np.ogrid[:80, :80]
mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius**2
terrain[mask] += (threat / 100.0) * 3.0

vmin, vmax = np.percentile(terrain, 5), np.percentile(terrain, 95)
fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(terrain, cmap="viridis", vmin=vmin, vmax=vmax)
ax.scatter([x], [y], c="white", s=30, marker="x")
ax.set_title(

"SYNTHETIC VISION • CORRIDOR RECON", color="white", fontsize=10
)
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
spine.set_color("white")
spine.set_linewidth(0.6)
txt = (
f"LAT {lat:.4f} LON {lon:.4f}\n"
f"CLARITY {clarity:.1f} THREAT {threat:.1f}"
)
ax.text(
0.02,
0.02,
txt,
transform=ax.transAxes,
fontsize=7,
color="white",
va="bottom",
ha="left",
bbox=dict(
facecolor="black", alpha=0.5, edgecolor="none", pad=3
),
)
fig.tight_layout()
st.pyplot(fig)

plt.close(fig)

# ------------------------------------------------------------
# Cord Cutter — Bayesian confidence
# ------------------------------------------------------------

def bayesian_confidence(telemetry: Dict[str, Any]):
prior_nav = 0.85
prior_comms = 0.90
prior_vision = 0.80

nav_drift = telemetry["nav_drift"]
comms_loss = telemetry["comms_loss"]
hot_ratio = telemetry["vision_hot_ratio"]
clarity = telemetry["clarity"]
threat = telemetry["threat_index"]

nav_likelihood = max(0.0, min(1.0, math.exp(-nav_drift / 40.0)))
comms_likelihood = max(0.0, min(1.0, math.exp(-comms_loss * 4.0)))
vision_likelihood = max(
0.0, min(1.0, 1.0 - abs(hot_ratio - 0.12) * 3.0)
)

nav_post = prior_nav * nav_likelihood
comms_post = prior_comms * comms_likelihood
vision_post = prior_vision * vision_likelihood

clamp01 = lambda x: max(0.0, min(1.0, x))

nav_conf = clamp01(nav_post)
comms_conf = clamp01(comms_post)
vision_conf = clamp01(vision_post)

clarity_factor = clarity / 100.0
threat_factor = max(0.2, 1.0 - (threat / 150.0))

combined = (
0.25 * nav_conf
+ 0.25 * comms_conf
+ 0.20 * vision_conf
+ 0.30 * clarity_factor * threat_factor
)
combined = clamp01(combined)

components = {
"nav_conf": round(nav_conf, 3),
"comms_conf": round(comms_conf, 3),
"vision_conf": round(vision_conf, 3),
"clarity_factor": round(clarity_factor, 3),
"threat_factor": round(threat_factor, 3),
"combined": round(combined, 3),
}

return combined, components

# ------------------------------------------------------------
# Telemetry generation (High-speed UAV)
# ------------------------------------------------------------

def classify_phase(mach: float, altitude_m: float, t: float) -> str:
if t < 5.0:
return "PRELAUNCH"
if mach < 0.3 and altitude_m < 1000.0:
return "TAKEOFF"
if mach < 0.8 and altitude_m < 5000.0:
return "CLIMB"
if mach < 2.1 and altitude_m >= 5000.0:
return "HIGH_SPEED_CRUISE"
if altitude_m < 3000.0 and mach < 0.6:
return "RETURN"
return "TRANSITION"

def generate_tick(prev_row: Dict[str, Any] | None) -> Dict[str, Any]:
# Per-tick deterministic RNG
ss.current_rng = rng_for_tick(ss.tick)
rng = tick_rng()

t = ss.mission_time + DT_SECONDS

ss.mission_stage_tick += 1
advance_mission_stage()
move_swarm_agents()

if ss.tick % 20 == 0:
generate_rple_insights()

env = ENVIRONMENTS[ss.environment]

# Flight state
if prev_row is None:
mach = 0.0
altitude_m = 0.0
g_load = 1.0
else:
mach = prev_row["mach"]
altitude_m = prev_row["altitude_m"]
g_load = prev_row["g_load"]

# High-speed UAV profile (no hypersonics)
if t <= 5.0:
mach = 0.0 + float(rng.normal(0.0, 0.01))
altitude_m = max(0.0, altitude_m + float(rng.normal(0.0, 1.0)))
elif t <= 35.0:
mach += float(rng.uniform(0.02, 0.06))
altitude_m += float(rng.uniform(50.0, 220.0))

elif t <= 65.0:
mach += float(rng.uniform(0.02, 0.05))
altitude_m += float(rng.uniform(60.0, 160.0))
elif t <= 95.0:
mach += float(rng.uniform(-0.03, 0.03))
altitude_m += float(rng.uniform(-80.0, 80.0))
else:
mach += float(rng.uniform(-0.06, -0.02))
altitude_m += float(rng.uniform(-200.0, -80.0))

mach = clamp(mach, 0.0, 2.5)
altitude_m = clamp(altitude_m, 0.0, 18000.0)

a_local = 295.0 # m/s, nominal low-alt speed of sound
velocity_mps = mach * a_local
rho0 = 1.225
rho = rho0 * math.exp(-altitude_m / 8000.0)
q_pa = 0.5 * rho * velocity_mps**2
q_kpa = clamp(q_pa / 1000.0, 0.0, 900.0)

thermal_index = clamp(
0.15
+ 0.55 * (mach / 2.3)
+ 0.20 * (q_kpa / 800.0)
+ env["thermal_bias"]
+ float(rng.normal(0.0, 0.02)),

0.0,
1.0,
)

if t <= 15.0:
g_load = 1.0 + float(rng.normal(0.4, 0.25))
elif t <= 55.0:
g_load = 1.0 + float(rng.normal(0.8, 0.4))
elif t <= 80.0:
g_load = 1.0 + float(rng.normal(1.3, 0.5))
else:
g_load = 1.0 + float(rng.normal(0.7, 0.3))

g_load = clamp(g_load, -1.0, 6.0)

latency = max(
40.0, env["latency_base"] + float(rng.normal(0.0, env["latency_jitter"]))
)
imu_drift = clamp(
abs(float(rng.normal(0.02 + env["imu_drift_bias"], 0.02))),
0.0,
0.15,
)

flight_phase = classify_phase(mach, altitude_m, t)

# AO state
if prev_row is None:
lat, lon = BASE_LAT, BASE_LON
threat_index = 40.0
civ_density = 0.3
nav_drift = 5.0
comms_loss = 0.0
else:
lat, lon = kharkiv_random_step(prev_row["lat"], prev_row["lon"])
threat_index = prev_row["threat_index"] + float(rng.uniform(-5, 5))
if rng.random() < 0.06:
threat_index += float(rng.uniform(8, 20))
civ_density = clamp(
prev_row["civ_density"] + float(rng.uniform(-0.08, 0.08)),
0.0,
1.0,
)
nav_drift = clamp(
prev_row["nav_drift"] + float(rng.uniform(-2, 2)), 0.0, 40.0
)
comms_loss = clamp(
prev_row["comms_loss"] + float(rng.uniform(-0.2, 0.2)),
0.0,
1.0,
)

threat_index = float(clamp(threat_index, 0.0, 100.0))
current_stage = get_current_mission_stage()

vision_grid = generate_synthetic_vision(threat_index, civ_density)
hot_pixel_ratio = float((vision_grid > 0.55).sum() / vision_grid.size)

row: Dict[str, Any] = {
"tick": ss.tick + 1,
"timestamp": datetime.utcnow().isoformat() + "Z",
"mission_time_s": round(t, 1),
"flight_phase": flight_phase,
"mission_stage": current_stage["code"],
"mission_stage_label": current_stage["label"],
"mission_stage_tick": ss.mission_stage_tick,
"mach": round(mach, 3),
"velocity_mps": round(velocity_mps, 1),
"altitude_m": round(altitude_m, 1),
"q_kpa": round(q_kpa, 1),
"thermal_index": round(thermal_index, 3),
"g_load": round(g_load, 3),
"link_latency_ms": round(latency, 1),
"imu_drift_deg_s": round(imu_drift, 4),
"lat": lat,
"lon": lon,
"threat_index": round(threat_index, 1),
"civ_density": round(civ_density, 2),

"nav_drift": round(nav_drift, 1),
"comms_loss": round(comms_loss, 2),
"vision_hot_ratio": round(hot_pixel_ratio, 3),
"vision_grid": vision_grid,
}

clarity_pack = compute_clarity_and_risk(row)
row.update(
{
"clarity": clarity_pack["clarity"],
"risk": clarity_pack["risk"],
"predicted_risk": clarity_pack["predicted_risk"],
"state": clarity_pack["state"],
"envelope_pressure": clarity_pack["envelope_pressure"],
}
)

combined_conf, conf_components = bayesian_confidence(row)
row["cc_combined"] = combined_conf
row["cc_nav_conf"] = conf_components["nav_conf"]
row["cc_comms_conf"] = conf_components["comms_conf"]
row["cc_vision_conf"] = conf_components["vision_conf"]
row["cc_clarity_factor"] = conf_components["clarity_factor"]
row["cc_threat_factor"] = conf_components["threat_factor"]

return row

# ------------------------------------------------------------
# Anomaly + volatility metrics
# ------------------------------------------------------------

def detect_anomalies(df: pd.DataFrame) -> List[Dict[str, Any]]:
if len(df) < 10:
return []
anomalies: List[Dict[str, Any]] = []
clarity_mean = df["clarity"].mean()
clarity_std = df["clarity"].std()
threat_mean = df["threat_index"].mean()
for _, row in df.tail(30).iterrows():
if clarity_std > 0 and abs(row["clarity"] - clarity_mean) > 2 * clarity_std:
anomalies.append(
{
"tick": int(row["tick"]),
"type": "Clarity Spike",
"severity": "HIGH"
if abs(row["clarity"] - clarity_mean) > 3 * clarity_std
else "MEDIUM",
"value": float(row["clarity"]),
}
)
if row["threat_index"] > threat_mean + 20:
anomalies.append(

{
"tick": int(row["tick"]),
"type": "Threat Surge",
"severity": "HIGH",
"value": float(row["threat_index"]),
}
)
return anomalies[-10:]

def compute_volatility_metrics(df: pd.DataFrame) -> Dict[str, float]:
if len(df) < 5:
return {}
tail = df.tail(40)
return {
"clarity_volatility": float(tail["clarity"].std()),
"threat_volatility": float(tail["threat_index"].std()),
"avg_clarity": float(tail["clarity"].mean()),
"avg_threat": float(tail["threat_index"].mean()),
"max_threat": float(tail["threat_index"].max()),
"min_clarity": float(tail["clarity"].min()),
}

# ------------------------------------------------------------
# Proposals
# ------------------------------------------------------------

def simulate_proposal_impact(
proposal: Dict[str, Any],
current_clarity: float,
current_threat: float,
):
rng = tick_rng()
if proposal["risk"] == "HIGH":
clarity_delta = float(rng.uniform(-5, 10))
threat_delta = float(rng.uniform(-15, 5))
else:
clarity_delta = float(rng.uniform(2, 12))
threat_delta = float(rng.uniform(-10, -2))
return {
"predicted_clarity": clamp(
current_clarity + clarity_delta, 0.0, 100.0
),
"predicted_threat": clamp(
current_threat + threat_delta, 0.0, 100.0
),
"clarity_change": clarity_delta,
"threat_change": threat_delta,
"confidence": float(rng.uniform(0.65, 0.92)),
}

def get_proposal_analytics():
if not ss.proposals:
return None
total = len(ss.proposals)
approved = sum(1 for p in ss.proposals if p["status"] == "APPROVED")
rejected = sum(1 for p in ss.proposals if p["status"] == "REJECTED")
deferred = sum(1 for p in ss.proposals if p["status"] == "DEFERRED")
pending = sum(1 for p in ss.proposals if p["status"] == "PENDING")
high_risk = sum(1 for p in ss.proposals if p["risk"] == "HIGH")
return {
"total": total,
"approved": approved,
"rejected": rejected,
"deferred": deferred,
"pending": pending,
"high_risk": high_risk,
"approval_rate": (approved / total * 100.0) if total > 0 else 0.0,
}

def maybe_generate_proposal(latest: Dict[str, Any]):
open_pending = [p for p in ss.proposals if p["status"] == "PENDING"]
if len(open_pending) >= 5:
return

rng = tick_rng()

clarity = latest["clarity"]
threat = latest["threat_index"]
civ = latest["civ_density"]
nav = latest["nav_drift"]
cc_combined = latest["cc_combined"]
cc_nav = latest["cc_nav_conf"]
cc_comms = latest["cc_comms_conf"]
envelope_pressure = latest["envelope_pressure"]

thresh = THRESHOLD_PRESETS[ss.threshold_preset]
clarity_thresh = thresh["clarity_threshold"]
threat_thresh = thresh["threat_threshold"]

triggers: List[str] = []
if clarity < clarity_thresh:
triggers.append("low_clarity")
if threat > threat_thresh:
triggers.append("high_threat")
if civ > 0.6:
triggers.append("high_civ_density")
if nav > 20:
triggers.append("nav_instability")
if cc_combined < 0.45 and (cc_nav < 0.4 or cc_comms < 0.4):
triggers.append("low_confidence")
if envelope_pressure > 1.0:

triggers.append("envelope_stress")

if not triggers:
return

pid = len(ss.proposals) + 1
title = "Stabilize corridor & trim envelope"

if "high_civ_density" in triggers:
title = "Tighten civilian protection envelope in AO"
if "nav_instability" in triggers and rng.random() < 0.5:
title = "Slow maneuver rate & reset nav reference frame"
if "envelope_stress" in triggers and rng.random() < 0.5:
title = "Reduce dynamic pressure and thermal loading"
if "low_confidence" in triggers:
title = "Hold profile; stabilize under degraded signals (Cord Cutter)"

rationale_bits: List[str] = []
if "low_clarity" in triggers:
rationale_bits.append(
f"clarity {clarity:.1f}% < {clarity_thresh:.1f}% ({ss.threshold_preset} threshold)"
)
if "high_threat" in triggers:
rationale_bits.append(
f"threat {threat:.1f} > {threat_thresh:.1f} ({ss.threshold_preset} threshold)"
)

if "high_civ_density" in triggers:
rationale_bits.append(f"elevated civilian density {civ:.2f}")
if "nav_instability" in triggers:
rationale_bits.append(f"nav drift {nav:.1f} m off corridor")
if "low_confidence" in triggers:
rationale_bits.append(
f"combined confidence {cc_combined*100:.1f}% "
f"(nav {cc_nav*100:.1f}%, comms {cc_comms*100:.1f}%)"
)
if "envelope_stress" in triggers:
rationale_bits.append(
f"flight envelope pressure elevated ({envelope_pressure:.3f})"
)

proposal = {
"id": pid,
"created": datetime.utcnow().strftime("%H:%M:%S"),
"title": title,
"rationale": "; ".join(rationale_bits),
"status": "PENDING",
"risk": "HIGH"
if (threat > 70 or clarity < 70 or envelope_pressure > 1.1)
else "MEDIUM",
"bounds": {
"max_speed_change": "≤ 15%",
"max_corridor_shift": "≤ 2.0 km",

"no_fire_control": True, # governance-only build
},
"snapshot": {
"tick": latest["tick"],
"clarity": latest["clarity"],
"risk": latest["risk"],
"threat_index": latest["threat_index"],
"civ_density": latest["civ_density"],
"nav_drift": latest["nav_drift"],
"q_kpa": latest["q_kpa"],
"thermal_index": latest["thermal_index"],
"ao_label": AO_LABEL,
},
}

ss.proposals.append(proposal)
log_event(
"PROPOSAL",
f"Proposal #{pid} queued: {title}",
{
"clarity": clarity,
"threat": threat,
"envelope_pressure": envelope_pressure,
},
)
record_audit_event(

"proposal_created", {"id": pid, "title": title, "triggers": triggers}
)

# ------------------------------------------------------------
# Event filtering
# ------------------------------------------------------------

def filter_events(events: List[Dict[str, Any]], level_filter: str, search_text: str):
filtered = events
if level_filter != "ALL":
filtered = [e for e in filtered if e["level"] == level_filter]
if search_text:
s = search_text.lower()
filtered = [e for e in filtered if s in e["msg"].lower()]
return filtered

# ------------------------------------------------------------
# UI — Header & Controls
# ------------------------------------------------------------

header_col1, header_col2 = st.columns([3, 1])
with header_col1:
st.markdown(
"<h1 style='margin-bottom:0;'>CASPER v3 — High-Speed UAV Governance
Console</h1>",
unsafe_allow_html=True,

)
st.caption(
f"Deterministic, human-gated autonomy demo — High-speed UAV + AO fusion. "
f"AO: **{AO_LABEL}**. All data synthetic and non-operational."
)
with header_col2:
st.write("")
st.write("")
st.toggle(
"Human Gate Open",
key="gate_open",
help="Gate must be open before approving any proposal.",
)

ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns(4)
with ctrl_col1:
if st.button("▶ Start Simulation", type="primary"):
ss.running = True
if not ss.swarm_agents:
init_swarm_agents(9)
generate_rple_insights()
log_event("INFO", "CASPER v3 simulation started.")
record_audit_event(
"sim_started",
{"ao_label": AO_LABEL, "run_id": ss.run_id, "rng_seed": ss.rng_seed},
)

with ctrl_col2:
if st.button("⏸ Pause"):
ss.running = False
log_event("INFO", "Simulation paused.")
record_audit_event("sim_paused", {})
with ctrl_col3:
if st.button("⟲ Reset"):
ss.running = False
init_state()
log_event("INFO", "System reset.")
record_audit_event("sim_reset", {})
with ctrl_col4:
st.markdown("**Gate Status**")
if ss.gate_open:
st.success("OPEN — proposals can be approved.")
else:
st.error("CLOSED — all proposals blocked.")

# ------------------------------------------------------------
# Simulation step
# ------------------------------------------------------------

now = time.time()
if ss.running and now - ss.last_update >= DT_SECONDS:
prev = ss.history[-1] if ss.history else None
new_row = generate_tick(prev)

ss.tick = new_row["tick"]
ss.mission_time = new_row["mission_time_s"]
ss.cam_frame_ix = (ss.cam_frame_ix + 1) % 10000

# Keep heavy vision grid separate from history to avoid bloat
row_for_history = dict(new_row)
if "vision_grid" in row_for_history:
del row_for_history["vision_grid"]
ss.history.append(row_for_history)
ss.history = ss.history[-MAX_HISTORY:]

ss.last_update = now

if new_row["state"] in ("HIGH_RISK", "CRITICAL"):
log_event(
"ALERT",
f"{new_row['state']} at tick {new_row['tick']} — "
f"clarity {new_row['clarity']:.1f}%, risk {new_row['risk']:.1f}%, "
f"threat {new_row['threat_index']:.1f}",
)
record_audit_event("high_risk_state", new_row)
maybe_generate_proposal(new_row)

# ------------------------------------------------------------
# Main body layout
# ------------------------------------------------------------

if not ss.history:
st.info("Press **Start Simulation** to bring CASPER v3 online.")
else:
df = pd.DataFrame(ss.history)
latest = df.iloc[-1]

# Top metrics
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
st.metric("Tick", int(latest["tick"]))
st.metric("State", latest["state"])
with m2:
st.metric("Mach", f"{latest['mach']:.2f}")
st.metric("Velocity (m/s)", f"{latest['velocity_mps']:.0f}")
with m3:
st.metric("Altitude (m)", f"{latest['altitude_m']:.0f}")
st.metric("q (kPa)", f"{latest['q_kpa']:.0f}")
with m4:
st.metric("Clarity (%)", f"{latest['clarity']:.1f}")
st.metric("Risk (%)", f"{latest['risk']:.1f}")
with m5:
st.metric("Threat Index", latest["threat_index"])
st.metric("Civ Density", latest["civ_density"])

st.markdown("---")

st.markdown("## 1. Live Operations Picture")

# Middle row: timeline, vision, map
top_left, top_mid, top_right = st.columns([1.4, 1.3, 1.1])

with top_left:
st.subheader("Flight & Threat Timeline (Last 150 Ticks)")
tail = df.tail(150).set_index("mission_time_s")
if not tail.empty:
chart_df = tail[["mach", "altitude_m", "clarity", "threat_index"]]
st.line_chart(chart_df)
else:
st.info("No data yet.")

st.subheader("Anomaly Detection & Volatility")
anomalies = detect_anomalies(df)
volatility = compute_volatility_metrics(df)
if anomalies:
st.warning(f"{len(anomalies)} anomalies detected in recent telemetry")
st.dataframe(
pd.DataFrame(anomalies),
use_container_width=True,
height=150,
)
if volatility:
vol_col1, vol_col2 = st.columns(2)

with vol_col1:
st.metric(
"Clarity Volatility",
f"{volatility['clarity_volatility']:.1f}",
)
st.metric("Avg Clarity", f"{volatility['avg_clarity']:.1f}%")
with vol_col2:
st.metric(
"Threat Volatility",
f"{volatility['threat_volatility']:.1f}",
)
st.metric("Max Threat", f"{volatility['max_threat']:.1f}")

with top_mid:
st.subheader("Synthetic Vision — Terrain Corridor")
draw_synthetic_vision(
lat=latest["lat"],
lon=latest["lon"],
clarity=latest["clarity"],
threat=latest["threat_index"],
)
st.caption(
"Rendered from synthetic telemetry. Moving marker + threat radius overlay."
)

st.subheader("Synthetic Camera Feed")

if st.checkbox("Enable Synthetic Camera Overlay", value=True):
cam_frame = generate_synthetic_camera_frame(
ss.terrain[20:60, 20:60], ss.cam_frame_ix, AO_LABEL
)
st.image(
cam_frame,
caption=f"Mode: {AO_LABEL} — simulated tactical viewport.",
use_container_width=True,
)
else:
st.caption("Camera overlay disabled.")

with top_right:
st.subheader(f"Flight + Swarm Track — {AO_LABEL}")
map_data = df[["lat", "lon", "threat_index"]].tail(120).copy()
map_data["color"] = map_data["threat_index"].apply(
lambda t: [255, int(255 * (1 - t / 100)), 0, 180]
if t > 50
else [0, 255, int(255 * t / 50), 180]
)

boundary_coords = [
[BASE_LON - LON_DELTA, BASE_LAT - LAT_DELTA],
[BASE_LON + LON_DELTA, BASE_LAT - LAT_DELTA],
[BASE_LON + LON_DELTA, BASE_LAT + LAT_DELTA],
[BASE_LON - LON_DELTA, BASE_LAT + LAT_DELTA],

[BASE_LON - LON_DELTA, BASE_LAT - LAT_DELTA],
]

boundary_layer = pdk.Layer(
"PathLayer",
data=[{"path": boundary_coords}],
get_path="path",
get_color=[255, 255, 0, 150],
width_min_pixels=2,
pickable=False,
)

track_layer = pdk.Layer(
"ScatterplotLayer",
data=map_data,
get_position=["lon", "lat"],
get_fill_color="color",
get_radius=80,
pickable=True,
)

layers = [boundary_layer, track_layer]

if ss.swarm_agents:
swarm_data = []
for agent in ss.swarm_agents:

role_color = {
"SCOUT": [0, 200, 255, 200],
"RELAY": [255, 165, 0, 200],
"MARKER": [150, 0, 255, 200],
}
swarm_data.append(
{
"lon": agent["lon"],
"lat": agent["lat"],
"id": agent["id"],
"role": agent["role"],
"color": role_color.get(
agent["role"], [200, 200, 200, 200]
),
}
)
swarm_layer = pdk.Layer(
"ScatterplotLayer",
data=swarm_data,
get_position=["lon", "lat"],
get_fill_color="color",
get_radius=120,
pickable=True,
)
layers.append(swarm_layer)

view_state = pdk.ViewState(
latitude=BASE_LAT,
longitude=BASE_LON,
zoom=10,
pitch=0,
)

deck = pdk.Deck(
layers=layers,
initial_view_state=view_state,
map_style="mapbox://styles/mapbox/dark-v10",
tooltip={"text": "{id} - {role}"},
)
st.pydeck_chart(deck)
st.caption(f"AO Center: ({BASE_LAT:.4f}, {BASE_LON:.4f})")
st.caption(f"AO Bounds: ±{LAT_DELTA:.3f} lat, ±{LON_DELTA:.3f} lon")
st.caption("Threat color scale: Low | Medium | High")
if ss.swarm_agents:
st.caption("Swarm roles: SCOUT | RELAY | MARKER")

st.subheader("Operational Snapshot")
st.write(
f"**Local Picture:** Clarity **{latest['clarity']:.1f}%**, "
f"risk **{latest['risk']:.1f}%**, threat **{latest['threat_index']:.1f}**, "
f"civ density **{latest['civ_density']:.2f}**, "
f"comms loss **{latest['comms_loss']:.2f}**."

)
st.caption("All numbers are synthetic; governance demo only.")

st.markdown("---")
st.markdown("## 2. Safety & Confidence Engines")

# Cord Cutter + Mission stages + RPLE
st.subheader("Cord Cutter — Signal Confidence Analysis")
cc_col1, cc_col2, cc_col3, cc_col4 = st.columns(4)
with cc_col1:
st.metric("Combined Confidence", f"{latest['cc_combined']*100:.1f}%")
st.progress(latest["cc_combined"])
st.caption(
"Overall confidence from nav, comms, vision, clarity, threat."
)
with cc_col2:
st.metric("Nav Confidence", f"{latest['cc_nav_conf']*100:.1f}%")
st.metric(
"Comms Confidence", f"{latest['cc_comms_conf']*100:.1f}%"
)
with cc_col3:
st.metric(
"Vision Confidence", f"{latest['cc_vision_conf']*100:.1f}%"
)
st.metric(
"Threat Factor", f"{latest['cc_threat_factor']*100:.1f}%"

)
with cc_col4:
st.metric(
"Clarity Factor", f"{latest['cc_clarity_factor']*100:.1f}%"
)

with st.expander("Cord Cutter Evidence Breakdown"):
st.write(
{
"nav_drift": latest["nav_drift"],
"comms_loss": latest["comms_loss"],
"vision_hot_ratio": latest["vision_hot_ratio"],
"nav_confidence": latest["cc_nav_conf"],
"comms_confidence": latest["cc_comms_conf"],
"vision_confidence": latest["cc_vision_conf"],
}
)

mission_col, rple_col = st.columns([1, 1])
with mission_col:
st.subheader("Mission Stage Progression")
current_stage = get_current_mission_stage()
stage_progress = ss.mission_stage_tick / current_stage["duration"]
st.metric("Current Stage", current_stage["label"])
st.progress(min(stage_progress, 1.0))
st.caption(

f"Stage tick {ss.mission_stage_tick}/{current_stage['duration']} "
f"({current_stage['code']})"
)
with st.expander("All Mission Stages"):
for idx, stage in enumerate(MISSION_STAGES):
status = (
"✓"
if idx < ss.mission_stage_index
else (
"▶"
if idx == ss.mission_stage_index
else "•"
)
)
st.write(
f"{status} **{stage['label']}** — {stage['description']}"
)

with rple_col:
st.subheader("RPLE — Predictive Logic Insights")
if ss.rple_insights:
for insight in ss.rple_insights:
with st.expander(
f"{insight['status']} • {insight['domain']}", expanded=False
):
st.write(f"**Insight:** {insight['insight']}")

c1, c2 = st.columns(2)
with c1:
st.metric(
"Confidence",
f"{insight['confidence']*100:.0f}%",
)
with c2:
st.metric(
"Novelty", f"{insight['novelty']*100:.0f}%"
)
else:
st.info(
"RPLE insights will appear after the system has observed enough data."
)

if ss.audit_log:
with st.expander("Audit Chain (SHA-256 Verified)"):
audit_df = pd.DataFrame(ss.audit_log[-25:])
audit_df["hash_short"] = audit_df["hash"].apply(
lambda h: h[:12] + "..."
)
audit_df["prev_short"] = audit_df["prev_hash"].apply(
lambda h: h[:12] + "..."
)
st.dataframe(
audit_df[["tick", "event_type", "prev_short", "hash_short"]],

use_container_width=True,
height=220,
)
st.caption(
"Each entry hashes (tick, timestamp, event_type, payload, prev_hash) "
"to create a tamper-evident chain."
)

st.markdown("---")
st.markdown("## 3. Governance & Audit Layer")

bottom_left, bottom_mid, bottom_right = st.columns([1.3, 0.9, 0.9])

# Proposals
with bottom_left:
st.subheader("Human-Gated Proposals")
if not ss.proposals:
st.info("No proposals yet. CASPER v3 is monitoring only.")
else:
labels = [
f"#{p['id']} [{p['status']}] {p['title']}"
for p in ss.proposals
]
selected_label = st.selectbox("Select proposal", options=labels)
selected_id = int(selected_label.split(" ")[0].replace("#", ""))
selected = next(

p for p in ss.proposals if p["id"] == selected_id
)

st.markdown(f"**Title:** {selected['title']}")
st.markdown(
f"**Status:** `{selected['status']}` | **Risk:** `{selected['risk']}`"
)
st.markdown(f"**Created:** {selected['created']}")
st.markdown("**Rationale:**")
st.write(selected["rationale"])

st.markdown("**Bounds & Guardrails:**")
bounds_df = pd.DataFrame(
[
["Max speed change", selected["bounds"]["max_speed_change"]],
["Max corridor shift", selected["bounds"]["max_corridor_shift"]],
[
"Fire Control Authority",
"DISABLED (Governance-Only Build)",
],
],
columns=["Parameter", "Constraint"],
)
st.table(bounds_df)

st.markdown("**Snapshot at Proposal Time:**")

st.json(selected["snapshot"])

st.markdown("**Predicted Impact Simulation:**")
impact = simulate_proposal_impact(
selected,
selected["snapshot"]["clarity"],
selected["snapshot"]["threat_index"],
)
ic1, ic2, ic3 = st.columns(3)
with ic1:
st.metric(
"Predicted Clarity",
f"{impact['predicted_clarity']:.1f}%",
f"{impact['clarity_change']:+.1f}%",
)
with ic2:
st.metric(
"Predicted Threat",
f"{impact['predicted_threat']:.1f}",
f"{impact['threat_change']:+.1f}",
)
with ic3:
st.metric("Confidence", f"{impact['confidence']:.0%}")

col_a, col_b, col_c = st.columns(3)
with col_a:

approve_click = st.button(
"Approve", disabled=selected["status"] != "PENDING"
)
with col_b:
reject_click = st.button(
"Reject", disabled=selected["status"] != "PENDING"
)
with col_c:
defer_click = st.button(
"Defer", disabled=selected["status"] != "PENDING"
)

if approve_click and selected["status"] == "PENDING":
if not ss.gate_open:
st.warning(
"Human gate is CLOSED. Open the gate to approve."
)
log_event(
"WARN",
f"Attempted approval of proposal #{selected['id']} with gate closed.",
)
record_audit_event(
"approve_blocked_gate_closed",
{"id": selected["id"]},
)
else:

selected["status"] = "APPROVED"
log_event(
"APPROVED",
f"Proposal #{selected['id']} approved by operator.",
)
record_audit_event(
"proposal_approved",
{
"id": selected["id"],
"title": selected["title"],
},
)

if reject_click and selected["status"] == "PENDING":
selected["status"] = "REJECTED"
log_event(
"REJECTED",
f"Proposal #{selected['id']} rejected by operator.",
)
record_audit_event(
"proposal_rejected",
{"id": selected["id"], "title": selected["title"]},
)

if defer_click and selected["status"] == "PENDING":
selected["status"] = "DEFERRED"

log_event(
"INFO",
f"Proposal #{selected['id']} deferred by operator.",
)
record_audit_event(
"proposal_deferred",
{"id": selected["id"], "title": selected["title"]},
)

# Events + export
with bottom_mid:
st.subheader("Event Feed")
fc1, fc2 = st.columns(2)
with fc1:
level_filter = st.selectbox(
"Filter level",
["ALL", "INFO", "WARN", "ALERT", "PROPOSAL", "APPROVED", "REJECTED"],
)
with fc2:
search_text = st.text_input("Search", value="")

if not ss.events:
st.info("No events yet.")
else:
ev_filtered = filter_events(ss.events, level_filter, search_text)
if ev_filtered:

feed_df = pd.DataFrame(ev_filtered[-40:])[::-1]
st.dataframe(
feed_df, use_container_width=True, height=260
)
st.caption(
f"Showing {len(ev_filtered)} filtered events of {len(ss.events)} total."
)
else:
st.warning("No events match current filters.")

st.subheader("Data Export")
ec1, ec2 = st.columns(2)
with ec1:
telemetry_records = df.tail(200).copy()
payload = {
"generated_at_utc": datetime.utcnow().isoformat() + "Z",
"ao_label": AO_LABEL,
"run_id": ss.run_id,
"rng_seed": ss.rng_seed,
"events": ss.events,
"telemetry_tail": json.loads(
telemetry_records.to_json(orient="records")
),
"proposals": ss.proposals,
"audit_log": ss.audit_log,
}

audit_json_bytes = json.dumps(payload, indent=2).encode("utf-8")
st.download_button(
"Download Audit JSON",
data=audit_json_bytes,
file_name="casper_v3_audit.json",
mime="application/json",
use_container_width=True,
)
with ec2:
csv_buffer = io.StringIO()
export_cols = [
"tick",
"timestamp",
"mission_time_s",
"clarity",
"risk",
"threat_index",
"civ_density",
"nav_drift",
"comms_loss",
"mach",
"altitude_m",
"q_kpa",
"flight_phase",
"state",
"lat",

"lon",
]
export_cols = [c for c in export_cols if c in df.columns]
df[export_cols].tail(300).to_csv(csv_buffer, index=False)
st.download_button(
"Download Telemetry CSV",
data=csv_buffer.getvalue(),
file_name="casper_v3_telemetry.csv",
mime="text/csv",
use_container_width=True,
)

with bottom_right:
st.subheader("Proposal Analytics")
analytics = get_proposal_analytics()
if analytics:
ac1, ac2 = st.columns(2)
with ac1:
st.metric("Total Proposals", analytics["total"])
st.metric("Approved", analytics["approved"])
st.metric("Rejected", analytics["rejected"])
with ac2:
st.metric("Pending", analytics["pending"])
st.metric("High Risk", analytics["high_risk"])
st.metric(
"Approval Rate", f"{analytics['approval_rate']:.0f}%"

)

status_df = pd.DataFrame(
{
"Status": [
"Approved",
"Rejected",
"Deferred",
"Pending",
],
"Count": [
analytics["approved"],
analytics["rejected"],
analytics["deferred"],
analytics["pending"],
],
}
).set_index("Status")
st.bar_chart(status_df)
else:
st.info(
"No proposals yet. Analytics will appear once proposals are generated."
)

st.subheader("Raw Telemetry (Last 40 Ticks)")
show_cols = [

"tick",
"timestamp",
"mission_time_s",
"clarity",
"risk",
"threat_index",
"civ_density",
"nav_drift",
"comms_loss",
"mach",
"altitude_m",
"q_kpa",
"flight_phase",
"state",
]
show_cols = [c for c in show_cols if c in df.columns]
st.dataframe(
df[show_cols].tail(40).set_index("tick"),
use_container_width=True,
height=260,
)

# ------------------------------------------------------------
# Auto-rerun for streaming effect
# ------------------------------------------------------------

if ss.running:
time.sleep(0.25)
st.rerun()
