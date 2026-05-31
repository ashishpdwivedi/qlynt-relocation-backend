from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
import traceback
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# 1. FastAPI App Setup
# ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Qlynt Pan-India Relocation Optimizer API",
    description="Hybrid XGBoost + 2026 Market Inflation Engine.",
    version="7.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "final_model_v2.pkl")
TEMPLATE_PATH = os.path.join(BASE_DIR, "templates", "index.html")

# ─────────────────────────────────────────────────────────────────
# 2. Config & Inflation Multipliers (2026 Market Calibration)
# ─────────────────────────────────────────────────────────────────
try:
    model_pipeline = joblib.load(MODEL_PATH)
    USE_MODEL = True
except:
    model_pipeline = None
    USE_MODEL = False

# Adjusts 2022 dataset prices to 2026 market values
CITY_INFLATION_2026 = {
    "Delhi": 1.35, "Mumbai": 1.75, "Bangalore": 1.60, 
    "Hyderabad": 1.45, "Kolkata": 1.25, "Chennai": 1.35
}

MODEL_KNOWN_CITIES = {"Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"}

FURNISH_MAP = {
    "Unfurnished": "Unfurnished",
    "Semi-Furnished": "Semi-Furnished",
    "Fully Furnished": "Furnished",
}

# ─────────────────────────────────────────────────────────────────
# 3. Locality Hub Matrix (40 Landmarks per City)
# ─────────────────────────────────────────────────────────────────
INDIA_LOCALITIES_HUBS = {
    "Delhi": [
        {"name": "Hauz Khas", "lat": 28.5494, "lon": 77.2001, "factor": 1.2}, {"name": "Saket", "lat": 28.5244, "lon": 77.2066, "factor": 1.1},
        {"name": "Connaught Place", "lat": 28.6304, "lon": 77.2177, "factor": 1.4}, {"name": "South Extension", "lat": 28.5750, "lon": 77.2210, "factor": 1.3},
        {"name": "Dwarka", "lat": 28.5921, "lon": 77.0461, "factor": 0.9}, {"name": "Rohini", "lat": 28.7041, "lon": 77.1025, "factor": 0.8},
        {"name": "Vasant Kunj", "lat": 28.5387, "lon": 77.1554, "factor": 1.2}, {"name": "Karol Bagh", "lat": 28.6514, "lon": 77.1907, "factor": 1.0},
        {"name": "Okhla", "lat": 28.5411, "lon": 77.2831, "factor": 0.85}, {"name": "Greater Kailash", "lat": 28.5484, "lon": 77.2333, "factor": 1.3},
        {"name": "Nehru Place", "lat": 28.5499, "lon": 77.2450, "factor": 1.1}, {"name": "Pitampura", "lat": 28.6946, "lon": 77.1481, "factor": 0.85},
        {"name": "Rajouri Garden", "lat": 28.6477, "lon": 77.1245, "factor": 0.95}, {"name": "Janakpuri", "lat": 28.6214, "lon": 77.0878, "factor": 0.9},
        {"name": "Laxmi Nagar", "lat": 28.6304, "lon": 77.2777, "factor": 0.75}, {"name": "Model Town", "lat": 28.7119, "lon": 77.1856, "factor": 0.8},
        {"name": "Vasant Vihar", "lat": 28.5600, "lon": 77.1622, "factor": 1.4}, {"name": "RK Puram", "lat": 28.5630, "lon": 77.1751, "factor": 1.0},
        {"name": "Mayur Vihar", "lat": 28.6015, "lon": 77.3000, "factor": 0.75}, {"name": "Chattarpur", "lat": 28.4950, "lon": 77.1820, "factor": 0.7},
        {"name": "Kalkaji", "lat": 28.5447, "lon": 77.2526, "factor": 0.95}, {"name": "Defence Colony", "lat": 28.5714, "lon": 77.2295, "factor": 1.4},
        {"name": "Green Park", "lat": 28.5573, "lon": 77.2066, "factor": 1.1}, {"name": "Malviya Nagar", "lat": 28.5365, "lon": 77.2096, "factor": 0.95},
        {"name": "Vikaspuri", "lat": 28.6338, "lon": 77.0601, "factor": 0.75}, {"name": "Tilak Nagar", "lat": 28.6366, "lon": 77.0935, "factor": 0.7},
        {"name": "Punjabi Bagh", "lat": 28.6656, "lon": 77.1265, "factor": 1.0}, {"name": "Paschim Vihar", "lat": 28.6644, "lon": 77.0988, "factor": 0.8},
        {"name": "Sarita Vihar", "lat": 28.5333, "lon": 77.2838, "factor": 0.85}, {"name": "Preet Vihar", "lat": 28.6416, "lon": 77.2917, "factor": 0.9},
        {"name": "IP Extension", "lat": 28.6298, "lon": 77.3056, "factor": 0.8}, {"name": "Shalimar Bagh", "lat": 28.7027, "lon": 77.1654, "factor": 0.8},
        {"name": "Dwarka Sec 10", "lat": 28.5830, "lon": 77.0435, "factor": 0.9}, {"name": "Dwarka Sec 21", "lat": 28.5524, "lon": 77.0537, "factor": 0.85},
        {"name": "Saket J-Block", "lat": 28.5270, "lon": 77.2180, "factor": 1.0}, {"name": "Munirka", "lat": 28.5562, "lon": 77.1738, "factor": 0.75},
        {"name": "Satya Niketan", "lat": 28.5872, "lon": 77.1709, "factor": 0.7}, {"name": "Nawada", "lat": 28.6212, "lon": 77.0574, "factor": 0.65},
        {"name": "Katwaria Sarai", "lat": 28.5422, "lon": 77.1837, "factor": 0.75}, {"name": "Uttam Nagar", "lat": 28.6206, "lon": 77.0651, "factor": 0.6}
    ],
    "Mumbai": [
        {"name": "Andheri West", "lat": 19.1176, "lon": 72.8339, "factor": 1.2}, {"name": "Bandra West", "lat": 19.0600, "lon": 72.8311, "factor": 1.8},
        {"name": "Thane West", "lat": 19.2183, "lon": 72.9781, "factor": 0.7}, {"name": "Dadar", "lat": 19.0178, "lon": 72.8478, "factor": 1.3},
        {"name": "Borivali West", "lat": 19.2292, "lon": 72.8572, "factor": 0.7}, {"name": "Malad West", "lat": 19.1860, "lon": 72.8485, "factor": 0.85},
        {"name": "Powai", "lat": 19.1176, "lon": 72.9060, "factor": 1.1}, {"name": "Colaba", "lat": 18.9067, "lon": 72.8147, "factor": 1.9},
        {"name": "Worli", "lat": 19.0125, "lon": 72.8189, "factor": 1.6}, {"name": "Juhu", "lat": 19.1075, "lon": 72.8260, "factor": 1.6},
        {"name": "Santacruz", "lat": 19.0844, "lon": 72.8358, "factor": 1.3}, {"name": "Chembur", "lat": 19.0622, "lon": 72.8974, "factor": 1.0},
        {"name": "Kharghar", "lat": 19.0270, "lon": 73.0658, "factor": 0.65}, {"name": "Kurla", "lat": 19.0825, "lon": 72.8673, "factor": 0.8},
        {"name": "Mulund", "lat": 19.1726, "lon": 72.9426, "factor": 0.9}, {"name": "Ghatkopar", "lat": 19.0865, "lon": 72.9090, "factor": 1.0},
        {"name": "Matunga", "lat": 19.0234, "lon": 72.8550, "factor": 1.3}, {"name": "Sion", "lat": 19.0392, "lon": 72.8600, "factor": 1.0},
        {"name": "Parel", "lat": 19.0060, "lon": 72.8400, "factor": 1.2}, {"name": "Lower Parel", "lat": 19.0000, "lon": 72.8290, "factor": 1.4},
        {"name": "Mahim", "lat": 19.0370, "lon": 72.8380, "factor": 1.2}, {"name": "Marine Drive", "lat": 18.9320, "lon": 72.8260, "factor": 2.0},
        {"name": "Churchgate", "lat": 18.9300, "lon": 72.8240, "factor": 1.9}, {"name": "Bandra East", "lat": 19.0500, "lon": 72.8500, "factor": 1.3},
        {"name": "Kandivali East", "lat": 19.2000, "lon": 72.8600, "factor": 0.8}, {"name": "Borivali East", "lat": 19.2300, "lon": 72.8650, "factor": 0.75},
        {"name": "Malad East", "lat": 19.1800, "lon": 72.8500, "factor": 0.8}, {"name": "Bhayandar", "lat": 19.2900, "lon": 72.8500, "factor": 0.5},
        {"name": "Vasai", "lat": 19.3900, "lon": 72.8100, "factor": 0.4}, {"name": "Virar", "lat": 19.4600, "lon": 72.8000, "factor": 0.3},
        {"name": "Dombivli", "lat": 19.2100, "lon": 73.0800, "factor": 0.5}, {"name": "Kalyan", "lat": 19.2300, "lon": 73.1300, "factor": 0.5},
        {"name": "Navi Mumbai", "lat": 19.0300, "lon": 73.0100, "factor": 0.8}, {"name": "Belapur", "lat": 19.0100, "lon": 73.0400, "factor": 0.8},
        {"name": "Seawoods", "lat": 19.0200, "lon": 73.0200, "factor": 0.9}, {"name": "Nerul", "lat": 19.0300, "lon": 73.0100, "factor": 0.85},
        {"name": "Goregaon East", "lat": 19.1600, "lon": 72.8500, "factor": 0.9}, {"name": "Vikhroli", "lat": 19.1100, "lon": 72.9200, "factor": 0.9},
        {"name": "Bhandup", "lat": 19.1400, "lon": 72.9300, "factor": 0.8}, {"name": "Kalwa", "lat": 19.1900, "lon": 72.9900, "factor": 0.6}
    ],
    "Bangalore": [
        {"name": "Whitefield", "lat": 12.9698, "lon": 77.7500, "factor": 1.2}, {"name": "Koramangala", "lat": 12.9352, "lon": 77.6244, "factor": 1.6},
        {"name": "HSR Layout", "lat": 12.9102, "lon": 77.6450, "factor": 1.3}, {"name": "Indiranagar", "lat": 12.9719, "lon": 77.6412, "factor": 1.7},
        {"name": "Electronic City", "lat": 12.8452, "lon": 77.6602, "factor": 0.8}, {"name": "Jayanagar", "lat": 12.9307, "lon": 77.5832, "factor": 1.2},
        {"name": "Marathahalli", "lat": 12.9569, "lon": 77.7011, "factor": 1.0}, {"name": "BTM Layout", "lat": 12.9166, "lon": 77.6101, "factor": 1.1},
        {"name": "Yelahanka", "lat": 13.1007, "lon": 77.5963, "factor": 0.8}, {"name": "Bellandur", "lat": 12.9304, "lon": 77.6784, "factor": 1.2},
        {"name": "Hebbal", "lat": 13.0357, "lon": 77.5970, "factor": 1.1}, {"name": "Sarjapur", "lat": 12.8715, "lon": 77.7854, "factor": 0.9},
        {"name": "Banashankari", "lat": 12.9255, "lon": 77.5468, "factor": 0.95}, {"name": "Malleshwaram", "lat": 13.0033, "lon": 77.5645, "factor": 1.3},
        {"name": "Rajajinagar", "lat": 12.9982, "lon": 77.5530, "factor": 1.1}, {"name": "CV Raman Nagar", "lat": 12.9863, "lon": 77.6677, "factor": 1.0},
        {"name": "Ulsoor", "lat": 12.9774, "lon": 77.6212, "factor": 1.4}, {"name": "Basavanagudi", "lat": 12.9419, "lon": 77.5755, "factor": 1.1},
        {"name": "Frazer Town", "lat": 13.0019, "lon": 77.6108, "factor": 1.2}, {"name": "Hennur", "lat": 13.0456, "lon": 77.6366, "factor": 0.9},
        {"name": "JP Nagar", "lat": 12.9063, "lon": 77.5857, "factor": 1.1}, {"name": "Domlur", "lat": 12.9600, "lon": 77.6380, "factor": 1.3},
        {"name": "Cox Town", "lat": 12.9900, "lon": 77.6100, "factor": 1.2}, {"name": "Vasanth Nagar", "lat": 12.9800, "lon": 77.5900, "factor": 1.5},
        {"name": "Richmond Town", "lat": 12.9600, "lon": 77.6000, "factor": 1.7}, {"name": "Shanti Nagar", "lat": 12.9500, "lon": 77.5900, "factor": 1.3},
        {"name": "Wilson Garden", "lat": 12.9400, "lon": 77.5900, "factor": 1.0}, {"name": "Whitefield Hope Farm", "lat": 12.9700, "lon": 77.7400, "factor": 1.0},
        {"name": "Brookefield", "lat": 12.9600, "lon": 77.7200, "factor": 1.0}, {"name": "Thippasandra", "lat": 12.9700, "lon": 77.6500, "factor": 1.1},
        {"name": "Kaggadasapura", "lat": 12.9800, "lon": 77.6700, "factor": 0.9}, {"name": "Kundalahalli", "lat": 12.9600, "lon": 77.7100, "factor": 1.1},
        {"name": "Harlur", "lat": 12.9100, "lon": 77.6500, "factor": 1.0}, {"name": "Bommanahalli", "lat": 12.9000, "lon": 77.6200, "factor": 0.8},
        {"name": "Arekere", "lat": 12.8900, "lon": 77.6000, "factor": 0.8}, {"name": "Talaghattapura", "lat": 12.8700, "lon": 77.5400, "factor": 0.6},
        {"name": "Kanakapura Road", "lat": 12.8900, "lon": 77.5600, "factor": 0.7}, {"name": "Bannerghatta Road", "lat": 12.8900, "lon": 77.5900, "factor": 0.8},
        {"name": "Begur", "lat": 12.8800, "lon": 77.6200, "factor": 0.75}, {"name": "Electronic City Ph2", "lat": 12.8300, "lon": 77.6700, "factor": 0.7}
    ],
    "Hyderabad": [
        {"name": "Gachibowli", "lat": 17.4401, "lon": 78.3489, "factor": 1.3}, {"name": "Hitech City", "lat": 17.4483, "lon": 78.3741, "factor": 1.4},
        {"name": "Kukatpally", "lat": 17.4855, "lon": 78.3973, "factor": 0.9}, {"name": "Jubilee Hills", "lat": 17.4367, "lon": 78.4065, "factor": 1.8},
        {"name": "Madhapur", "lat": 17.4485, "lon": 78.3908, "factor": 1.3}, {"name": "Manikonda", "lat": 17.3977, "lon": 78.3916, "factor": 1.0},
        {"name": "Kondapur", "lat": 17.4622, "lon": 78.3568, "factor": 1.1}, {"name": "Banjara Hills", "lat": 17.4174, "lon": 78.4414, "factor": 1.9},
        {"name": "Ameerpet", "lat": 17.4374, "lon": 78.4482, "factor": 0.9}, {"name": "Miyapur", "lat": 17.4933, "lon": 78.3512, "factor": 0.8},
        {"name": "Begumpet", "lat": 17.4437, "lon": 78.4526, "factor": 1.0}, {"name": "Uppal", "lat": 17.4048, "lon": 78.5607, "factor": 0.7},
        {"name": "Kokapet", "lat": 17.3916, "lon": 78.3243, "factor": 1.1}, {"name": "Nanakramguda", "lat": 17.4116, "lon": 78.3305, "factor": 1.2},
        {"name": "Kothaguda", "lat": 17.4610, "lon": 78.3650, "factor": 1.0}, {"name": "Tellapur", "lat": 17.5100, "lon": 78.2700, "factor": 0.7},
        {"name": "Attapur", "lat": 17.3750, "lon": 78.4230, "factor": 0.8}, {"name": "Sainikpuri", "lat": 17.4855, "lon": 78.5360, "factor": 0.75},
        {"name": "Dilsukhnagar", "lat": 17.3670, "lon": 78.5280, "factor": 0.7}, {"name": "Gachibowli Outer", "lat": 17.4260, "lon": 78.3500, "factor": 1.1},
        {"name": "Tolichowki", "lat": 17.4000, "lon": 78.4000, "factor": 0.85}, {"name": "Somajiguda", "lat": 17.4300, "lon": 78.4500, "factor": 1.2},
        {"name": "Secunderabad", "lat": 17.4300, "lon": 78.5000, "factor": 0.9}, {"name": "Himayat Nagar", "lat": 17.4000, "lon": 78.4800, "factor": 1.0},
        {"name": "KPHB", "lat": 17.4900, "lon": 78.3900, "factor": 0.8}, {"name": "Bowenpally", "lat": 17.4700, "lon": 78.4600, "factor": 0.75},
        {"name": "Lakdikapul", "lat": 17.4000, "lon": 78.4500, "factor": 0.9}, {"name": "Malakpet", "lat": 17.3700, "lon": 78.4800, "factor": 0.65},
        {"name": "Alwal", "lat": 17.5000, "lon": 78.5000, "factor": 0.7}, {"name": "Beeramguda", "lat": 17.5200, "lon": 78.2500, "factor": 0.6},
        {"name": "Nizampet", "lat": 17.5100, "lon": 78.3800, "factor": 0.7}, {"name": "Bachupally", "lat": 17.5300, "lon": 78.3800, "factor": 0.7},
        {"name": "LB Nagar", "lat": 17.3400, "lon": 78.5500, "factor": 0.65}, {"name": "Mehdipatnam", "lat": 17.3900, "lon": 78.4300, "factor": 0.8},
        {"name": "Shaikpet", "lat": 17.4100, "lon": 78.4000, "factor": 0.9}, {"name": "Manikonda Extension", "lat": 17.3900, "lon": 78.3800, "factor": 0.85},
        {"name": "Serilingampally", "lat": 17.4800, "lon": 78.3000, "factor": 0.8}, {"name": "Kompally", "lat": 17.5300, "lon": 78.4800, "factor": 0.7},
        {"name": "Shamshabad", "lat": 17.2500, "lon": 78.4000, "factor": 0.6}, {"name": "Khajaguda", "lat": 17.4200, "lon": 78.3700, "factor": 1.0}
    ],
    "Kolkata": [
        {"name": "Salt Lake", "lat": 22.5804, "lon": 88.4143, "factor": 1.2}, {"name": "New Town", "lat": 22.5898, "lon": 88.4748, "factor": 1.1},
        {"name": "Garia", "lat": 22.4650, "lon": 88.3903, "factor": 0.7}, {"name": "Ballygunge", "lat": 22.5273, "lon": 88.3664, "factor": 1.6},
        {"name": "Bandel", "lat": 22.9231, "lon": 88.3789, "factor": 0.6}, {"name": "Dumdum", "lat": 22.6178, "lon": 88.4031, "factor": 0.8},
        {"name": "Jadavpur", "lat": 22.4994, "lon": 88.3694, "factor": 1.0}, {"name": "Tollygunge", "lat": 22.4934, "lon": 88.3475, "factor": 0.9},
        {"name": "Rajarhat", "lat": 22.6152, "lon": 88.4619, "factor": 1.0}, {"name": "Behala", "lat": 22.4950, "lon": 88.3194, "factor": 0.75},
        {"name": "Lake Gardens", "lat": 22.4980, "lon": 88.3530, "factor": 0.9}, {"name": "Kasba", "lat": 22.5110, "lon": 88.3840, "factor": 0.95},
        {"name": "Howrah", "lat": 22.5958, "lon": 88.2636, "factor": 0.7}, {"name": "Baranagar", "lat": 22.6450, "lon": 88.3650, "factor": 0.75},
        {"name": "Kaikhali", "lat": 22.6280, "lon": 88.4250, "factor": 0.7}, {"name": "Santoshpur", "lat": 22.4860, "lon": 88.3850, "factor": 0.8},
        {"name": "Park Street", "lat": 22.5530, "lon": 88.3510, "factor": 1.7}, {"name": "Alipore", "lat": 22.5310, "lon": 88.3310, "factor": 1.7},
        {"name": "Baguiati", "lat": 22.6180, "lon": 88.4120, "factor": 0.8}, {"name": "Gariahat", "lat": 22.5150, "lon": 88.3680, "factor": 1.2},
        {"name": "Ruby Area", "lat": 22.5100, "lon": 88.3900, "factor": 1.0}, {"name": "Mukundapur", "lat": 22.4900, "lon": 88.4000, "factor": 0.85},
        {"name": "Baruipur", "lat": 22.3500, "lon": 88.4300, "factor": 0.5}, {"name": "Sonarpur", "lat": 22.4300, "lon": 88.4200, "factor": 0.6},
        {"name": "Madhyamgram", "lat": 22.7000, "lon": 88.4400, "factor": 0.6}, {"name": "Barrackpore", "lat": 22.7500, "lon": 88.3600, "factor": 0.5},
        {"name": "Kestopur", "lat": 22.6000, "lon": 88.4100, "factor": 0.7}, {"name": "Tangra", "lat": 22.5500, "lon": 88.3900, "factor": 0.8},
        {"name": "Bhowanipore", "lat": 22.5300, "lon": 88.3400, "factor": 1.3}, {"name": "Kalighat", "lat": 22.5200, "lon": 88.3400, "factor": 1.2},
        {"name": "Rashbehari", "lat": 22.5100, "lon": 88.3500, "factor": 1.2}, {"name": "Golf Green", "lat": 22.4900, "lon": 88.3600, "factor": 1.0},
        {"name": "Santragachi", "lat": 22.5700, "lon": 88.2900, "factor": 0.6}, {"name": "Uttarpara", "lat": 22.6700, "lon": 88.3400, "factor": 0.5},
        {"name": "Serampore", "lat": 22.7500, "lon": 88.3400, "factor": 0.5}, {"name": "Shyambazar", "lat": 22.6000, "lon": 88.3700, "factor": 0.9},
        {"name": "Maniktala", "lat": 22.5800, "lon": 88.3800, "factor": 0.9}, {"name": "Kankurgachi", "lat": 22.5800, "lon": 88.4000, "factor": 1.0},
        {"name": "New Alipore", "lat": 22.5100, "lon": 88.3200, "factor": 1.1}, {"name": "Beleghata", "lat": 22.5700, "lon": 88.3900, "factor": 0.85}
    ],
    "Chennai": [
        {"name": "Velachery", "lat": 12.9802, "lon": 80.2227, "factor": 1.0}, {"name": "Adyar", "lat": 13.0012, "lon": 80.2565, "factor": 1.5},
        {"name": "Anna Nagar", "lat": 13.0850, "lon": 80.2101, "factor": 1.4}, {"name": "Thoraipakkam", "lat": 12.9430, "lon": 80.2343, "factor": 0.9},
        {"name": "Tambaram", "lat": 12.9229, "lon": 80.1275, "factor": 0.7}, {"name": "Adambakkam", "lat": 12.9892, "lon": 80.2078, "factor": 0.85},
        {"name": "Thiruvanmiyur", "lat": 12.9830, "lon": 80.2594, "factor": 1.2}, {"name": "Medavakkam", "lat": 12.9191, "lon": 80.1932, "factor": 0.7},
        {"name": "Perungudi", "lat": 12.9654, "lon": 80.2414, "factor": 1.0}, {"name": "Porur", "lat": 13.0382, "lon": 80.1565, "factor": 0.85},
        {"name": "OMR", "lat": 12.9200, "lon": 80.2300, "factor": 0.95}, {"name": "Kodambakkam", "lat": 13.0510, "lon": 80.2210, "factor": 1.1},
        {"name": "Mylapore", "lat": 13.0330, "lon": 80.2640, "factor": 1.3}, {"name": "Besant Nagar", "lat": 12.9990, "lon": 80.2670, "factor": 1.4},
        {"name": "Sholinganallur", "lat": 12.9000, "lon": 80.2270, "factor": 0.9}, {"name": "Pallavaram", "lat": 12.9570, "lon": 80.1480, "factor": 0.7},
        {"name": "Ambattur", "lat": 13.1140, "lon": 80.1500, "factor": 0.75}, {"name": "Chromepet", "lat": 12.9490, "lon": 80.1430, "factor": 0.7},
        {"name": "Guindy", "lat": 13.0060, "lon": 80.2200, "factor": 1.1}, {"name": "T. Nagar", "lat": 13.0400, "lon": 80.2330, "factor": 1.4},
        {"name": "Adayar", "lat": 13.0000, "lon": 80.2500, "factor": 1.4}, {"name": "Saidapet", "lat": 13.0200, "lon": 80.2200, "factor": 1.0},
        {"name": "Vadapalani", "lat": 13.0500, "lon": 80.2100, "factor": 1.0}, {"name": "Virugambakkam", "lat": 13.0500, "lon": 80.1900, "factor": 0.9},
        {"name": "Valasaravakkam", "lat": 13.0400, "lon": 80.1800, "factor": 0.85}, {"name": "Madipakkam", "lat": 12.9600, "lon": 80.1900, "factor": 0.8},
        {"name": "Nanganallur", "lat": 12.9800, "lon": 80.1800, "factor": 0.8}, {"name": "Kilpauk", "lat": 13.0800, "lon": 80.2400, "factor": 1.2},
        {"name": "Royapettah", "lat": 13.0400, "lon": 80.2600, "factor": 1.3}, {"name": "Alwarpet", "lat": 13.0300, "lon": 80.2500, "factor": 1.4},
        {"name": "Nungambakkam", "lat": 13.0600, "lon": 80.2400, "factor": 1.4}, {"name": "Egmore", "lat": 13.0700, "lon": 80.2500, "factor": 1.2},
        {"name": "Triplicane", "lat": 13.0500, "lon": 80.2700, "factor": 1.1}, {"name": "Perambur", "lat": 13.1000, "lon": 80.2300, "factor": 0.8},
        {"name": "Kolathur", "lat": 13.1100, "lon": 80.2100, "factor": 0.8}, {"name": "Ayapakkam", "lat": 13.0900, "lon": 80.1600, "factor": 0.7},
        {"name": "Puzhal", "lat": 13.1600, "lon": 80.2000, "factor": 0.6}, {"name": "Avadi", "lat": 13.1200, "lon": 80.1000, "factor": 0.6},
        {"name": "Poonamallee", "lat": 13.0500, "lon": 80.1000, "factor": 0.65}, {"name": "Madhavaram", "lat": 13.1500, "lon": 80.2300, "factor": 0.7}
    ]
}

# ─────────────────────────────────────────────────────────────────
# 4. Request Schema
# ─────────────────────────────────────────────────────────────────
class RelocationInput(BaseModel):
    office_location: str
    max_budget: float
    preferred_bhk: int
    property_size: float
    furnishing_status: str
    bathrooms: int
    area_type: str
    tenant_preference: str

# ─────────────────────────────────────────────────────────────────
# 5. Serve Frontend & Optimize
# ─────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve():
    return FileResponse(TEMPLATE_PATH)

@app.post("/optimize")
def optimize_relocation(input_data: RelocationInput):
    try:
        geolocator = Nominatim(user_agent="qlynt_prod_v7")
        location = geolocator.geocode(f"{input_data.office_location}, India", timeout=10)
        if not location:
            raise HTTPException(status_code=404, detail="Location not found.")
        
        target_city = "Delhi"
        addr = location.address.lower()
        for city in ["Delhi", "Mumbai", "Bangalore", "Hyderabad", "Kolkata", "Chennai"]:
            if city.lower() in addr:
                target_city = city
                break
        
        hubs = INDIA_LOCALITIES_HUBS.get(target_city, INDIA_LOCALITIES_HUBS["Delhi"])
        
        # Batch Predict
        batch = [{
            "locality": hub["name"],
            "bhk": input_data.preferred_bhk,
            "size": input_data.property_size,
            "city": target_city,
            "furnishing_status": input_data.furnishing_status,
            "bathroom": input_data.bathrooms,
            "area_type": input_data.area_type,
            "tenant_preferred": input_data.tenant_preference
        } for hub in hubs]
        
        df = pd.DataFrame(batch)
        raw_preds = model_pipeline.predict(df)
        
        # Market Calibration (Fixing low rent output)
        inflation = CITY_INFLATION_2026.get(target_city, 1.3)
        res = []
        for i, hub in enumerate(hubs):
            # Applying 2026 Calibration Multiplier to 2022 raw output
            rent = int(np.round(np.expm1(raw_preds[i]) * inflation))
            
            if rent <= input_data.max_budget:
                dist = geodesic((location.latitude, location.longitude), (hub["lat"], hub["lon"])).km
                res.append({
                    "locality": hub["name"],
                    "predicted_rent": rent,
                    "distance_km": round(dist, 1)
                })
        
        return {"success": True, "recommendations": sorted(res, key=lambda x: x['distance_km'])[:5]}
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
