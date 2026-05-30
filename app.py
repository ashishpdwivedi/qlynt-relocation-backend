from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# 1. FastAPI App
# ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Qlynt Pan-India Relocation Optimizer API",
    description="Hybrid XGBoost + locality-multiplier rent engine for Indian cities.",
    version="6.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────
# 2. Load Model
#    Features: bhk | size | city | furnishing_status | area_type
#              bathroom | floor_num | total_floors | floor_ratio
#    Target:   log(rent)  →  we expm1() to get ₹ back
#    Known cities:  Bangalore, Chennai, Delhi, Hyderabad, Kolkata, Mumbai
#    Known furnishing: Furnished | Semi-Furnished | Unfurnished
# ─────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "final_model_v2.pkl")

try:
    model_pipeline = joblib.load(MODEL_PATH)
    USE_MODEL = True
    print("✅ Model loaded successfully!")
except Exception as e:
    model_pipeline = None
    USE_MODEL = False
    print(f"⚠️  Model load failed: {e} — using fallback estimation engine.")

MODEL_KNOWN_CITIES = {"Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"}

# Map "Fully Furnished" → "Furnished" (model training label)
FURNISH_MAP = {
    "Unfurnished":    "Unfurnished",
    "Semi-Furnished": "Semi-Furnished",
    "Fully Furnished": "Furnished",
}

# ─────────────────────────────────────────────────────────────────
# 3. Locality Hub Matrix
#    factor: multiplier vs city average (premium > 1.0, budget < 1.0)
#    Estimated floor stats per locality type (mid-rise standard)
# ─────────────────────────────────────────────────────────────────
INDIA_LOCALITIES_HUBS = {
    "Delhi": [
        {"name": "Hauz Khas",        "lat": 28.5494, "lon": 77.2001, "factor": 1.25, "floor": 2, "total": 4},
        {"name": "Saket",            "lat": 28.5244, "lon": 77.2066, "factor": 1.15, "floor": 2, "total": 5},
        {"name": "Connaught Place",  "lat": 28.6304, "lon": 77.2177, "factor": 1.50, "floor": 3, "total": 6},
        {"name": "Katwaria Sarai",   "lat": 28.5422, "lon": 77.1837, "factor": 0.85, "floor": 1, "total": 3},
        {"name": "Nawada",           "lat": 28.6212, "lon": 77.0574, "factor": 0.60, "floor": 1, "total": 3},
        {"name": "Laxmi Nagar",      "lat": 28.6304, "lon": 77.2777, "factor": 0.70, "floor": 1, "total": 4},
        {"name": "Uttam Nagar",      "lat": 28.6206, "lon": 77.0651, "factor": 0.55, "floor": 1, "total": 3},
        {"name": "Janakpuri",        "lat": 28.6214, "lon": 77.0878, "factor": 0.75, "floor": 2, "total": 4},
        {"name": "Dwarka Sector 12", "lat": 28.5921, "lon": 77.0461, "factor": 0.80, "floor": 2, "total": 5},
        {"name": "Rohini",           "lat": 28.7041, "lon": 77.1025, "factor": 0.72, "floor": 1, "total": 4},
        {"name": "Vasant Kunj",      "lat": 28.5387, "lon": 77.1554, "factor": 1.20, "floor": 2, "total": 4},
        {"name": "Karol Bagh",       "lat": 28.6514, "lon": 77.1907, "factor": 1.05, "floor": 2, "total": 5},
        {"name": "Okhla Phase 3",    "lat": 28.5411, "lon": 77.2831, "factor": 0.90, "floor": 2, "total": 4},
    ],
    "Mumbai": [
        {"name": "Andheri West",   "lat": 19.1176, "lon": 72.8339, "factor": 1.10, "floor": 5, "total": 12},
        {"name": "Bandra West",    "lat": 19.0600, "lon": 72.8311, "factor": 1.75, "floor": 6, "total": 14},
        {"name": "Thane West",     "lat": 19.2183, "lon": 72.9781, "factor": 0.68, "floor": 4, "total": 10},
        {"name": "Dadar",          "lat": 19.0178, "lon": 72.8478, "factor": 1.25, "floor": 5, "total": 10},
        {"name": "Borivali West",  "lat": 19.2292, "lon": 72.8572, "factor": 0.62, "floor": 3, "total": 8},
        {"name": "Malad West",     "lat": 19.1860, "lon": 72.8485, "factor": 0.78, "floor": 4, "total": 10},
        {"name": "Kandivali West", "lat": 19.2046, "lon": 72.8522, "factor": 0.68, "floor": 3, "total": 9},
        {"name": "Chembur",        "lat": 19.0622, "lon": 72.8974, "factor": 0.88, "floor": 4, "total": 9},
        {"name": "Goregaon West",  "lat": 19.1633, "lon": 72.8412, "factor": 0.84, "floor": 4, "total": 10},
        {"name": "Powai",          "lat": 19.1176, "lon": 72.9060, "factor": 1.00, "floor": 5, "total": 12},
        {"name": "Mira Road East", "lat": 19.2817, "lon": 72.8561, "factor": 0.50, "floor": 3, "total": 8},
    ],
    "Bangalore": [
        {"name": "Whitefield",      "lat": 12.9698, "lon": 77.7500, "factor": 1.30, "floor": 3, "total": 8},
        {"name": "Koramangala",     "lat": 12.9352, "lon": 77.6244, "factor": 1.65, "floor": 3, "total": 7},
        {"name": "HSR Layout",      "lat": 12.9102, "lon": 77.6450, "factor": 1.40, "floor": 2, "total": 6},
        {"name": "Indiranagar",     "lat": 12.9719, "lon": 77.6412, "factor": 1.75, "floor": 3, "total": 7},
        {"name": "Electronic City", "lat": 12.8452, "lon": 77.6602, "factor": 0.82, "floor": 4, "total": 10},
        {"name": "Jayanagar",       "lat": 12.9307, "lon": 77.5832, "factor": 1.18, "floor": 2, "total": 5},
        {"name": "Marathahalli",    "lat": 12.9569, "lon": 77.7011, "factor": 1.05, "floor": 3, "total": 7},
        {"name": "BTM Layout",      "lat": 12.9166, "lon": 77.6101, "factor": 1.12, "floor": 2, "total": 5},
        {"name": "Yelahanka",       "lat": 13.1007, "lon": 77.5963, "factor": 0.76, "floor": 2, "total": 5},
        {"name": "Bellandur",       "lat": 12.9304, "lon": 77.6784, "factor": 1.24, "floor": 3, "total": 8},
    ],
    "Hyderabad": [
        {"name": "Gachibowli",    "lat": 17.4401, "lon": 78.3489, "factor": 1.35, "floor": 3, "total": 8},
        {"name": "Hitech City",   "lat": 17.4483, "lon": 78.3741, "factor": 1.45, "floor": 4, "total": 9},
        {"name": "Kukatpally",    "lat": 17.4855, "lon": 78.3973, "factor": 0.92, "floor": 2, "total": 6},
        {"name": "Jubilee Hills", "lat": 17.4367, "lon": 78.4065, "factor": 1.85, "floor": 2, "total": 5},
        {"name": "Madhapur",      "lat": 17.4485, "lon": 78.3908, "factor": 1.38, "floor": 3, "total": 8},
        {"name": "Manikonda",     "lat": 17.3977, "lon": 78.3916, "factor": 0.98, "floor": 2, "total": 6},
        {"name": "Kondapur",      "lat": 17.4622, "lon": 78.3568, "factor": 1.18, "floor": 3, "total": 7},
        {"name": "Banjara Hills", "lat": 17.4174, "lon": 78.4414, "factor": 1.95, "floor": 2, "total": 4},
        {"name": "Ameerpet",      "lat": 17.4374, "lon": 78.4482, "factor": 0.85, "floor": 2, "total": 5},
        {"name": "Miyapur",       "lat": 17.4933, "lon": 78.3512, "factor": 0.78, "floor": 2, "total": 6},
    ],
    "Kolkata": [
        {"name": "Salt Lake City Sector 2", "lat": 22.5804, "lon": 88.4143, "factor": 1.30, "floor": 2, "total": 5},
        {"name": "New Town",      "lat": 22.5898, "lon": 88.4748, "factor": 1.14, "floor": 3, "total": 7},
        {"name": "Garia",         "lat": 22.4650, "lon": 88.3903, "factor": 0.73, "floor": 1, "total": 3},
        {"name": "Ballygunge",    "lat": 22.5273, "lon": 88.3664, "factor": 1.62, "floor": 2, "total": 4},
        {"name": "Bandel",        "lat": 22.9231, "lon": 88.3789, "factor": 0.57, "floor": 1, "total": 2},
        {"name": "Dumdum Park",   "lat": 22.6178, "lon": 88.4031, "factor": 0.81, "floor": 1, "total": 3},
        {"name": "South Dum Dum", "lat": 22.6111, "lon": 88.4011, "factor": 0.77, "floor": 1, "total": 3},
        {"name": "Jadavpur",      "lat": 22.4994, "lon": 88.3694, "factor": 0.97, "floor": 2, "total": 4},
        {"name": "Tollygunge",    "lat": 22.4934, "lon": 88.3475, "factor": 0.89, "floor": 1, "total": 3},
        {"name": "Rajarhat",      "lat": 22.6152, "lon": 88.4619, "factor": 1.05, "floor": 3, "total": 7},
        {"name": "Behala",        "lat": 22.4950, "lon": 88.3194, "factor": 0.73, "floor": 1, "total": 3},
    ],
    "Chennai": [
        {"name": "Velachery",     "lat": 12.9802, "lon": 80.2227, "factor": 1.10, "floor": 2, "total": 5},
        {"name": "Adyar",         "lat": 13.0012, "lon": 80.2565, "factor": 1.52, "floor": 2, "total": 5},
        {"name": "Anna Nagar",    "lat": 13.0850, "lon": 80.2101, "factor": 1.38, "floor": 2, "total": 6},
        {"name": "Thoraipakkam",  "lat": 12.9430, "lon": 80.2343, "factor": 0.96, "floor": 3, "total": 7},
        {"name": "Tambaram",      "lat": 12.9229, "lon": 80.1275, "factor": 0.69, "floor": 1, "total": 3},
        {"name": "Adambakkam",    "lat": 12.9892, "lon": 80.2078, "factor": 0.90, "floor": 2, "total": 4},
        {"name": "Thiruvanmiyur", "lat": 12.9830, "lon": 80.2594, "factor": 1.24, "floor": 2, "total": 5},
        {"name": "Medavakkam",    "lat": 12.9191, "lon": 80.1932, "factor": 0.76, "floor": 2, "total": 4},
        {"name": "Perungudi",     "lat": 12.9654, "lon": 80.2414, "factor": 1.03, "floor": 3, "total": 6},
        {"name": "Porur",         "lat": 13.0382, "lon": 80.1565, "factor": 0.90, "floor": 2, "total": 4},
    ]
}

# ─────────────────────────────────────────────────────────────────
# 4. Fallback engine for Pune / model-offline scenarios
# ─────────────────────────────────────────────────────────────────
BHK_MULT     = {1: 1.0, 2: 1.6, 3: 2.3, 4: 3.1}
FURNISH_MULT = {"Unfurnished": 1.0, "Semi-Furnished": 1.18, "Fully Furnished": 1.38}

def fallback_estimate(base_rent, bhk, size, furnishing):
    bhk_m     = BHK_MULT.get(bhk, 1.0 + (bhk - 1) * 0.55)
    furnish_m = FURNISH_MULT.get(furnishing, 1.0)
    size_f    = max(0.75, min(1.0 + ((size - 600) / 100) * 0.03, 1.8))
    return int(round(base_rent * bhk_m * furnish_m * size_f / 500) * 500)

# ─────────────────────────────────────────────────────────────────
# 5. Request Schema
# ─────────────────────────────────────────────────────────────────
class RelocationInput(BaseModel):
    office_location:   str
    max_budget:        float
    preferred_bhk:     int
    property_size:     float
    furnishing_status: str
    bathrooms:         int = 2

# ─────────────────────────────────────────────────────────────────
# 6. Health Check
# ─────────────────────────────────────────────────────────────────
@app.get("/")
def check_status():
    return {
        "status":       "online",
        "coverage":     list(INDIA_LOCALITIES_HUBS.keys()),
        "model_active": USE_MODEL,
        "version":      "6.0"
    }

# ─────────────────────────────────────────────────────────────────
# 7. Core Optimization Route
# ─────────────────────────────────────────────────────────────────
@app.post("/optimize")
def optimize_relocation(input_data: RelocationInput):

    # ── Geocode ─────────────────────────────────────────────────────
    try:
        geolocator = Nominatim(user_agent="qlynt_pan_india_agent_v6")
        location   = geolocator.geocode(f"{input_data.office_location}, India", timeout=10)
        if not location:
            raise HTTPException(status_code=404, detail="Could not pin location. Try a clearer landmark or area name.")
        office_lat    = location.latitude
        office_lon    = location.longitude
        resolved_addr = location.address.lower()
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=503, detail="Geocoding timed out. Please try again.")

    # ── City detection ──────────────────────────────────────────────
    city_aliases = {
        "Delhi":     ["delhi", "new delhi"],
        "Mumbai":    ["mumbai", "bombay"],
        "Bangalore": ["bangalore", "bengaluru"],
        "Hyderabad": ["hyderabad"],
        "Kolkata":   ["kolkata", "calcutta"],
        "Chennai":   ["chennai", "madras"],
    }
    target_city = None
    for city, aliases in city_aliases.items():
        if any(a in resolved_addr for a in aliases):
            target_city = city
            break
    if not target_city:
        ncr = ["gurgaon", "gurugram", "noida", "faridabad", "ghaziabad"]
        target_city = "Delhi" if any(k in resolved_addr for k in ncr) else "Delhi"

    localities = INDIA_LOCALITIES_HUBS.get(target_city, [])
    bathrooms  = max(1, min(input_data.bathrooms, input_data.preferred_bhk + 1))

    # ── Predict rents ───────────────────────────────────────────────
    rent_map = {}

    if USE_MODEL and model_pipeline and target_city in MODEL_KNOWN_CITIES:
        model_furnishing = FURNISH_MAP.get(input_data.furnishing_status, "Semi-Furnished")

        batch_rows = []
        for area in localities:
            floor_num   = area["floor"]
            total_floors = area["total"]
            floor_ratio  = round(floor_num / max(total_floors, 1), 3)
            batch_rows.append({
                "bhk":              input_data.preferred_bhk,
                "size":             input_data.property_size,
                "city":             target_city,
                "furnishing_status": model_furnishing,
                "area_type":        "Super Area",
                "bathroom":         bathrooms,
                "floor_num":        floor_num,
                "total_floors":     total_floors,
                "floor_ratio":      floor_ratio,
            })

        input_df  = pd.DataFrame(batch_rows)
        log_preds = model_pipeline.predict(input_df)
        base_preds = np.expm1(log_preds)

        for i, area in enumerate(localities):
            raw = base_preds[i] * area["factor"]
            rent_map[area["name"]] = int(round(raw / 500) * 500)
    else:
        for area in localities:
            base = area.get("base_rent", 15000)
            rent_map[area["name"]] = fallback_estimate(
                base, input_data.preferred_bhk,
                input_data.property_size, input_data.furnishing_status
            )

    # ── Filter, distance, sort ──────────────────────────────────────
    suggestions = []
    for area in localities:
        rent = rent_map[area["name"]]
        if rent > input_data.max_budget:
            continue
        dist_km = geodesic((office_lat, office_lon), (area["lat"], area["lon"])).km
        suggestions.append({
            "locality":          area["name"],
            "predicted_rent":    rent,
            "distance_km":       round(dist_km, 1),
            "commute_time_mins": int((dist_km / 24) * 60) + 7,
        })

    suggestions = sorted(suggestions, key=lambda x: x["distance_km"])
    addr_parts  = location.address.split(",")
    workplace_display = ", ".join(p.strip() for p in addr_parts[:2]) if len(addr_parts) >= 2 else location.address

    return {
        "success":          True,
        "detected_city":    target_city,
        "model_used":       USE_MODEL and target_city in MODEL_KNOWN_CITIES,
        "searched_workplace": {
            "name": workplace_display,
            "lat":  office_lat,
            "lon":  office_lon,
        },
        "recommendations":  suggestions[:5],
    }
