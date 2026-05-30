from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
import os
from geopy.distance import geodesic
from geopy.geocoders import Nominatim

# ─────────────────────────────────────────────
# 1. Initialize FastAPI App
# ─────────────────────────────────────────────
app = FastAPI(
    title="Qlynt Pan-India Relocation Optimizer API",
    description="ML-powered real-time commute & rent optimizer for Indian cities.",
    version="4.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# 2. Load XGBoost Model (optional - graceful fallback)
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "final_model_v2.pkl")

try:
    with open(MODEL_PATH, 'rb') as f:
        model_pipeline = pickle.load(f)
    print("✅ Model pipeline loaded successfully!")
    USE_MODEL = True
except Exception as e:
    print(f"⚠️  Model not found or failed to load: {e}")
    print("📊 Falling back to locality-based rent estimation engine.")
    model_pipeline = None
    USE_MODEL = False

# ─────────────────────────────────────────────
# 3. Locality Hub Matrix with Base Rent Benchmarks
#    Each locality has: name, lat, lon, base_rent (unfurnished 1BHK baseline)
#    Rent scales by BHK, size, furnishing multipliers at prediction time.
# ─────────────────────────────────────────────
INDIA_LOCALITIES_HUBS = {
    "Delhi": [
        {"name": "Hauz Khas",         "lat": 28.5494, "lon": 77.2001, "base_rent": 22000},
        {"name": "Saket",             "lat": 28.5244, "lon": 77.2066, "base_rent": 20000},
        {"name": "Connaught Place",   "lat": 28.6304, "lon": 77.2177, "base_rent": 28000},
        {"name": "Katwaria Sarai",    "lat": 28.5422, "lon": 77.1837, "base_rent": 15000},
        {"name": "Nawada",            "lat": 28.6212, "lon": 77.0574, "base_rent": 10000},
        {"name": "Laxmi Nagar",       "lat": 28.6304, "lon": 77.2777, "base_rent": 12000},
        {"name": "Uttam Nagar",       "lat": 28.6206, "lon": 77.0651, "base_rent": 9500},
        {"name": "Janakpuri",         "lat": 28.6214, "lon": 77.0878, "base_rent": 13000},
        {"name": "Dwarka Sector 12",  "lat": 28.5921, "lon": 77.0461, "base_rent": 14000},
        {"name": "Rohini",            "lat": 28.7041, "lon": 77.1025, "base_rent": 12500},
        {"name": "Vasant Kunj",       "lat": 28.5387, "lon": 77.1554, "base_rent": 21000},
        {"name": "Karol Bagh",        "lat": 28.6514, "lon": 77.1907, "base_rent": 18000},
        {"name": "Okhla Phase 3",     "lat": 28.5411, "lon": 77.2831, "base_rent": 16000},
    ],
    "Mumbai": [
        {"name": "Andheri West",      "lat": 19.1176, "lon": 72.8339, "base_rent": 35000},
        {"name": "Bandra West",       "lat": 19.0600, "lon": 72.8311, "base_rent": 55000},
        {"name": "Thane West",        "lat": 19.2183, "lon": 72.9781, "base_rent": 22000},
        {"name": "Dadar",             "lat": 19.0178, "lon": 72.8478, "base_rent": 40000},
        {"name": "Borivali West",     "lat": 19.2292, "lon": 72.8572, "base_rent": 20000},
        {"name": "Malad West",        "lat": 19.1860, "lon": 72.8485, "base_rent": 25000},
        {"name": "Kandivali West",    "lat": 19.2046, "lon": 72.8522, "base_rent": 22000},
        {"name": "Chembur",           "lat": 19.0622, "lon": 72.8974, "base_rent": 28000},
        {"name": "Goregaon West",     "lat": 19.1633, "lon": 72.8412, "base_rent": 27000},
        {"name": "Powai",             "lat": 19.1176, "lon": 72.9060, "base_rent": 32000},
        {"name": "Mira Road East",    "lat": 19.2817, "lon": 72.8561, "base_rent": 16000},
    ],
    "Bangalore": [
        {"name": "Whitefield",        "lat": 12.9698, "lon": 77.7500, "base_rent": 22000},
        {"name": "Koramangala",       "lat": 12.9352, "lon": 77.6244, "base_rent": 28000},
        {"name": "HSR Layout",        "lat": 12.9102, "lon": 77.6450, "base_rent": 24000},
        {"name": "Indiranagar",       "lat": 12.9719, "lon": 77.6412, "base_rent": 30000},
        {"name": "Electronic City",   "lat": 12.8452, "lon": 77.6602, "base_rent": 14000},
        {"name": "Jayanagar",         "lat": 12.9307, "lon": 77.5832, "base_rent": 20000},
        {"name": "Marathahalli",      "lat": 12.9569, "lon": 77.7011, "base_rent": 18000},
        {"name": "BTM Layout",        "lat": 12.9166, "lon": 77.6101, "base_rent": 19000},
        {"name": "Yelahanka",         "lat": 13.1007, "lon": 77.5963, "base_rent": 13000},
        {"name": "Bellandur",         "lat": 12.9304, "lon": 77.6784, "base_rent": 21000},
    ],
    "Hyderabad": [
        {"name": "Gachibowli",        "lat": 17.4401, "lon": 78.3489, "base_rent": 20000},
        {"name": "Hitech City",       "lat": 17.4483, "lon": 78.3741, "base_rent": 22000},
        {"name": "Kukatpally",        "lat": 17.4855, "lon": 78.3973, "base_rent": 14000},
        {"name": "Jubilee Hills",     "lat": 17.4367, "lon": 78.4065, "base_rent": 28000},
        {"name": "Madhapur",          "lat": 17.4485, "lon": 78.3908, "base_rent": 21000},
        {"name": "Manikonda",         "lat": 17.3977, "lon": 78.3916, "base_rent": 15000},
        {"name": "Kondapur",          "lat": 17.4622, "lon": 78.3568, "base_rent": 18000},
        {"name": "Banjara Hills",     "lat": 17.4174, "lon": 78.4414, "base_rent": 30000},
        {"name": "Ameerpet",          "lat": 17.4374, "lon": 78.4482, "base_rent": 13000},
        {"name": "Miyapur",           "lat": 17.4933, "lon": 78.3512, "base_rent": 12000},
    ],
    "Kolkata": [
        {"name": "Salt Lake City Sector 2", "lat": 22.5804, "lon": 88.4143, "base_rent": 16000},
        {"name": "New Town",          "lat": 22.5898, "lon": 88.4748, "base_rent": 14000},
        {"name": "Garia",             "lat": 22.4650, "lon": 88.3903, "base_rent": 9000},
        {"name": "Ballygunge",        "lat": 22.5273, "lon": 88.3664, "base_rent": 20000},
        {"name": "Bandel",            "lat": 22.9231, "lon": 88.3789, "base_rent": 7000},
        {"name": "Dumdum Park",       "lat": 22.6178, "lon": 88.4031, "base_rent": 10000},
        {"name": "South Dum Dum",     "lat": 22.6111, "lon": 88.4011, "base_rent": 9500},
        {"name": "Jadavpur",          "lat": 22.4994, "lon": 88.3694, "base_rent": 12000},
        {"name": "Tollygunge",        "lat": 22.4934, "lon": 88.3475, "base_rent": 11000},
        {"name": "Rajarhat",          "lat": 22.6152, "lon": 88.4619, "base_rent": 13000},
        {"name": "Behala",            "lat": 22.4950, "lon": 88.3194, "base_rent": 9000},
    ],
    "Chennai": [
        {"name": "Velachery",         "lat": 12.9802, "lon": 80.2227, "base_rent": 16000},
        {"name": "Adyar",             "lat": 13.0012, "lon": 80.2565, "base_rent": 22000},
        {"name": "Anna Nagar",        "lat": 13.0850, "lon": 80.2101, "base_rent": 20000},
        {"name": "Thoraipakkam",      "lat": 12.9430, "lon": 80.2343, "base_rent": 14000},
        {"name": "Tambaram",          "lat": 12.9229, "lon": 80.1275, "base_rent": 10000},
        {"name": "Adambakkam",        "lat": 12.9892, "lon": 80.2078, "base_rent": 13000},
        {"name": "Thiruvanmiyur",     "lat": 12.9830, "lon": 80.2594, "base_rent": 18000},
        {"name": "Medavakkam",        "lat": 12.9191, "lon": 80.1932, "base_rent": 11000},
        {"name": "Perungudi",         "lat": 12.9654, "lon": 80.2414, "base_rent": 15000},
        {"name": "Porur",             "lat": 13.0382, "lon": 80.1565, "base_rent": 13000},
    ],
    "Pune": [
        {"name": "Baner",             "lat": 18.5590, "lon": 73.7868, "base_rent": 18000},
        {"name": "Hinjewadi",         "lat": 18.5913, "lon": 73.7389, "base_rent": 15000},
        {"name": "Kothrud",           "lat": 18.5074, "lon": 73.8077, "base_rent": 17000},
        {"name": "Viman Nagar",       "lat": 18.5679, "lon": 73.9143, "base_rent": 20000},
        {"name": "Wakad",             "lat": 18.5982, "lon": 73.7627, "base_rent": 16000},
        {"name": "Hadapsar",          "lat": 18.5018, "lon": 73.9260, "base_rent": 13000},
        {"name": "Aundh",             "lat": 18.5598, "lon": 73.8075, "base_rent": 19000},
        {"name": "Magarpatta",        "lat": 18.5099, "lon": 73.9285, "base_rent": 21000},
        {"name": "Katraj",            "lat": 18.4530, "lon": 73.8626, "base_rent": 10000},
        {"name": "Pimple Saudagar",   "lat": 18.6127, "lon": 73.7997, "base_rent": 15000},
    ],
}

# ─────────────────────────────────────────────
# 4. Rent Estimation Engine (used when model unavailable)
#    Formula: base_rent × bhk_multiplier × furnishing_multiplier × size_factor
# ─────────────────────────────────────────────
BHK_MULTIPLIERS   = {1: 1.0, 2: 1.6, 3: 2.3, 4: 3.1}
FURNISH_MULTIPLIERS = {
    "Unfurnished":    1.0,
    "Semi-Furnished": 1.18,
    "Fully Furnished": 1.38,
}

def estimate_rent(base_rent: int, bhk: int, size: float, furnishing: str) -> int:
    bhk_m     = BHK_MULTIPLIERS.get(bhk, 1.0 + (bhk - 1) * 0.55)
    furnish_m = FURNISH_MULTIPLIERS.get(furnishing, 1.0)
    # Size factor: baseline is 600 sqft; every 100 sqft above/below adjusts ±3%
    size_factor = 1.0 + ((size - 600) / 100) * 0.03
    size_factor = max(0.75, min(size_factor, 1.8))   # clamp extremes
    raw = base_rent * bhk_m * furnish_m * size_factor
    # Round to nearest ₹500 for realistic feel
    return int(round(raw / 500) * 500)

# ─────────────────────────────────────────────
# 5. Request Schema
# ─────────────────────────────────────────────
class RelocationInput(BaseModel):
    office_location: str
    max_budget: float
    preferred_bhk: int
    property_size: float
    furnishing_status: str

# ─────────────────────────────────────────────
# 6. Health Check
# ─────────────────────────────────────────────
@app.get("/")
def check_status():
    return {
        "status": "online",
        "coverage": list(INDIA_LOCALITIES_HUBS.keys()),
        "model_active": USE_MODEL
    }

# ─────────────────────────────────────────────
# 7. Core Optimization Route
# ─────────────────────────────────────────────
@app.post("/optimize")
def optimize_relocation(input_data: RelocationInput):

    # ── Geocode office location ──────────────────────────────────────
    try:
        geolocator = Nominatim(user_agent="qlynt_pan_india_agent_v4")
        location   = geolocator.geocode(f"{input_data.office_location}, India", timeout=10)
        if not location:
            raise HTTPException(status_code=404, detail="Could not pin location. Try a clearer landmark or area name.")
        office_lat     = location.latitude
        office_lon     = location.longitude
        resolved_addr  = location.address.lower()
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=503, detail="Geocoding timed out. Please try again in a moment.")

    # ── Detect city from resolved address ───────────────────────────
    target_city = None
    city_aliases = {
        "Delhi":     ["delhi", "new delhi"],
        "Mumbai":    ["mumbai", "bombay"],
        "Bangalore": ["bangalore", "bengaluru"],
        "Hyderabad": ["hyderabad"],
        "Kolkata":   ["kolkata", "calcutta"],
        "Chennai":   ["chennai", "madras"],
        "Pune":      ["pune", "pimpri"],
    }
    for city, aliases in city_aliases.items():
        if any(alias in resolved_addr for alias in aliases):
            target_city = city
            break

    # Extended NCR / metro fallbacks
    if not target_city:
        ncr_keywords = ["gurgaon", "gurugram", "noida", "faridabad", "ghaziabad"]
        if any(kw in resolved_addr for kw in ncr_keywords):
            target_city = "Delhi"
        else:
            target_city = "Delhi"   # global safety net

    localities = INDIA_LOCALITIES_HUBS.get(target_city, [])

    # ── Build predictions ────────────────────────────────────────────
    suggestions = []

    if USE_MODEL and model_pipeline:
        # XGBoost model path
        batch_records = [{
            "bhk":              input_data.preferred_bhk,
            "size":             input_data.property_size,
            "city":             target_city,
            "area_name":        area["name"],
            "furnishing_status": input_data.furnishing_status,
            "area_type":        "Super Area",
        } for area in localities]
        input_df = pd.DataFrame(batch_records)
        predicted_rents = model_pipeline.predict(input_df)
        rent_map = {area["name"]: int(np.round(predicted_rents[i])) for i, area in enumerate(localities)}
    else:
        # Fallback estimation engine
        rent_map = {
            area["name"]: estimate_rent(
                area["base_rent"],
                input_data.preferred_bhk,
                input_data.property_size,
                input_data.furnishing_status
            )
            for area in localities
        }

    # ── Distance + commute calc ──────────────────────────────────────
    for area in localities:
        predicted_rent = rent_map[area["name"]]
        if predicted_rent > input_data.max_budget:
            continue

        distance_km       = geodesic((office_lat, office_lon), (area["lat"], area["lon"])).km
        est_commute_mins  = int((distance_km / 24) * 60) + 7  # city avg 24 km/h

        suggestions.append({
            "locality":          area["name"],
            "predicted_rent":    predicted_rent,
            "distance_km":       round(distance_km, 1),
            "commute_time_mins": est_commute_mins,
        })

    suggestions = sorted(suggestions, key=lambda x: x["distance_km"])

    # Build a display name for the searched workplace
    addr_parts = location.address.split(",")
    workplace_display = ", ".join(p.strip() for p in addr_parts[:2]) if len(addr_parts) >= 2 else location.address

    return {
        "success": True,
        "detected_city":    target_city,
        "model_used":       USE_MODEL,
        "searched_workplace": {
            "name": workplace_display,
            "lat":  office_lat,
            "lon":  office_lon,
        },
        "recommendations": suggestions[:5],   # return top 5 closest within budget
    }
