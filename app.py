from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
import os
from geopy.distance import geodesic
from geopy.geocoders import Nominatim

# 1. Initialize the FastAPI App
app = FastAPI(
    title="Qlynt Pan-India Relocation Optimizer API",
    description="Scalable machine learning engine running real-time commute constraints on your housing dataset.",
    version="3.0"
)

# Enable CORS so your frontend can communicate securely with this API from anywhere
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Resolve Dynamic Working Directories for Production/Cloud Deployment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "rent_optimizer_pipeline.pkl")

# Load the Trained XGBoost Pipeline Model
try:
    with open(MODEL_PATH, 'rb') as f:
        model_pipeline = pickle.load(f)
    print("🎯 Model pipeline loaded flawlessly!")
except Exception as e:
    print(f"❌ Initialization error: {e}")
    model_pipeline = None

# 3. Comprehensive Benchmark Matrix for Indian Localities
INDIA_LOCALITIES_HUBS = {
    "Delhi": [
        {"name": "Hauz Khas", "lat": 28.5494, "lon": 77.2001},
        {"name": "Saket", "lat": 28.5244, "lon": 77.2066},
        {"name": "Connaught Place", "lat": 28.6304, "lon": 77.2177},
        {"name": "Katwaria Sarai", "lat": 28.5422, "lon": 77.1837},
        {"name": "Nawada", "lat": 28.6212, "lon": 77.0574},
        {"name": "Laxmi Nagar", "lat": 28.6304, "lon": 77.2777},
        {"name": "Uttam Nagar", "lat": 28.6206, "lon": 77.0651},
        {"name": "Janakpuri", "lat": 28.6214, "lon": 77.0878},
        {"name": "Dwarka Sector 12", "lat": 28.5921, "lon": 77.0461},
        {"name": "Rohini", "lat": 28.7041, "lon": 77.1025},
        {"name": "Vasant Kunj", "lat": 28.5387, "lon": 77.1554},
        {"name": "Karol Bagh", "lat": 28.6514, "lon": 77.1907},
        {"name": "Okhla Phase 3", "lat": 28.5411, "lon": 77.2831}
    ],
    "Mumbai": [
        {"name": "Andheri West", "lat": 19.1176, "lon": 72.8339},
        {"name": "Bandra West", "lat": 19.0600, "lon": 72.8311},
        {"name": "Thane West", "lat": 19.2183, "lon": 72.9781},
        {"name": "Dadar", "lat": 19.0178, "lon": 72.8478},
        {"name": "Borivali West", "lat": 19.2292, "lon": 72.8572},
        {"name": "Malad West", "lat": 19.1860, "lon": 72.8485},
        {"name": "Kandivali West", "lat": 19.2046, "lon": 72.8522},
        {"name": "Chembur", "lat": 19.0622, "lon": 72.8974},
        {"name": "Goregaon West", "lat": 19.1633, "lon": 72.8412},
        {"name": "Powai", "lat": 19.1176, "lon": 72.9060},
        {"name": "Mira Road East", "lat": 19.2817, "lon": 72.8561}
    ],
    "Bangalore": [
        {"name": "Whitefield", "lat": 12.9698, "lon": 77.7500},
        {"name": "Koramangala", "lat": 12.9352, "lon": 77.6244},
        {"name": "HSR Layout", "lat": 12.9102, "lon": 77.6450},
        {"name": "Indiranagar", "lat": 12.9719, "lon": 77.6412},
        {"name": "Electronic City", "lat": 12.8452, "lon": 77.6602},
        {"name": "Jayanagar", "lat": 12.9307, "lon": 77.5832},
        {"name": "Marathahalli", "lat": 12.9569, "lon": 77.7011},
        {"name": "BTM Layout", "lat": 12.9166, "lon": 77.6101},
        {"name": "Yelahanka", "lat": 13.1007, "lon": 77.5963},
        {"name": "Bellandur", "lat": 12.9304, "lon": 77.6784}
    ],
    "Hyderabad": [
        {"name": "Gachibowli", "lat": 17.4401, "lon": 78.3489},
        {"name": "Hitech City", "lat": 17.4483, "lon": 78.3741},
        {"name": "Kukatpally", "lat": 17.4855, "lon": 78.3973},
        {"name": "Jubilee Hills", "lat": 17.4367, "lon": 78.4065},
        {"name": "Madhapur", "lat": 17.4485, "lon": 78.3908},
        {"name": "Manikonda", "lat": 17.3977, "lon": 78.3916},
        {"name": "Kondapur", "lat": 17.4622, "lon": 78.3568},
        {"name": "Banjara Hills", "lat": 17.4174, "lon": 78.4414},
        {"name": "Ameerpet", "lat": 17.4374, "lon": 78.4482},
        {"name": "Miyapur", "lat": 17.4933, "lon": 78.3512}
    ],
    "Kolkata": [
        {"name": "Salt Lake City Sector 2", "lat": 22.5804, "lon": 88.4143},
        {"name": "New Town", "lat": 22.5898, "lon": 88.4748},
        {"name": "Garia", "lat": 22.4650, "lon": 88.3903},
        {"name": "Ballygunge", "lat": 22.5273, "lon": 88.3664},
        {"name": "Bandel", "lat": 22.9231, "lon": 88.3789},
        {"name": "Dumdum Park", "lat": 22.6178, "lon": 88.4031},
        {"name": "South Dum Dum", "lat": 22.6111, "lon": 88.4011},
        {"name": "Jadavpur", "lat": 22.4994, "lon": 88.3694},
        {"name": "Tollygunge", "lat": 22.4934, "lon": 88.3475},
        {"name": "Rajarhat", "lat": 22.6152, "lon": 88.4619},
        {"name": "Behala", "lat": 22.4950, "lon": 88.3194}
    ],
    "Chennai": [
        {"name": "Velachery", "lat": 12.9802, "lon": 80.2227},
        {"name": "Adyar", "lat": 13.0012, "lon": 80.2565},
        {"name": "Anna Nagar", "lat": 13.0850, "lon": 80.2101},
        {"name": "Thoraipakkam", "lat": 12.9430, "lon": 80.2343},
        {"name": "Tambaram", "lat": 12.9229, "lon": 80.1275},
        {"name": "Adambakkam", "lat": 12.9892, "lon": 80.2078},
        {"name": "Thiruvanmiyur", "lat": 12.9830, "lon": 80.2594},
        {"name": "Medavakkam", "lat": 12.9191, "lon": 80.1932},
        {"name": "Perungudi", "lat": 12.9654, "lon": 80.2414},
        {"name": "Porur", "lat": 13.0382, "lon": 80.1565}
    ]
}

# Define the structure of the incoming request JSON
class RelocationInput(BaseModel):
    office_location: str
    max_budget: float
    preferred_bhk: int
    property_size: float
    furnishing_status: str

# 4. Route to Serve the Beautiful Frontend HTML Page Directly
@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    html_path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return """
    <body style="background-color:#020617; color:#f8fafc; font-family:sans-serif; text-align:center; padding-top:100px;">
        <h1 style="color:#34d399;">🎯 Qlynt Backend Active</h1>
        <p>Your model is online, but index.html was not found in the root directory yet.</p>
    </body>
    """

# 5. Core Optimization Route
@app.post("/optimize")
def optimize_relocation(input_data: RelocationInput):
    if not model_pipeline:
        raise HTTPException(status_code=500, detail="ML Core Engine offline.")

    try:
        geolocator = Nominatim(user_agent="qlynt_pan_india_agent_v3")
        geo_query = f"{input_data.office_location}, India"
        location = geolocator.geocode(geo_query, timeout=10)
        
        if not location:
            raise HTTPException(status_code=404, detail="Could not pin location. Try specifying a clearer address.")
        
        office_lat = location.latitude
        office_lon = location.longitude
        resolved_address = location.address.lower()
    except Exception as e:
        raise HTTPException(status_code=503, detail="Geocoding handshake timed out. Try executing again.")

    # 6. Detect which city context the user is looking for dynamically
    target_city = None
    for city in INDIA_LOCALITIES_HUBS.keys():
        if city.lower() in resolved_address:
            target_city = city
            break

    if not target_city:
        # Fallback context check for extended NCR territories
        if "gurgaon" in resolved_address or "noida" in resolved_address:
            target_city = "Delhi"
        else:
            target_city = "Delhi"  # System default safety net fallback

    localities_to_scan = INDIA_LOCALITIES_HUBS.get(target_city, [])
    suggestions = []

    # 7. Build dynamic prediction batch data frame for the XGBoost pipeline
    batch_records = [{
        'bhk': input_data.preferred_bhk,
        'size': input_data.property_size,
        'city': target_city,
        'furnishing_status': input_data.furnishing_status,
        'area_type': 'Super Area'
    } for _ in localities_to_scan]
    
    input_df = pd.DataFrame(batch_records)
    predicted_rents = model_pipeline.predict(input_df)

    # 8. Process Distance Matrix Metrics
    for idx, area in enumerate(localities_to_scan):
        distance_km = geodesic((office_lat, office_lon), (area["lat"], area["lon"])).km
        
        # Estimate commute time assuming average city transit speed flow
        est_commute_mins = int((distance_km / 24) * 60) + 7
        predicted_rent = int(np.round(predicted_rents[idx]))

        # Filter out anything that breaches the user's pocket allowance boundary
        if predicted_rent <= input_data.max_budget:
            suggestions.append({
                "locality": area["name"],
                "predicted_rent": predicted_rent,
                "distance_km": round(distance_km, 1),
                "commute_time_mins": est_commute_mins
            })

    # Sort results to present closest geographical layout matches first
    suggestions = sorted(suggestions, key=lambda x: x['distance_km'])

    return {
        "success": True,
        "detected_city": target_city,
        "searched_workplace": {
            "name": location.address.split(",")[0] + ", " + location.address.split(",")[1],
            "lat": office_lat,
            "lon": office_lon
        },
        "recommendations": suggestions[:3]
    }
