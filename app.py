from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
import os
import traceback
from geopy.distance import geodesic
from geopy.geocoders import Nominatim

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "final_model_v2.pkl")

# Load model
try:
    model_pipeline = joblib.load(MODEL_PATH)
    print("🎯 Model loaded successfully!")
    print("Model Type:", type(model_pipeline))
except Exception as e:
    print("❌ MODEL LOAD ERROR")
    traceback.print_exc()
    model_pipeline = None

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

@app.get("/", response_class=HTMLResponse)
def serve_index():
    with open(os.path.join(BASE_DIR, "index.html"), "r", encoding="utf-8") as f:
        return f.read()

@app.post("/optimize")
async def optimize(data: dict):

    if model_pipeline is None:
        raise HTTPException(status_code=500, detail="Model failed to load.")

    try:
        print("\n====================")
        print("NEW REQUEST RECEIVED")
        print("====================")
        print("Payload:", data)

        office_location = data.get("office_location", "")

        # Geocode
        geolocator = Nominatim(user_agent="qlynt_app_v5")

        print("Geocoding:", office_location)

        location = geolocator.geocode(
            office_location + ", India",
            timeout=10
        )

        print("Geocode Result:", location)

        if not location:
            raise Exception("Location not found")

        target_city = "Delhi"

        for city in INDIA_LOCALITIES_HUBS.keys():
            if city.lower() in location.address.lower():
                target_city = city
                break

        print("Detected City:", target_city)

        hubs = INDIA_LOCALITIES_HUBS.get(
            target_city,
            INDIA_LOCALITIES_HUBS["Delhi"]
        )

        # Prepare input dataframe
        rows = []

        for _ in hubs:
            rows.append({
                "bhk": int(data.get("preferred_bhk", 2)),
                "size": float(data.get("property_size", 900)),
                "city": target_city,
                "furnishing_status": data.get(
                    "furnishing_status",
                    "Semi-Furnished"
                ),
                "area_type": "Super Area"
            })

        df = pd.DataFrame(rows)

        print("\nINPUT DATAFRAME")
        print(df.head())
        print("Columns:", df.columns.tolist())
        print("Shape:", df.shape)

        # Prediction
        print("\nRUNNING PREDICTION...")

        preds = model_pipeline.predict(df)

        print("Prediction Successful")
        print(preds)

        results = []

        for i, hub in enumerate(hubs):

            rent = int(preds[i])

            if rent <= float(data.get("max_budget", 25000)):

                dist = geodesic(
                    (location.latitude, location.longitude),
                    (hub["lat"], hub["lon"])
                ).km

                results.append({
                    "locality": hub["name"],
                    "predicted_rent": rent,
                    "distance_km": round(dist, 1),
                    "commute_time_mins": int((dist / 24) * 60) + 7
                })

        return {
            "success": True,
            "searched_workplace": {
                "name": location.address.split(",")[0]
            },
            "recommendations": sorted(
                results,
                key=lambda x: x["distance_km"]
            )[:3]
        }

    except Exception as e:

        print("\n====================")
        print("FULL ERROR TRACEBACK")
        print("====================")

        traceback.print_exc()

        return {
            "success": False,
            "error": str(e),
            "error_type": str(type(e))
        }
