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
        {"name": "Saket", "lat": 28.5244, "lon": 77.2066}
    ],
    "Mumbai": [
        {"name": "Andheri West", "lat": 19.1176, "lon": 72.8339}
    ],
    "Bangalore": [
        {"name": "Whitefield", "lat": 12.9698, "lon": 77.7500}
    ],
    "Hyderabad": [
        {"name": "Gachibowli", "lat": 17.4401, "lon": 78.3489}
    ],
    "Kolkata": [
        {"name": "New Town", "lat": 22.5898, "lon": 88.4748}
    ],
    "Chennai": [
        {"name": "Velachery", "lat": 12.9802, "lon": 80.2227}
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
