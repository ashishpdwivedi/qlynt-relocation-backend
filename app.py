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
        {"name":"Hauz Khas","lat":28.5494,"lon":77.2001},
        {"name":"Saket","lat":28.5244,"lon":77.2066},
        {"name":"Malviya Nagar","lat":28.5355,"lon":77.2100},
        {"name":"Green Park","lat":28.5585,"lon":77.2066},
        {"name":"Munirka","lat":28.5543,"lon":77.1734},
        {"name":"Katwaria Sarai","lat":28.5422,"lon":77.1837},
        {"name":"RK Puram","lat":28.5677,"lon":77.1760},
        {"name":"Vasant Kunj","lat":28.5387,"lon":77.1554},
        {"name":"Lajpat Nagar","lat":28.5677,"lon":77.2431},
        {"name":"Defence Colony","lat":28.5717,"lon":77.2336},
        {"name":"South Extension","lat":28.5688,"lon":77.2204},
        {"name":"Nehru Place","lat":28.5494,"lon":77.2512},
        {"name":"Kalkaji","lat":28.5496,"lon":77.2588},
        {"name":"Connaught Place","lat":28.6304,"lon":77.2177},
        {"name":"Karol Bagh","lat":28.6514,"lon":77.1907},
        {"name":"Rajouri Garden","lat":28.6415,"lon":77.1218},
        {"name":"Janakpuri","lat":28.6214,"lon":77.0878},
        {"name":"Uttam Nagar","lat":28.6206,"lon":77.0651},
        {"name":"Dwarka Sector 12","lat":28.5921,"lon":77.0461},
        {"name":"Dwarka Sector 21","lat":28.5512,"lon":77.0565},
        {"name":"Pitampura","lat":28.7033,"lon":77.1313},
        {"name":"Shalimar Bagh","lat":28.7141,"lon":77.1644},
        {"name":"Model Town","lat":28.7056,"lon":77.1905},
        {"name":"Rohini","lat":28.7041,"lon":77.1025},
        {"name":"Laxmi Nagar","lat":28.6304,"lon":77.2777},
        {"name":"Mayur Vihar","lat":28.6065,"lon":77.2942},
        {"name":"Preet Vihar","lat":28.6417,"lon":77.2922},
        {"name":"Shahdara","lat":28.6737,"lon":77.2890},
        {"name":"Nawada","lat":28.6212,"lon":77.0574},
        {"name":"Mukherjee Nagar","lat":28.7100,"lon":77.2050}
    ],
    "Noida": [{"name":f"Sector {i}","lat":28.55+(i*0.001),"lon":77.30+(i*0.001)} for i in range(1,31)],
    "Gurgaon": [{"name":f"Sector {i}","lat":28.42+(i*0.001),"lon":77.05+(i*0.001)} for i in range(1,31)],
    "Mumbai": [{"name":f"Mumbai Locality {i}","lat":19.00+(i*0.002),"lon":72.80+(i*0.001)} for i in range(1,31)],
    "Bangalore": [{"name":f"Bangalore Locality {i}","lat":12.90+(i*0.002),"lon":77.55+(i*0.002)} for i in range(1,31)],
    "Hyderabad": [{"name":f"Hyderabad Locality {i}","lat":17.35+(i*0.002),"lon":78.30+(i*0.002)} for i in range(1,31)],
    "Chennai": [{"name":f"Chennai Locality {i}","lat":12.90+(i*0.002),"lon":80.15+(i*0.002)} for i in range(1,31)],
    "Kolkata": [{"name":f"Kolkata Locality {i}","lat":22.45+(i*0.002),"lon":88.30+(i*0.002)} for i in range(1,31)],
    "Pune": [{"name":f"Pune Locality {i}","lat":18.45+(i*0.002),"lon":73.75+(i*0.002)} for i in range(1,31)],
    "Ahmedabad": [{"name":f"Ahmedabad Locality {i}","lat":23.00+(i*0.002),"lon":72.50+(i*0.002)} for i in range(1,31)]
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
