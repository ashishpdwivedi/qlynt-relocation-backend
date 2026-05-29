from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import pickle
import pandas as pd
import numpy as np
import os
from geopy.distance import geodesic
from geopy.geocoders import Nominatim

app = FastAPI()

# Enable CORS
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "final_model_v2.pkl")

# Load Model
try:
    with open(MODEL_PATH, 'rb') as f:
        model_pipeline = pickle.load(f)
    print("🎯 Model loaded!")
except Exception as e:
    print(f"❌ Load error: {e}")
    model_pipeline = None

# Localities Matrix (Keep your existing dictionary here)
INDIA_LOCALITIES_HUBS = {
    "Delhi": [{"name": "Hauz Khas", "lat": 28.5494, "lon": 77.2001}, {"name": "Saket", "lat": 28.5244, "lon": 77.2066}],
    "Mumbai": [{"name": "Andheri West", "lat": 19.1176, "lon": 72.8339}],
    "Bangalore": [{"name": "Whitefield", "lat": 12.9698, "lon": 77.7500}],
    "Hyderabad": [{"name": "Gachibowli", "lat": 17.4401, "lon": 78.3489}],
    "Kolkata": [{"name": "New Town", "lat": 22.5898, "lon": 88.4748}],
    "Chennai": [{"name": "Velachery", "lat": 12.9802, "lon": 80.2227}]
}

@app.get("/", response_class=HTMLResponse)
def serve_index():
    with open(os.path.join(BASE_DIR, "index.html"), "r") as f:
        return f.read()

@app.post("/optimize")
async def optimize(data: dict):
    if not model_pipeline:
        raise HTTPException(status_code=500, detail="Engine offline.")
    
    try:
        # Geocode
        geolocator = Nominatim(user_agent="qlynt_app")
        location = geolocator.geocode(data.get("office_location") + ", India", timeout=10)
        if not location: raise Exception("Location not found")
        
        # Determine city (simple matching)
        target_city = "Delhi" # default
        for city in INDIA_LOCALITIES_HUBS.keys():
            if city.lower() in location.address.lower():
                target_city = city
        
        # Predict
        hubs = INDIA_LOCALITIES_HUBS.get(target_city, INDIA_LOCALITIES_HUBS["Delhi"])
        df = pd.DataFrame([{
            'bhk': int(data.get("preferred_bhk")),
            'size': float(data.get("property_size")),
            'city': target_city,
            'furnishing_status': data.get("furnishing_status"),
            'area_type': 'Super Area'
        } for _ in hubs])
        
        preds = model_pipeline.predict(df)
        
        results = []
        for i, hub in enumerate(hubs):
            rent = int(preds[i])
            if rent <= float(data.get("max_budget")):
                results.append({
                    "locality": hub["name"],
                    "predicted_rent": rent,
                    "distance_km": round(geodesic((location.latitude, location.longitude), (hub["lat"], hub["lon"])).km, 1),
                    "commute_time_mins": int((geodesic((location.latitude, location.longitude), (hub["lat"], hub["lon"])).km / 24) * 60) + 7
                })
        
        return {"success": True, "searched_workplace": {"name": location.address.split(",")[0]}, "recommendations": sorted(results, key=lambda x: x['distance_km'])[:3]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
