from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
import traceback
from geopy.distance import geodesic
from geopy.geocoders import Nominatim

app = FastAPI(title="Qlynt Pan-India Optimizer")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "final_model_v2.pkl")

# Load Model
try:
    model_pipeline = joblib.load(MODEL_PATH)
    print("🎯 Model loaded successfully!")
except Exception as e:
    print(f"❌ Load error: {e}")
    model_pipeline = None

# FULL 120-LOCATION MATRIX
INDIA_LOCALITIES_HUBS = {
    "Delhi": [
        {"name": "Hauz Khas", "lat": 28.5494, "lon": 77.2001}, {"name": "Saket", "lat": 28.5244, "lon": 77.2066},
        {"name": "Connaught Place", "lat": 28.6304, "lon": 77.2177}, {"name": "Katwaria Sarai", "lat": 28.5422, "lon": 77.1837},
        {"name": "Nawada", "lat": 28.6212, "lon": 77.0574}, {"name": "Laxmi Nagar", "lat": 28.6304, "lon": 77.2777},
        {"name": "Uttam Nagar", "lat": 28.6206, "lon": 77.0651}, {"name": "Janakpuri", "lat": 28.6214, "lon": 77.0878},
        {"name": "Dwarka", "lat": 28.5921, "lon": 77.0461}, {"name": "Rohini", "lat": 28.7041, "lon": 77.1025},
        {"name": "Vasant Kunj", "lat": 28.5387, "lon": 77.1554}, {"name": "Karol Bagh", "lat": 28.6514, "lon": 77.1907},
        {"name": "Okhla", "lat": 28.5411, "lon": 77.2831}, {"name": "Malviya Nagar", "lat": 28.5365, "lon": 77.2096},
        {"name": "Greater Kailash", "lat": 28.5484, "lon": 77.2333}, {"name": "Green Park", "lat": 28.5573, "lon": 77.2066},
        {"name": "Defence Colony", "lat": 28.5714, "lon": 77.2295}, {"name": "Punjabi Bagh", "lat": 28.6656, "lon": 77.1265},
        {"name": "Model Town", "lat": 28.7119, "lon": 77.1856}, {"name": "Paschim Vihar", "lat": 28.6644, "lon": 77.0988}
    ],
    "Mumbai": [
        {"name": "Andheri West", "lat": 19.1176, "lon": 72.8339}, {"name": "Bandra West", "lat": 19.0600, "lon": 72.8311},
        {"name": "Thane West", "lat": 19.2183, "lon": 72.9781}, {"name": "Dadar", "lat": 19.0178, "lon": 72.8478},
        {"name": "Borivali West", "lat": 19.2292, "lon": 72.8572}, {"name": "Malad West", "lat": 19.1860, "lon": 72.8485},
        {"name": "Kandivali West", "lat": 19.2046, "lon": 72.8522}, {"name": "Chembur", "lat": 19.0622, "lon": 72.8974},
        {"name": "Goregaon West", "lat": 19.1633, "lon": 72.8412}, {"name": "Powai", "lat": 19.1176, "lon": 72.9060},
        {"name": "Mira Road", "lat": 19.2817, "lon": 72.8561}, {"name": "Vashi", "lat": 19.0760, "lon": 72.9982},
        {"name": "Colaba", "lat": 18.9067, "lon": 72.8147}, {"name": "Worli", "lat": 19.0125, "lon": 72.8189},
        {"name": "Juhu", "lat": 19.1075, "lon": 72.8260}, {"name": "Santacruz", "lat": 19.0844, "lon": 72.8358},
        {"name": "Kharghar", "lat": 19.0270, "lon": 73.0658}, {"name": "Kurla", "lat": 19.0825, "lon": 72.8673},
        {"name": "Mulund", "lat": 19.1726, "lon": 72.9426}, {"name": "Ghatkopar", "lat": 19.0865, "lon": 72.9090}
    ],
    "Bangalore": [
        {"name": "Whitefield", "lat": 12.9698, "lon": 77.7500}, {"name": "Koramangala", "lat": 12.9352, "lon": 77.6244},
        {"name": "HSR Layout", "lat": 12.9102, "lon": 77.6450}, {"name": "Indiranagar", "lat": 12.9719, "lon": 77.6412},
        {"name": "Electronic City", "lat": 12.8452, "lon": 77.6602}, {"name": "Jayanagar", "lat": 12.9307, "lon": 77.5832},
        {"name": "Marathahalli", "lat": 12.9569, "lon": 77.7011}, {"name": "BTM Layout", "lat": 12.9166, "lon": 77.6101},
        {"name": "Yelahanka", "lat": 13.1007, "lon": 77.5963}, {"name": "Bellandur", "lat": 12.9304, "lon": 77.6784},
        {"name": "Hebbal", "lat": 13.0357, "lon": 77.5970}, {"name": "Sarjapur", "lat": 12.8715, "lon": 77.7854},
        {"name": "Banashankari", "lat": 12.9255, "lon": 77.5468}, {"name": "Malleshwaram", "lat": 13.0033, "lon": 77.5645},
        {"name": "Rajajinagar", "lat": 12.9982, "lon": 77.5530}, {"name": "CV Raman Nagar", "lat": 12.9863, "lon": 77.6677},
        {"name": "Ulsoor", "lat": 12.9774, "lon": 77.6212}, {"name": "Basavanagudi", "lat": 12.9419, "lon": 77.5755},
        {"name": "Frazer Town", "lat": 13.0019, "lon": 77.6108}, {"name": "Hennur", "lat": 13.0456, "lon": 77.6366}
    ],
    "Hyderabad": [
        {"name": "Gachibowli", "lat": 17.4401, "lon": 78.3489}, {"name": "Hitech City", "lat": 17.4483, "lon": 78.3741},
        {"name": "Kukatpally", "lat": 17.4855, "lon": 78.3973}, {"name": "Jubilee Hills", "lat": 17.4367, "lon": 78.4065},
        {"name": "Madhapur", "lat": 17.4485, "lon": 78.3908}, {"name": "Manikonda", "lat": 17.3977, "lon": 78.3916},
        {"name": "Kondapur", "lat": 17.4622, "lon": 78.3568}, {"name": "Banjara Hills", "lat": 17.4174, "lon": 78.4414},
        {"name": "Ameerpet", "lat": 17.4374, "lon": 78.4482}, {"name": "Miyapur", "lat": 17.4933, "lon": 78.3512},
        {"name": "Begumpet", "lat": 17.4437, "lon": 78.4526}, {"name": "Uppal", "lat": 17.4048, "lon": 78.5607},
        {"name": "Kokapet", "lat": 17.3916, "lon": 78.3243}, {"name": "Nanakramguda", "lat": 17.4116, "lon": 78.3305},
        {"name": "Kothaguda", "lat": 17.4610, "lon": 78.3650}, {"name": "Tellapur", "lat": 17.5100, "lon": 78.2700},
        {"name": "Attapur", "lat": 17.3750, "lon": 78.4230}, {"name": "Sainikpuri", "lat": 17.4855, "lon": 78.5360},
        {"name": "Dilsukhnagar", "lat": 17.3670, "lon": 78.5280}, {"name": "Gachibowli Outer", "lat": 17.4260, "lon": 78.3500}
    ],
    "Kolkata": [
        {"name": "Salt Lake", "lat": 22.5804, "lon": 88.4143}, {"name": "New Town", "lat": 22.5898, "lon": 88.4748},
        {"name": "Garia", "lat": 22.4650, "lon": 88.3903}, {"name": "Ballygunge", "lat": 22.5273, "lon": 88.3664},
        {"name": "Bandel", "lat": 22.9231, "lon": 88.3789}, {"name": "Dumdum", "lat": 22.6178, "lon": 88.4031},
        {"name": "Jadavpur", "lat": 22.4994, "lon": 88.3694}, {"name": "Tollygunge", "lat": 22.4934, "lon": 88.3475},
        {"name": "Rajarhat", "lat": 22.6152, "lon": 88.4619}, {"name": "Behala", "lat": 22.4950, "lon": 88.3194},
        {"name": "Lake Gardens", "lat": 22.4980, "lon": 88.3530}, {"name": "Kasba", "lat": 22.5110, "lon": 88.3840},
        {"name": "Howrah", "lat": 22.5958, "lon": 88.2636}, {"name": "Baranagar", "lat": 22.6450, "lon": 88.3650},
        {"name": "Kaikhali", "lat": 22.6280, "lon": 88.4250}, {"name": "Santoshpur", "lat": 22.4860, "lon": 88.3850},
        {"name": "Park Street", "lat": 22.5530, "lon": 88.3510}, {"name": "Alipore", "lat": 22.5310, "lon": 88.3310},
        {"name": "Baguiati", "lat": 22.6180, "lon": 88.4120}, {"name": "Gariahat", "lat": 22.5150, "lon": 88.3680}
    ],
    "Chennai": [
        {"name": "Velachery", "lat": 12.9802, "lon": 80.2227}, {"name": "Adyar", "lat": 13.0012, "lon": 80.2565},
        {"name": "Anna Nagar", "lat": 13.0850, "lon": 80.2101}, {"name": "Thoraipakkam", "lat": 12.9430, "lon": 80.2343},
        {"name": "Tambaram", "lat": 12.9229, "lon": 80.1275}, {"name": "Adambakkam", "lat": 12.9892, "lon": 80.2078},
        {"name": "Thiruvanmiyur", "lat": 12.9830, "lon": 80.2594}, {"name": "Medavakkam", "lat": 12.9191, "lon": 80.1932},
        {"name": "Perungudi", "lat": 12.9654, "lon": 80.2414}, {"name": "Porur", "lat": 13.0382, "lon": 80.1565},
        {"name": "OMR", "lat": 12.9200, "lon": 80.2300}, {"name": "Kodambakkam", "lat": 13.0510, "lon": 80.2210},
        {"name": "Mylapore", "lat": 13.0330, "lon": 80.2640}, {"name": "Besant Nagar", "lat": 12.9990, "lon": 80.2670},
        {"name": "Sholinganallur", "lat": 12.9000, "lon": 80.2270}, {"name": "Pallavaram", "lat": 12.9570, "lon": 80.1480},
        {"name": "Ambattur", "lat": 13.1140, "lon": 80.1500}, {"name": "Chromepet", "lat": 12.9490, "lon": 80.1430},
        {"name": "Guindy", "lat": 13.0060, "lon": 80.2200}, {"name": "T. Nagar", "lat": 13.0400, "lon": 80.2330}
    ]
}

class RelocationInput(BaseModel):
    office_location: str
    max_budget: float
    preferred_bhk: int
    property_size: float
    furnishing_status: str
    bathrooms: int
    area_type: str
    tenant_preference: str

@app.post("/optimize")
async def optimize(input_data: RelocationInput):
    if model_pipeline is None:
        raise HTTPException(status_code=500, detail="Model offline.")
    
    try:
        geolocator = Nominatim(user_agent="qlynt_final_v3")
        loc = geolocator.geocode(f"{input_data.office_location}, India", timeout=10)
        
        target_city = "Delhi"
        addr = loc.address.lower() if loc else ""
        for city in INDIA_LOCALITIES_HUBS.keys():
            if city.lower() in addr:
                target_city = city
                break
        
        hubs = INDIA_LOCALITIES_HUBS.get(target_city, INDIA_LOCALITIES_HUBS["Delhi"])
        
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
        preds = model_pipeline.predict(df)
        
        res = []
        for i, hub in enumerate(hubs):
            rent = int(preds[i])
            if rent <= input_data.max_budget:
                dist = geodesic((loc.latitude, loc.longitude), (hub["lat"], hub["lon"])).km if loc else 0
                res.append({"locality": hub["name"], "predicted_rent": rent, "distance_km": round(dist, 1)})
        
        return {"success": True, "recommendations": sorted(res, key=lambda x: x['distance_km'])[:3]}
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
