# munchmap_helpers.py
from typing import List, Dict, Any, Optional
import os, math, requests
import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# --- Keys: prefer Streamlit Secrets; fallback to environment vars ---
def _sec(name: str) -> Optional[str]:
    try:
        return st.secrets.get(name)
    except Exception:
        return None

PLACES_API_KEY = (
    _sec("GOOGLE_MAPS_API_KEY") or _sec("PLACES_API_KEY") or
    os.getenv("GOOGLE_MAPS_API_KEY") or os.getenv("PLACES_API_KEY")
)
GEMINI_API_KEY = _sec("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

def _ensure_key():
    """Stop the app with a clear message if the key is missing."""
    if not PLACES_API_KEY:
        st.error("Google Places API key missing. Add GOOGLE_MAPS_API_KEY to Streamlit Secrets (or set env).")
        st.stop()
        
#---------geo + price helpers ----------
def in_usa(lat: float, lng: float) -> bool:
    conus = 24.396308 <= lat <= 49.384358 and -124.848974 <= lng <= -66.885444
    ak = 51.214183 <= lat <= 71.538800 and -179.148909 <= lng <= -129.979500
    hi = 18.910361 <= lat <= 22.235600 and -160.247100 <= lng <= -154.806773
    return conus or ak or hi

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0; p = math.pi/180
    dlat, dlon = (lat2-lat1)*p, (lon2-lon1)*p
    a = (math.sin(dlat/2)**2 + math.cos(lat1*p)*math.cos(lat2*p)*math.sin(dlon/2)**2)
    return 2*R*math.asin(math.sqrt(a))

def price_to_symbol(level: Optional[str]) -> str:
    m = {"PRICE_LEVEL_FREE":"$", "PRICE_LEVEL_INEXPENSIVE":"$", "PRICE_LEVEL_MODERATE":"$$",
         "PRICE_LEVEL_EXPENSIVE":"$$$", "PRICE_LEVEL_VERY_EXPENSIVE":"$$$$"}
    return m.get(level, "—")

def symbol_to_expected_range(symbol: str):
    return {"$":(0,15),"$$":(15,35),"$$$":(35,70),"$$$$":(70,999)}.get(symbol,(0,999))

def place_photo_url(photo_name: Optional[str]) -> Optional[str]:
    return None if not photo_name else f"https://places.googleapis.com/v1/{photo_name}/media?maxHeightPx=720&key={PLACES_API_KEY}"

# ---------- Google Places (v1) Nearby Search ----------
def places_nearby(lat: float, lng: float, radius_m: float, max_count: int = 20) -> List[Dict[str, Any]]:
    """
    Server-side call to Places API (v1) `places:searchNearby`.
    Requirements:
      - POST method
      - X-Goog-FieldMask header (required)
      - API key must NOT be referrer-restricted (use no app restriction; restrict by API)
    """
    _ensure_key()

    url = "https://places.googleapis.com/v1/places:searchNearby"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": PLACES_API_KEY,
        "X-Goog-FieldMask": ",".join([
            "places.id","places.displayName","places.rating","places.userRatingCount",
            "places.priceLevel","places.formattedAddress","places.location",
            "places.primaryType","places.types","places.websiteUri"
        ]),
    }
    payload = {
        "includedPrimaryTypes": ["restaurant"],
        "includedTypes": ["meal_takeaway", "cafe"],
        "maxResultCount": int(min(max_count, 20)),
        "locationRestriction": {
            "circle": {"center": {"latitude": float(lat), "longitude": float(lng)}, "radius": float(radius_m)}
        },
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        # Show helpful detail (common: 403 due to key restriction/billing)
        st.error(f"Places Nearby error {resp.status_code}: {resp.text[:600]}")
        raise
    return resp.json().get("places", [])

# ---------- Google Places (v1) Place Details ----------
def place_details(place_id: str) -> Dict[str, Any]:
    """
    Fetches details for a single place id.
    """
    _ensure_key()

    url = f"https://places.googleapis.com/v1/places/{place_id}"
    headers = {
        "X-Goog-Api-Key": PLACES_API_KEY,
        "X-Goog-FieldMask": ",".join([
            "id",
            "websiteUri",
            "currentOpeningHours",
            "internationalPhoneNumber",
            "photos.name",
            # add lightweight extras if you want:
            "rating",
            "userRatingCount",
        ]),
    }
    resp = requests.get(url, headers=headers, timeout=30)
    if not resp.ok:
        st.error(f"Place details error {resp.status_code}: {resp.text[:600]}")
        return {}
    return resp.json()
# ---------- Gemini (short blurb) ----------
def _gemini_call(prompt: str, model: str, max_tokens: int, temperature: float = 0.4):
    if not GEMINI_API_KEY:
        return False, 0, {"error":"Missing GEMINI_API_KEY"}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {"Content-Type":"application/json"}
    params = {"key": GEMINI_API_KEY}
    body = {
        "contents":[{"role":"user","parts":[{"text": prompt}]}],
        "generationConfig":{"temperature": temperature, "maxOutputTokens": max_tokens, "responseMimeType":"text/plain"}
    }
    r = requests.post(url, headers=headers, params=params, json=body, timeout=30)
    ok = r.ok
    try: data = r.json()
    except Exception: data = {"raw": r.text}
    return ok, r.status_code, data

def gemini_blurb(prompt: str, model: str = "gemini-2.5-flash") -> str:
    for cap in (120, 300, 500, 800):
        ok, status, data = _gemini_call(prompt, model, cap)
        if not ok: 
            continue
        cand = (data.get("candidates") or [None])[0]
        if cand:
            parts = (cand.get("content") or {}).get("parts") or []
            for p in parts:
                if isinstance(p, dict) and p.get("text"):
                    return p["text"].strip()
            if cand.get("finishReason") == "MAX_TOKENS":
                continue
    return "Hmm… based on your choices, I’d try the top-rated nearby spots that fit your budget and vibe."

# ---------- Scoring (max-budget only) ----------
def score_place(p, lat, lng, budget_max):
    rating = float(p.get("rating", 0.0) or 0.0)
    urc = float(p.get("userRatingCount", 0) or 0)
    loc = p.get("location", {})
    dist_m = haversine_m(lat, lng, loc.get("latitude", 0), loc.get("longitude", 0))
    symbol = price_to_symbol(p.get("priceLevel"))
    lo, hi = symbol_to_expected_range(symbol)
    # Only max budget matters; small penalty if place likely above max.
    budget_penalty = 0.5 if lo > float(budget_max or 0) else 0.0
    score = rating + (math.log10(urc + 1) * 0.2) - (dist_m/1000.0 * 0.12) - budget_penalty
    return score, dist_m

# ---------- Orchestrator (soft filters + fallbacks) ----------
def recommend(lat, lng, budget_min=0, budget_max=40, mood="", cuisine="", radius_m=1609.34, top_k=5):
    # budget_min kept for backward compat; ignored in scoring.
    if not in_usa(lat, lng):
        raise ValueError("Location must be within the USA.")

    # Enforce ≥ 1 mile
    radius_m = max(float(radius_m or 0), 1609.34)

    # Fetch nearby
    raw = places_nearby(lat, lng, radius_m, max_count=20)

    # Cuisine filter (broad; drop if empty)
    def cuisine_match(p):
        if not cuisine or not cuisine.strip(): 
            return True
        blob = " ".join([p["displayName"]["text"], p.get("formattedAddress",""), " ".join(p.get("types", []))]).lower()
        return all(k.strip().lower() in blob for k in cuisine.split(","))

    filtered = [p for p in raw if cuisine_match(p)]
    if len(filtered) == 0 and cuisine and cuisine.strip():
        filtered = raw[:]  # drop cuisine filter

    # Score/sort
    scored = []
    for p in filtered:
        sc, dist_m = score_place(p, lat, lng, budget_max)
        scored.append((sc, dist_m, p))
    scored.sort(key=lambda t: t[0], reverse=True)

    # If still too few, widen radius once
    if len(scored) < top_k:
        wider = radius_m * 2
        raw2 = places_nearby(lat, lng, wider, max_count=20)
        filtered2 = [p for p in raw2 if cuisine_match(p)] if (cuisine and cuisine.strip()) else raw2
        if len(filtered2) == 0 and cuisine and cuisine.strip():
            filtered2 = raw2[:]
        scored = []
        for p in filtered2:
            sc, dist_m = score_place(p, lat, lng, budget_max)
            scored.append((sc, dist_m, p))
        scored.sort(key=lambda t: t[0], reverse=True)

    # Build results
    results = []
    for sc, dist_m, p in scored[: max(top_k + 5, top_k)]:
        details = place_details(p["id"]) or {}
        photos = details.get("photos", [])
        photo_name = photos[0]["name"] if photos else None
        results.append({
        "name": p["displayName"]["text"],
        "rating": p.get("rating"),
        "user_ratings_total": p.get("userRatingCount"),
        "price": price_to_symbol(p.get("priceLevel")),
        "address": p.get("formattedAddress"),
        "distance_m": round(dist_m),
        "maps_link": f"https://www.google.com/maps/place/?q=place_id:{p['id']}",
        "pin_link":  f"https://www.google.com/maps/search/?api=1&query={lat},{lng}",
        "website": details.get("websiteUri"),
        "phone": details.get("internationalPhoneNumber"),
        "photo": place_photo_url(photo_name),
        "place_id": p["id"],
        "score": round(sc, 3),
        })
       
    top = results[:top_k]

    # Tiny prompt for Gemini
    slim = [f"{i}. {r['name']} — {r.get('rating','N/A')}★" for i, r in enumerate(top[:3], 1)]
    prompt = f"""You're a concise, friendly food guide.
Budget: up to ${budget_max:.0f} per person.
Mood: {mood or 'not specified'}; Cuisine: {cuisine or 'any'}.
Write two short sentences starting with "Hmm… based on your choices, I'd suggest…" Mention 2–3 picks by name. <50 words.

Options:
{chr(10).join(slim)}
"""
    blurb = gemini_blurb(prompt)
    return {"blurb": blurb, "results": top}
