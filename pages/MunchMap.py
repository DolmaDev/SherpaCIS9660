import streamlit as st
import pandas as pd
from geopy.geocoders import Nominatim
from munchmap_helpers import recommend  

st.title("🍽️ MunchMap — Restaurant Recommender")

# Location via US ZIP
zip_code = st.text_input("📍 ZIP Code (US Only)", placeholder="e.g., 10010")
latitude, longitude = None, None
if zip_code.strip():
    try:
        geolocator = Nominatim(user_agent="munchmap_app")
        loc = geolocator.geocode({"postalcode": zip_code.strip(), "country": "US"})
        if loc:
            latitude, longitude = loc.latitude, loc.longitude
            st.success(f"Found location: {loc.address}")
        else:
            st.error("Could not find that ZIP code in the US. Try another.")
    except Exception as e:
        st.error(f"Geocoding error: {e}")

col1, col2 = st.columns(2)

with col1:
    budget_max = st.number_input("Budget max ($/person) excluding tips and tax", value=30, step=1)

    moods_map = {
    "😋 Feeling adventurous": "fusion, gastropub",
    "🍔 Craving comfort food": "burgers, fried chicken, diner",
    "🍣 Sushi vibes": "sushi, japanese",
    "🌱 Plant-based today": "vegan, vegetarian",
    "🍕 Pizza night energy": "pizza",
    "🥗 Healthy & fresh": "salad, mediterranean",
    "🍩 Sweet tooth calling": "dessert, bakery, ice cream",
    "🍜 Noodle craving": "ramen, pho, noodles",
    "🌮 Taco & spice mood": "mexican, tacos",
    "🍷 Fancy dining": "fine dining, french, steakhouse",
    "🔥 Something bold & spicy": "sichuan, indian, thai",
    "🧀 Cheesy goodness": "pizza, mac and cheese",
    "🍤 Seafood feast": "seafood, oysters",
    "🥩 Meat lover mode": "bbq, steakhouse",
    "🤷‍♂️ Surprise me!": "",
}

# Streamlit selectbox with just the keys as options
mood = st.selectbox("Mood", list(moods_map.keys()))

# Get the cuisine keywords for the chosen mood
cuisine_keywords = moods_map[mood]
with col2:
    cuisine = st.text_input("Cuisine keywords", value="dim sum")

# Radius in miles ➜ meters for API
radius_miles = st.slider("Radius (miles)", 0.1, 10.0, 1.0, 0.1)
radius_meters = radius_miles * 1609.34  #keeping meters for API

# Top results
top_k = st.slider("Top results to show", 3, 10, 5, 1)

# -Quick Summary helper-
def show_quick_summary(results, blurb_text: str = ""):
    st.markdown("## MunchMap Thinks")
    if blurb_text:
        st.write(blurb_text)

    # bullet list with direct links
    lines = []
    for r in results:
        name = r["name"]
        rating = r.get("rating", "N/A")
        price = r.get("price", "—")
        dist = r.get("distance_m", 0) or 0
        dist_label = f"{dist/1000:.1f} km" if dist >= 1000 else f"{int(dist)} m"
        maps_link = r.get("maps_link")
        website = r.get("website")

        link_bits = []
        if maps_link: link_bits.append(f"[Open in Google Maps]({maps_link})")
        if website:   link_bits.append(f"[Website]({website})")
        links_joined = " · ".join(link_bits) if link_bits else ""

        lines.append(f"**{name}** — {rating}★ · {price} · {dist_label}" + (f" · {links_joined}" if links_joined else ""))
    if lines:
        st.markdown("\n".join([f"- {ln}" for ln in lines]))

    # optional compact table
    with st.expander("See quick table"):
        cols = ["name", "rating", "price", "distance_m", "maps_link", "website"]
        df = pd.DataFrame([{k: r.get(k) for k in cols} for r in results])
        if not df.empty:
            # robust distance conversion
            df["distance_km"] = (df["distance_m"].fillna(0).astype(float)) / 1000.0
            st.dataframe(df[["name","rating","price","distance_km","maps_link","website"]], use_container_width=True)

# - Run search -
if st.button("Recommend top picks for me"):
    if latitude is None or longitude is None:
        st.error("Please enter a valid US ZIP code first.")
    else:
        try:
            out = recommend(
                lat=latitude,           
                lng=longitude,          
                budget_max=budget_max,
                mood=mood,
                cuisine=cuisine,
                radius_m=radius_meters, 
                top_k=top_k,
            )

            blurb = out["blurb"]
            results = out["results"]

            # KPI row (optional)
            st.markdown("---")
            colA, colB, colC = st.columns(3)
            with colA:
                st.metric("Top picks", len(results))
            with colB:
                avg_rating = (sum([(r.get("rating") or 0) for r in results]) / len(results)) if results else 0
                st.metric("Avg rating", f"{avg_rating:.2f}★")
            with colC:
                nearest_m = min([(r.get("distance_m") or 0) for r in results]) if results else 0
                st.metric("Nearest", f"{nearest_m/1000:.1f} km" if nearest_m >= 1000 else f"{int(nearest_m)} m")

            # ✅ Show Quick Summary (your requirement)
            st.markdown("---")
            show_quick_summary(results, blurb_text=blurb)

            # Cards (detailed view)
            st.markdown("---")
            st.subheader("Top Picks")
            for r in results:
                with st.container(border=True):
                    st.markdown(f"**{r['name']}** — {r.get('rating','N/A')}★ · {r['price']} · {r['distance_m']} m")
                    st.write(r.get("address") or "—")
                    link_line = []
                    link_line = []
                    if r.get("maps_link"):link_line.append(f"[Open Place Page]({r['maps_link']})")
                    if r.get("pin_link"):link_line.append(f"[View Pin]({r['pin_link']})")
                    if r.get("website"):link_line.append(f"[Website]({r['website']})")
                    if link_line:st.markdown(" · ".join(link_line))
                    if r.get("photo"): st.image(r["photo"], use_container_width=False, width=320)

        except Exception as e:
            st.error(f"Error: {e}")
