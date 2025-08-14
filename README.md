# SherpaCIS9660 — MunchMap & Analytics Suite (Streamlit)

A multi-page Streamlit project combining:
- **MunchMap** — restaurant exploration / aI integrated suggestions
- **Sales Revenue Predictor** — simple ML pipeline for revenue prediction
- **Online Retail Visualizations** — EDA on the classic Online Retail Excel dataset
- **Tree-Nut Image Classifier** — image classification demo (almond/cashew/walnut)
---

## 1) Project Structure

SherpaCIS9660/
├─ pages/
│ ├─ About.py # Online Retail EDA
│ ├─ MunchMap.py # Restaurant explorer (Streamlit)
│ ├─ SalesRevenuePredictor.py # ML predictor (Streamlit)
│ └─ TreeNutClassifier.py # Image classifier (Streamlit)
├─ data/
│ ├─ Online Retail.xlsx # ~23.7 MB
│ ├─ munchmap_sources.csv # (optional; if your app uses a CSV)
│ └─ treenuts/
	│ ├─ almond/ *.jpg
	│ ├─ cashew/ *.jpg
	│ └─ walnut/ *.jpg
├─ models/
│ ├─ treenut_model.h5 # or .tflite (if you export TFLite)
│ └─ label_map.json # optional, if used
├─ requirements.txt
├─ .gitignore
└─ README.md
---

## 2) Prerequisites

- Python **3.12 recommended** (TensorFlow compatibility is smoother)
  - If you must use 3.13, the image classifier may require workarounds.
- Git
- (Optional) A virtual environment tool (`venv` is built into Python)
- For the Tree-Nut classifier:
  - Either **TensorFlow** (`tensorflow`) or
  - **TensorFlow-macOS** (`tensorflow-macos`) + **Metal plugin** (`tensorflow-metal`) on Apple Silicon

---

## 3) Local Setup & Running

# Clone & enter the repo
git clone https://github.com/DolmaDev/SherpaCIS9660.git
cd SherpaCIS9660

# Create & activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app (replace file as needed)
streamlit run pages/MunchMap.py
---

## 4 Deployment (Streamlit Cloud)

- Push your repo to GitHub (make sure requirements.txt is up to date).

- Go to Streamlit Cloud and create a new app.

- Link your GitHub repo and select the main file to run (e.g., pages/MunchMap.py).

- Under Advanced Settings, set Python version to 3.12 for TensorFlow compatibility.

-Click Deploy — your app will be live in minutes.

## 5) License & Credits

License:
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.


Credits:

Online Retail dataset from the UCI Machine Learning Repository.

Tree-Nut images from public domain sources (almond, cashew, walnut).

Restaurant data powered by the Google Places API (© Google, Terms of Service).

AI-powered features use the Gemini 2.5 FlashLite API (© Google DeepMind, Terms of Service).

Built with Streamlit, TensorFlow, and other open-source libraries listed in requirements.txt.