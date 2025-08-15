# SherpaCIS9660 — AI-Powered Analytics Suite (Streamlit)

A multi-page Streamlit project combining:
- **MunchMap** — Restaurant exploration with AI-integrated suggestions
- **Sales Revenue Predictor** — ML pipeline for revenue prediction with XGBoost (R²=0.62)
- **Tree Nut Image Classifier** — Random Forest classifier with 88.9% accuracy

---

## 1) Project Structure

```
SherpaCIS9660/
├─ pages/
│  ├─ MunchMap.py              # Restaurant explorer
│  ├─ Proj1.py                 # Sales Revenue Predictor 
│  └─ TreeNutsImageClassifier.py # Random Forest image classifier
├─ data/
│  └─ Online Retail.xlsx       # 23.7 MB dataset for sales prediction
├─ Nuts_small/                 # Training image data (organized)
│  ├─ Almonds/                 # Almond training images
│  ├─ Cashews/                 # Cashew training images  
│  └─ Walnuts/                 # Walnut training images
├─ artifacts/                  # Build artifacts
├─ .venv/                      # Virtual environment
├─ .streamlit/                 # Streamlit configuration
├─ preprocessing&modeling_treenutclassifier.py # Original tree nut training notebook
├─ preprocessing&modeling_proj1.py # Sales prediction modeling process
├─ best_nut_model.pkl          # Main Random Forest model (522 KB)
├─ nut_scaler.pkl              # Feature scaler for model (2 KB)
├─ nut_model_rank1_random_forest.pkl    # Backup: Random Forest
├─ nut_model_rank2_logistic_regression.pkl # Backup: Logistic Regression
├─ nut_model_rank3_svm_(rbf).pkl        # Backup: SVM model
├─ nut_samples.png             # Training sample visualization (1.6 MB)
├─ nut_feature_extraction_visualization.png # Feature extraction demo (618 KB)
├─ Nuts_small.zip              # Compressed training data
├─ treenutclassifier.py        # Random Forest backend (4 KB)
├─ treenutclassifier_backup.py # Backup of previous version
├─ model_report.py             # Sales model evaluation utilities
├─ munchmap_helpers.py         # Restaurant search utilities
├─ keras_safe.py               # TensorFlow safety utilities
├─ main.py                     # Main navigation page (4 KB)
├─ requirements.txt            # Python dependencies
├─ train_local.py              # Local model training script
├─ resize_nuts.py              # Image preprocessing utilities
├─ .env                        # Environment variables (not tracked)
├─ .gitignore                  # Git ignore rules
└─ README.md                   # Project documentation
```

---

## 2) Prerequisites

- **Python 3.12** (recommended for compatibility)
- Git
- Virtual environment (`venv`)
- **API Keys** (add to `.env` file):
  - Google Places API (for MunchMap)
  - Gemini API (for AI features)

---

## 3) Local Setup & Running

```bash
# Clone & enter the repo
git clone https://github.com/DolmaDev/SherpaCIS9660.git
cd SherpaCIS9660

# Create & activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Create .env file with your API keys
echo "PLACES_API_KEY=your_google_places_key" > .env
echo "GEMINI_API_KEY=your_gemini_key" >> .env

# Run Streamlit app
streamlit run main.py
```

---

## 4) Features & Models

### **Tree Nut Image Classifier**
- **Model Type**: Random Forest with 100 trees
- **Test Accuracy**: 88.9%
- **Cross-validation**: 90.4% ± 2.5%
- **Features**: 71 extracted features per image
  - RGB color features (3)
  - Local Binary Pattern texture (10) 
  - Histogram of Oriented Gradients (50)
  - Edge detection features (4)
  - Statistical features (4)
- **Classes**: Almond, Cashew, Walnut
- **Training Data**: 270 images, tested on 90 images

### **Sales Revenue Predictor** 
- **Model Type**: XGBoost Regressor
- **Performance**: R² = 0.6198, MAE = 9,751
- **Features**: 7 engineered features including:
  - Previous day revenue (35% importance)
  - 7-day moving average (25% importance)
  - Seasonal patterns and trends
- **Data**: Online Retail dataset (23.7 MB)

### **MunchMap Restaurant Explorer**
- Google Places API integration
- AI-powered recommendations via Gemini
- Interactive location-based search
- Real-time restaurant data

---

## 5) Model Performance

### **Tree Nut Classifier Results:**
| Model | Test Accuracy | Cross-Validation |
|-------|---------------|------------------|
| **Random Forest** | **88.9%** | 90.4% ± 2.5% |
| Logistic Regression | 88.9% | 87.8% ± 4.5% |
| SVM (RBF) | 87.8% | 85.6% ± 3.2% |
| K-Nearest Neighbors | 77.8% | 70.0% ± 4.0% |
| Gaussian Naive Bayes | 77.8% | 83.3% ± 2.0% |

### **Per-Class Performance (Random Forest):**
| Class | Precision | Recall | F1-Score | Test Samples |
|-------|-----------|--------|----------|--------------|
| Almond | 0.93 | 0.87 | 0.90 | 30 |
| Cashew | 0.83 | 0.83 | 0.83 | 30 |
| Walnut | 0.91 | 0.97 | 0.94 | 30 |

### **Sales Prediction Results:**
| Model | MAE | RMSE | R² Score |
|-------|-----|------|----------|
| **XGBoost** | **9,752** | **13,452** | **0.6198** |
| Random Forest | 9,019 | 14,310 | 0.5697 |
| Gradient Boosting | 9,957 | 14,616 | 0.5511 |

---

## 6) Technical Implementation

### **Feature Extraction Pipeline (Tree Nut Classifier):**
1. **Image Preprocessing**: Resize to 64×64, convert to RGB
2. **Color Analysis**: Extract RGB channel means
3. **Texture Analysis**: Local Binary Pattern (LBP) histogram
4. **Shape Analysis**: Histogram of Oriented Gradients (HOG)
5. **Edge Detection**: Prewitt horizontal/vertical filters
6. **Statistical Analysis**: Grayscale mean, std, min, max
7. **Feature Scaling**: StandardScaler normalization
8. **Classification**: Random Forest with 100 estimators

### **Dependencies:**
```
streamlit>=1.36          # Web framework
scikit-learn            # Machine learning models
scikit-image            # Image feature extraction
pandas, numpy           # Data processing
matplotlib, seaborn     # Visualizations
requests                # API calls
geopy                   # Location services
plotly                  # Interactive plots
python-dotenv           # Environment variables
```

---

## 7) Deployment

### **Streamlit Cloud:**
1. Push repository to GitHub
2. Connect to Streamlit Cloud
3. Set Python version to 3.12
4. Add environment variables for API keys:
   - `PLACES_API_KEY`
   - `GEMINI_API_KEY`
5. Deploy with `main.py` as entry point

### **Local Development:**
- All trained models included (`.pkl` files)
- No GPU required (CPU-optimized Random Forest)
- Lightweight deployment (~1.5 MB total model files)

---

## 8) File Organization

### **Model Files:**
- `best_nut_model.pkl` - Main Random Forest classifier
- `nut_scaler.pkl` - Feature scaling parameters
- Backup models for comparison and fallback

### **Development & Training Files:**
- `preprocessing&modeling_treenutclassifier.py` - Complete tree nut classifier development
- `preprocessing&modeling_proj1.py` - Sales prediction model development  
- `train_local.py` - Local model training script
- `resize_nuts.py` - Image preprocessing utilities

### **Visualization Files:**
- `nut_samples.png` - Training data samples
- `nut_feature_extraction_visualization.png` - Feature extraction demo

### **Training Data:**
- `Nuts_small/` - 360 organized training images
- `Nuts_small.zip` - Compressed dataset backup

### **Backend Files:**
- `treenutclassifier.py` - Random Forest backend
- `model_report.py` - Sales model evaluation utilities
- `munchmap_helpers.py` - Restaurant search utilities

---

## 9) License & Credits

**License:** MIT License

**Data Sources:**
- Online Retail dataset: UCI Machine Learning Repository
- Tree nut images: 360 manually curated and organized images
- Restaurant data: Google Places API
- AI features: Google Gemini API

**Built With:**
- **Frontend**: Streamlit
- **ML Framework**: scikit-learn  
- **Image Processing**: scikit-image
- **APIs**: Google Places, Google Gemini
- **Deployment**: Streamlit Cloud

---

## 10) Contact & Support

For questions, issues, or contributions:
- GitHub Issues: [Create an issue](https://github.com/DolmaDev/SherpaCIS9660/issues)
- Repository: [https://github.com/DolmaDev/SherpaCIS9660](https://github.com/DolmaDev/SherpaCIS9660)

Built by **DolmaDev** | CIS 9660 Project | 2025