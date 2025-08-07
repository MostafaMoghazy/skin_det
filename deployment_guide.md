# 🔬 SkinAI - AI-Powered Skin Condition Detection

A comprehensive Streamlit application for detecting skin conditions (skin cancer, eczema, vitiligo) using deep learning, with integrated doctor finder functionality for Egyptian governorates.

![SkinAI Banner](https://img.shields.io/badge/SkinAI-v1.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.13+-orange.svg)

## 🌟 Features

### 🎯 AI-Powered Detection
- **Advanced CNN Model**: Deep learning model with trained `skindisease.h5` model
- **Multi-Class Classification**: Detects skin cancer, eczema, and vitiligo
- **Real-time Analysis**: Instant predictions with confidence scores
- **Image Enhancement**: Automatic image preprocessing for better accuracy

### 🏥 Doctor Finder
- **Comprehensive Database**: Access to dermatologists across Egypt
- **Location-Based Search**: Find doctors in Cairo, Giza, Alexandria, and more
- **Detailed Information**: Doctor names, specialties, locations, and ratings
- **Export Functionality**: Download doctor lists in CSV format

### 📊 Interactive Dashboard
- **Beautiful UI**: Modern, responsive design with gradient themes
- **Visualization**: Interactive charts and graphs using Plotly
- **History Tracking**: Keep track of previous analyses
- **Mobile Friendly**: Responsive design for all devices

## 🚀 Streamlit Cloud Deployment

### Prerequisites
- GitHub account
- Streamlit Cloud account (free at share.streamlit.io)
- Your `skindisease.h5` model file

### Quick Deploy to Streamlit Cloud

1. **Upload to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit - SkinAI app"
   git branch -M main
   git remote add origin https://github.com/yourusername/skinai-app.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Connect your GitHub repository
   - Main file path: `app.py`
   - Click "Deploy!"

### Important Files for GitHub

Make sure you have these files in your repository:
- `app.py` - Main application
- `skindisease.h5` - Your trained model (must be in root directory)
- `requirements.txt` - Dependencies
- `model_utils.py` - Model utilities
- `doctor_scraper.py` - Doctor scraping functionality
- `.streamlit/config.toml` - Streamlit configuration

## 📁 Repository Structure for GitHub

```
skinai-app/
├── app.py                    # Main Streamlit application
├── skindisease.h5           # Your trained model (REQUIRED)
├── requirements.txt         # Python dependencies
├── model_utils.py          # Model utilities
├── doctor_scraper.py       # Doctor scraping
├── .streamlit/
│   └── config.toml         # Streamlit configuration
├── .gitignore             # Git ignore file
└── README.md             # This documentation
```

## 🔧 Local Testing (Optional)

If you want to test locally before deploying:

```bash
# Clone your repository
git clone https://github.com/yourusername/skinai-app.git
cd skinai-app

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py
```

## ⚠️ Important Notes for Deployment

### Model File
- **CRITICAL**: Your `skindisease.h5` file must be in the root directory
- The app is configured to automatically load this model
- File size limit on GitHub: 100MB (if larger, use Git LFS)

### GitHub Large File Storage (if model > 100MB)
```bash
# If your model is larger than 100MB
git lfs install
git lfs track "*.h5"
git add .gitattributes
git add skindisease.h5
git commit -m "Add large model file with LFS"
git push
```

### Streamlit Cloud Limits
- **Memory**: 1GB RAM limit
- **Storage**: Limited temporary storage
- **Processing**: CPU-only (no GPU)
- **Timeout**: 10-minute request timeout

## 🐛 Troubleshooting Streamlit Cloud

### Common Issues and Solutions

1. **Model Not Found Error**
   - Ensure `skindisease.h5` is in the repository root
   - Check file is properly committed to GitHub
   - Verify file path in code matches exact filename

2. **Memory Issues**
   ```python
   # If model is too large, you might need model optimization
   # Consider using TensorFlow Lite for smaller models
   ```

3. **Package Installation Errors**
   - Ensure all packages in `requirements.txt` are compatible
   - Remove version numbers if conflicts occur
   - Use the provided `requirements.txt` as-is

4. **Timeout Errors**
   - Streamlit Cloud has 10-minute timeout
   - Optimize model loading with caching
   - The app uses `@st.cache_resource` for efficiency

### App Not Loading?
Check these:
- ✅ Repository is public or you've given Streamlit access
- ✅ `app.py` exists in root directory
- ✅ `requirements.txt` is properly formatted
- ✅ `skindisease.h5` is in the repository
- ✅ All Python syntax is correct

## 🔒 Security and Privacy

### Medical Disclaimer
**This application is for educational purposes only. Always consult medical professionals for actual diagnosis.**

### Data Privacy
- No images are stored permanently
- No personal data collection
- All processing happens in memory
- Session data cleared on browser close

## 📊 Features Overview

### 1. **Home Page**
- Welcome dashboard with feature overview
- Platform statistics and metrics
- Quick navigation to main features

### 2. **Prediction Page**
- Image upload interface (JPG, PNG supported)
- Real-time AI analysis using your `skindisease.h5` model
- Confidence scores with interactive charts
- Medical recommendations based on predictions

### 3. **Doctor Finder**
- Select governorate (Cairo, Giza, Alexandria, etc.)
- Real-time scraping from Vezeeta.com
- Detailed doctor profiles with specialty and location
- Download doctor list as CSV

### 4. **History Page**
- Track previous predictions
- View analysis trends
- Export prediction history

### 5. **About Page**
- Application information
- Technology details
- Important medical disclaimers

## 🎨 Customization

The app is fully customized with:
- Medical-themed color scheme
- Professional gradient backgrounds
- Interactive charts and visualizations
- Mobile-responsive design
- Custom CSS styling

## 📈 Expected Performance

- **Model Loading**: ~3-5 seconds on first load
- **Prediction Time**: <2 seconds per image
- **Doctor Scraping**: 5-10 seconds per governorate
- **Memory Usage**: ~500MB during operation

## 🤝 Support

If you encounter issues:
1. Check the Streamlit Cloud logs
2. Verify all files are properly uploaded to GitHub
3. Ensure your model file is accessible
4. Check the GitHub repository structure

## 📞 Deployment Checklist

Before deploying to Streamlit Cloud, ensure:

- [ ] ✅ `skindisease.h5` model file is in repository root
- [ ] ✅ All required files are committed to GitHub
- [ ] ✅ Repository is public or Streamlit has access
- [ ] ✅ `requirements.txt` includes all dependencies
- [ ] ✅ `.streamlit/config.toml` is present
- [ ] ✅ `app.py` is in root directory
- [ ] ✅ No syntax errors in Python files

## 🚀 Deploy Now!

1. **Upload everything to GitHub**
2. **Go to share.streamlit.io**
3. **Connect your repository**
4. **Click Deploy**
5. **Your app will be live at: `https://yourapp.streamlit.app`**

---

**Ready to deploy? Just push to GitHub and connect to Streamlit Cloud - everything is configured and ready to go! 🎉**