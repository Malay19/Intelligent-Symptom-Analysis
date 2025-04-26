# Intelligent Symptoms Analysis for Communicable Diseases

> **Empowering Early Detection Through AI and Explainable Machine Learning**

---

## 🌟 Project Overview

Intelligent Symptoms Analysis for Communicable Diseases is a comprehensive AI system designed to assist in the **early detection** of major communicable diseases purely based on **user-reported symptoms**. The project delivers:

1. **End-to-End Pipeline**  
   - **Data Cleaning & Fusion**: Ingests a raw CSV of 4,900+ records with 130+ symptoms, enriches with demographic (age, gender, region) and temporal (season) metadata, and outputs a unified dataset.  
   - **Feature Engineering**: Cleans text, lemmatizes, vectorizes via TF-IDF (n-grams), integrates symptom severity, and encodes auxiliary features.  
   - **Model Training**: Benchmarks classical (Random Forest, XGBoost) and deep architectures (Bidirectional LSTM, Transformer/BERT).  
   - **Explainability**: Implements **SHAP** for global feature importance and **LIME** for local decision explanations.  
   - **Deployment**: Exposes a Flask REST API and an interactive Streamlit UI for real-time symptom input, prediction, and visualization.

2. **High Accuracy & Trust**  
   - Achieves **94.6% accuracy** with the Transformer model, **F1-score 94.1%**, and **ROC-AUC 97.8%**.  
   - Interpretable outputs increase user and clinician confidence.

3. **Modularity & Extensibility**  
   - Clear separation of data, model, XAI, and UI layers allows easy addition of new diseases, languages, or data sources.

---

## 🧠 Motivation & Thought Process

- **Clinical Challenge**: Communicable diseases (e.g., COVID-19, Malaria, Dengue, TB) share overlapping early symptoms—fever, cough, fatigue—making manual triage error-prone.  
- **Resource Constraints**: Rural and low-resource settings face diagnostic delays due to lab/testing bottlenecks and clinician shortages.  
- **Technology Gap**: Existing symptom checkers rely on static, rule-based logic and provide no transparency, eroding trust among healthcare professionals.  
- **Research Vision**: Combine **deep learning** (to model complex symptom patterns) with **explainable AI** (to reveal decision logic), creating a **scalable**, **accessible**, and **transparent** tool that complements clinical workflows.

---

## 📂 Repository Structure

```plaintext
├── cleancsv.py                # Cleans and fuses the raw dataset into Better_Cleaned_Dataset.csv
├── model_pipeline.py          # Trains models, applies XAI, saves artifacts in Output/
├── disease_prediction_app.py  # Streamlit UI for symptom input, prediction, and explanations
├── Output/                    # Generated models, plots, and cleaned datasets
├── requirements.yml           # Conda environment spec (Python 3.9)
└── README.md                  # This documentation
```
---

## ⚙️ Installation

- Prerequisites: Python 3.9
- Create Environment:
```bash
pip install -r requirements.txt
```

## 🚀 Usage
- Data Cleaning
```bash
python cleancsv.py
Produces Better_Cleaned_Dataset.csv in Output/.
```
- Model Training & XAI

```bash
python model_pipeline.py
Trains models, evaluates performance, generates SHAP & LIME plots in Output/.
```

- Launch Web App

```bash
streamlit run disease_prediction_app.py
```
→ Open http://localhost:8501 to interact.

- Google Colab Deployment
```bash
!wget -q -O - ipv4.icanhazip.com        # Get public IP (used as password)
!streamlit run disease_prediction_app.py & npx localtunnel --port 8501
```
👉 [Colab Streamlit tutorial](https://www.youtube.com/watch?v=ZZsyxIWdCko)

## 📈 Methodology
- Text Preprocessing: Regex cleaning, spaCy/NLTK tokenization, stop-word removal, lemmatization.
- Feature Vectorization: TF-IDF (1–2 grams, 5,000 features) weighted by symptom severity.
- Modeling:
  - Random Forest, XGBoost, Gradient Boosting for baselines.  
  - Bidirectional LSTM (2 layers, hidden_size=128, dropout=0.3).
  - Transformer/BERT encoder with custom classification head.
- Explainability:
  - SHAP: TreeExplainer for feature importances; summary & force plots.
  - LIME: Tabular explainer for local interpretability.
- Deployment: Flask API + Streamlit UI.

## 🏥 Dataset
- Source: Kaggle – [Diseases and Symptoms Dataset](https://www.kaggle.com/datasets/dhivyeshrk/diseases-and-symptoms-dataset)
- Enhancements: Added severity scores, demographics, seasonal metadata.

## 📈 Results

| Metric         | Transformer | BiLSTM   | Random Forest |
|:---------------|------------:|:--------:|:-------------:|
| Accuracy       |      94.6%  |  92.4%   |     90.2%     |
| Precision      |      93.9%  |  91.8%   |     89.5%     |
| Recall         |      94.3%  |  90.5%   |     88.7%     |
| F1-Score       |      94.1%  |  91.1%   |     89.1%     |
| ROC-AUC        |      97.8%  |  96.3%   |     94.7%     |

- Inference Time: ~1.2s per prediction
- Top Features: Fever, Cough, Fatigue

## 📌 Key Highlights
- Hybrid Architecture: Combines deep sequence modeling (LSTM) and attention-based Transformers.
- Explainable AI: Every prediction is accompanied by SHAP & LIME visualizations.
- Interactive UI: Streamlit app allows dynamic symptom management and real-time explanations.

## 🚀 Future Work
- Multilingual NLP: Support Hindi, Spanish, Arabic symptom inputs.

- Mobile App: Android/iOS with offline capability.

- Wearable Integration: Incorporate real-time sensor data (HR, temperature).

- Clinical Trials: Partner with hospitals for validation.

## 📜 License
This project is licensed under the MIT License.

## 📧 **Contact & Contributions**
Feel free to contribute by submitting a pull request or reporting issues!

💡 **Author:** Malay Patel 
📬 **Email:** malayajay.patel@gmail.com
🔗 **GitHub:** https://github.com/Malay19
   
