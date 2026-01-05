# Intelligent Resume Screening System (ML + NLP)

This project implements an Intelligent Resume Screening and Ranking System using Python, Natural Language Processing (NLP), and Machine Learning.
It automatically analyzes resumes and ranks candidates based on their relevance to a given job description.

## Project Features
- 🎯 **Interactive Web UI** built with Streamlit
- 📊 Resume analysis using TF-IDF vectorization
- 🤖 Supervised learning with Logistic Regression
- 📈 Automatic candidate ranking with relevance scores
- 📉 Model evaluation metrics (precision, recall, F1-score)
- 📁 Support for multiple resume uploads
- 🔍 Keyword matching and highlighting
- 📊 Interactive visualizations (bar charts, pie charts)
- 💾 Export results to CSV
- 🖥️ Both CLI and Web UI interfaces

## Project Structure
```
resume_screening_ml/
├─ data/
│  ├─ job_description.txt
│  └─ resumes/
├─ src/
│  ├─ main.py          # CLI version
│  └─ app.py           # Web UI version
├─ outputs/
├─ requirements.txt
└─ README.md
```

## Installation

### Prerequisites
- Python 3.8 or higher

### Install Dependencies
```bash
pip install streamlit pandas scikit-learn matplotlib plotly
```

Or install from requirements.txt:
```bash
pip install -r requirements.txt
```

## How to Run

### Option 1: Web UI (Recommended)
Launch the interactive Streamlit web application:
```bash
streamlit run src/app.py
```

Then open your browser to the URL shown (typically http://localhost:8501)

#### Web UI Features:
- **Resume Screening Tab**: Upload and analyze multiple resumes, view rankings, and download results
- **View Sample Resumes Tab**: Browse existing resume files with keyword highlighting
- **Model Performance Tab**: View classification metrics, confusion matrix, and feature importance
- **Upload & Test New Resume Tab**: Quickly test individual resumes with instant feedback

### Option 2: CLI Version
Run the command-line interface:
```bash
python src/main.py
```

This will:
- Train the model on sample resumes
- Generate evaluation metrics in `outputs/evaluation.txt`
- Create ranked candidate list in `outputs/ranked_candidates.csv`

## Usage Guide

### Using the Web Interface
1. **Launch the app**: `streamlit run src/app.py`
2. **Edit job description**: Modify the job requirements in the text area (or use the default)
3. **Upload resumes**: Upload your own .txt files or use existing samples
4. **Analyze**: Click "Analyze Resumes" to train the model and get results
5. **Review results**: View sortable table with scores, recommendations, and matched keywords
6. **Download**: Export results as CSV for further analysis
7. **Explore**: Check model performance metrics and feature importance

### Using the CLI
1. Place resume files (.txt) in `data/resumes/` folder
2. Update `data/job_description.txt` with your job requirements
3. Run `python src/main.py`
4. Check results in the `outputs/` folder

## How It Works
1. **Data Preparation**: Resumes are loaded and combined with job description text
2. **Feature Extraction**: TF-IDF vectorization extracts important terms
3. **Training**: Logistic Regression model learns from labeled data
4. **Scoring**: Each resume gets a relevance score (0-100%)
5. **Ranking**: Candidates are ranked by relevance and categorized:
   - **Highly Relevant**: Score > 70%
   - **Relevant**: Score 40-70%
   - **Not Relevant**: Score < 40%

## Customization
- Modify keywords in `build_training_data()` function to match your domain
- Adjust score thresholds in `categorize_candidate()` function
- Add more features to the model pipeline
- Customize visualizations and UI components

