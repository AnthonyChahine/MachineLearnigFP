"""
Streamlit Web Application for Intelligent Resume Screening System
"""

import os
import glob
import traceback
import re
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


# ==================== Constants ====================

# Keywords for resume analysis
MUST_HAVE_KEYWORDS = ["python", "sql", "excel"]
NICE_TO_HAVE_KEYWORDS = ["pandas", "power bi", "tableau", "statistics", 
                          "visualization", "scikit", "machine learning"]
ADDITIONAL_KEYWORDS = ["data", "analysis", "analyst", "dashboard", "reporting"]

# Combine all keywords for matching
ALL_KEYWORDS = MUST_HAVE_KEYWORDS + NICE_TO_HAVE_KEYWORDS + ADDITIONAL_KEYWORDS


# ==================== Reused Functions from main.py ====================

def read_text_file(path: str) -> str:
    """Read and return text content from a file."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()


def load_resumes(resume_dir: str):
    """Load all resume files from the specified directory."""
    paths = sorted(glob.glob(os.path.join(resume_dir, "*.txt")))
    if not paths:
        raise FileNotFoundError(f"No .txt resumes found in: {resume_dir}")

    resumes = []
    for p in paths:
        resumes.append(
            {
                "candidate_file": os.path.basename(p),
                "text": read_text_file(p),
            }
        )
    return pd.DataFrame(resumes)


def build_training_data(job_text: str, resumes_df: pd.DataFrame):
    """
    Create training data with labels using rule-based baseline.
    """
    labels = []
    for t in resumes_df["text"].str.lower().tolist():
        score = 0
        for k in MUST_HAVE_KEYWORDS:
            if k in t:
                score += 2
        for k in NICE_TO_HAVE_KEYWORDS:
            if k in t:
                score += 1

        # label: 1 = relevant, 0 = not relevant
        labels.append(1 if score >= 3 else 0)

    train_df = resumes_df.copy()
    train_df["label"] = labels

    # Add job description as extra context feature
    train_df["combined"] = (job_text + "\n\n" + train_df["text"]).astype(str)

    return train_df


def train_model(train_df: pd.DataFrame):
    """Train the machine learning model."""
    X = train_df["combined"].values
    y = train_df["label"].values

    # Handle small datasets safely
    stratify = y if len(set(y)) > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=stratify
    )

    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=2000)),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return model, (y_test, y_pred)


def score_and_rank(model, job_text: str, resumes_df: pd.DataFrame):
    """Score and rank resumes based on relevance."""
    combined = (job_text + "\n\n" + resumes_df["text"]).astype(str)
    # Probability of class 1 (relevant)
    proba = model.predict_proba(combined)[:, 1]

    out = resumes_df.copy()
    out["relevance_score"] = proba
    out = out.sort_values("relevance_score", ascending=False).reset_index(drop=True)
    return out


# ==================== New Helper Functions ====================

def categorize_candidate(score: float) -> str:
    """Categorize candidates based on their relevance score."""
    if score > 0.7:
        return "Highly Relevant"
    elif score >= 0.4:
        return "Relevant"
    else:
        return "Not Relevant"


def extract_matched_keywords(resume_text: str, job_text: str) -> list:
    """Extract keywords that appear in both resume and job description."""
    resume_lower = resume_text.lower()
    job_lower = job_text.lower()
    
    matched = []
    for keyword in ALL_KEYWORDS:
        if keyword in resume_lower and keyword in job_lower:
            matched.append(keyword)
    
    return matched


def highlight_keywords(text: str, keywords: list) -> str:
    """Highlight keywords in text using markdown."""
    highlighted_text = text
    for keyword in keywords:
        # Case-insensitive replacement with highlighting
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        highlighted_text = pattern.sub(f"**:{keyword.upper()}:**", highlighted_text)
    
    return highlighted_text


def get_score_color(score: float) -> str:
    """Return color code based on score."""
    if score > 0.7:
        return "green"
    elif score >= 0.4:
        return "orange"
    else:
        return "red"


def create_bar_chart(results_df: pd.DataFrame):
    """Create bar chart of candidate relevance scores."""
    fig = px.bar(
        results_df,
        x='candidate_file',
        y='relevance_score',
        title='Candidate Relevance Scores',
        labels={'candidate_file': 'Candidate', 'relevance_score': 'Relevance Score'},
        color='relevance_score',
        color_continuous_scale=['red', 'yellow', 'green']
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig


def create_pie_chart(results_df: pd.DataFrame):
    """Create pie chart showing distribution of recommendations."""
    recommendation_counts = results_df['recommendation'].value_counts()
    
    fig = go.Figure(data=[go.Pie(
        labels=recommendation_counts.index,
        values=recommendation_counts.values,
        marker=dict(colors=['#28a745', '#ffc107', '#dc3545'])
    )])
    
    fig.update_layout(title='Distribution of Recommendations')
    return fig


def get_top_features(model, n=10):
    """Get top TF-IDF features from the trained model."""
    try:
        tfidf = model.named_steps['tfidf']
        clf = model.named_steps['clf']
        
        feature_names = tfidf.get_feature_names_out()
        coefficients = clf.coef_[0]
        
        # Get top positive coefficients
        top_indices = coefficients.argsort()[-n:][::-1]
        top_features = [(feature_names[i], coefficients[i]) for i in top_indices]
        
        return top_features
    except Exception as e:
        return []


# ==================== Streamlit App ====================

def main():
    # Page configuration
    st.set_page_config(
        page_title="Resume Screening System",
        page_icon="🎯",
        layout="wide"
    )
    
    # Title
    st.title("🎯 Intelligent Resume Screening System")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 About")
        st.info(
            """
            This application uses Machine Learning and NLP to automatically 
            screen and rank resumes based on job requirements.
            """
        )
        
        st.header("📖 How to Use")
        st.markdown(
            """
            1. **Resume Screening**: Upload resumes and analyze them
            2. **View Samples**: Browse existing resume files
            3. **Model Performance**: Check ML model metrics
            4. **Test New Resume**: Upload individual resume for testing
            """
        )
        
        st.header("⚙️ Configuration")
        st.text("Model: Logistic Regression")
        st.text("Vectorizer: TF-IDF")
        st.text("Features: Unigrams + Bigrams")
    
    # Data paths
    job_path = os.path.join("data", "job_description.txt")
    resumes_dir = os.path.join("data", "resumes")
    
    # Load job description
    try:
        if os.path.exists(job_path):
            default_job_text = read_text_file(job_path)
        else:
            default_job_text = "We are hiring a Junior Data Analyst.\nRequired: Python, SQL, Excel, data visualization, basic statistics.\nNice to have: machine learning basics, pandas, Power BI or Tableau."
    except Exception as e:
        st.error(f"Error loading job description: {e}")
        default_job_text = ""
    
    # Initialize session state
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'train_df' not in st.session_state:
        st.session_state.train_df = None
    if 'eval_data' not in st.session_state:
        st.session_state.eval_data = None
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Resume Screening",
        "📄 View Sample Resumes",
        "📈 Model Performance",
        "🧪 Upload & Test New Resume"
    ])
    
    # ==================== Tab 1: Resume Screening ====================
    with tab1:
        st.header("Resume Screening & Analysis")
        
        # Job description input
        st.subheader("Job Description")
        job_text = st.text_area(
            "Edit the job description below:",
            value=default_job_text,
            height=150,
            help="This job description will be used to train the model and score resumes"
        )
        
        # File uploader
        st.subheader("Upload Resumes")
        uploaded_files = st.file_uploader(
            "Choose resume files (.txt format)",
            type=['txt'],
            accept_multiple_files=True,
            help="Upload one or more resume files in .txt format"
        )
        
        # Option to use existing resumes
        use_existing = st.checkbox("Use existing sample resumes from data/resumes folder", value=True)
        
        # Analyze button
        if st.button("🔍 Analyze Resumes", type="primary"):
            if not job_text.strip():
                st.error("Please provide a job description!")
            else:
                with st.spinner("Training model and analyzing resumes..."):
                    try:
                        # Load resumes
                        resumes_df = pd.DataFrame()
                        
                        if use_existing:
                            if os.path.exists(resumes_dir):
                                resumes_df = load_resumes(resumes_dir)
                        
                        if uploaded_files:
                            uploaded_resumes = []
                            for file in uploaded_files:
                                content = file.read().decode('utf-8', errors='ignore').strip()
                                uploaded_resumes.append({
                                    'candidate_file': file.name,
                                    'text': content
                                })
                            uploaded_df = pd.DataFrame(uploaded_resumes)
                            resumes_df = pd.concat([resumes_df, uploaded_df], ignore_index=True)
                        
                        if resumes_df.empty:
                            st.error("No resumes to analyze! Please upload files or check 'Use existing sample resumes'")
                        else:
                            # Train model
                            train_df = build_training_data(job_text, resumes_df)
                            model, eval_data = train_model(train_df)
                            
                            # Store in session state
                            st.session_state.model = model
                            st.session_state.train_df = train_df
                            st.session_state.eval_data = eval_data
                            
                            # Score and rank
                            results_df = score_and_rank(model, job_text, resumes_df)
                            
                            # Add recommendations and matched keywords
                            results_df['relevance_score_pct'] = (results_df['relevance_score'] * 100).round(2)
                            results_df['recommendation'] = results_df['relevance_score'].apply(categorize_candidate)
                            results_df['matched_keywords'] = results_df['text'].apply(
                                lambda x: ', '.join(extract_matched_keywords(x, job_text))
                            )
                            
                            st.success(f"✅ Analysis complete! Processed {len(results_df)} resumes.")
                            
                            # Display results table
                            st.subheader("📋 Results")
                            
                            display_df = results_df[['candidate_file', 'relevance_score_pct', 'recommendation', 'matched_keywords']].copy()
                            display_df.columns = ['Candidate', 'Score (%)', 'Recommendation', 'Matched Keywords']
                            
                            # Color-code the dataframe
                            def color_recommendation(val):
                                if val == 'Highly Relevant':
                                    return 'background-color: #d4edda'
                                elif val == 'Relevant':
                                    return 'background-color: #fff3cd'
                                else:
                                    return 'background-color: #f8d7da'
                            
                            styled_df = display_df.style.applymap(
                                color_recommendation,
                                subset=['Recommendation']
                            )
                            
                            st.dataframe(styled_df, use_container_width=True)
                            
                            # Download button
                            csv = results_df[['candidate_file', 'relevance_score_pct', 'recommendation', 'matched_keywords']].to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv,
                                file_name="ranked_candidates.csv",
                                mime="text/csv"
                            )
                            
                            # Visualizations
                            st.subheader("📊 Visualizations")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Bar chart
                                fig_bar = create_bar_chart(results_df)
                                st.plotly_chart(fig_bar, use_container_width=True)
                            
                            with col2:
                                # Pie chart
                                fig_pie = create_pie_chart(results_df)
                                st.plotly_chart(fig_pie, use_container_width=True)
                            
                    except Exception as e:
                        st.error(f"Error during analysis: {e}")
                        st.error(traceback.format_exc())
    
    # ==================== Tab 2: View Sample Resumes ====================
    with tab2:
        st.header("View Sample Resumes")
        
        try:
            if os.path.exists(resumes_dir):
                resume_files = sorted(glob.glob(os.path.join(resumes_dir, "*.txt")))
                
                if resume_files:
                    selected_file = st.selectbox(
                        "Select a resume to view:",
                        [os.path.basename(f) for f in resume_files]
                    )
                    
                    if selected_file:
                        file_path = os.path.join(resumes_dir, selected_file)
                        resume_content = read_text_file(file_path)
                        
                        st.subheader(f"📄 {selected_file}")
                        
                        # Show resume content
                        st.text_area("Resume Content:", resume_content, height=200)
                        
                        # Show keyword matches if job description is available
                        if job_text:
                            matched = extract_matched_keywords(resume_content, job_text)
                            if matched:
                                st.subheader("🔑 Matched Keywords")
                                st.write(", ".join(matched))
                                
                                # Highlighted version
                                st.subheader("Highlighted Resume")
                                highlighted = highlight_keywords(resume_content, matched)
                                st.markdown(highlighted)
                            else:
                                st.info("No keyword matches found with current job description")
                else:
                    st.warning("No resume files found in data/resumes folder")
            else:
                st.error(f"Resumes directory not found: {resumes_dir}")
        except Exception as e:
            st.error(f"Error loading resumes: {e}")
    
    # ==================== Tab 3: Model Performance ====================
    with tab3:
        st.header("Model Performance Metrics")
        
        if st.session_state.model is None:
            st.info("👈 Train a model first in the 'Resume Screening' tab")
        else:
            try:
                model = st.session_state.model
                y_test, y_pred = st.session_state.eval_data
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                
                # Calculate accuracy
                accuracy = (y_test == y_pred).mean()
                
                with col1:
                    st.metric("Accuracy", f"{accuracy:.2%}")
                
                with col2:
                    st.metric("Test Samples", len(y_test))
                
                with col3:
                    unique_labels = len(set(y_test))
                    st.metric("Classes", unique_labels)
                
                # Classification Report
                st.subheader("📊 Classification Report")
                report = classification_report(y_test, y_pred, digits=3)
                st.text(report)
                
                # Confusion Matrix
                st.subheader("📈 Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                
                fig, ax = plt.subplots(figsize=(6, 4))
                im = ax.imshow(cm, cmap='Blues')
                ax.set_xticks([0, 1])
                ax.set_yticks([0, 1])
                ax.set_xticklabels(['Not Relevant', 'Relevant'])
                ax.set_yticklabels(['Not Relevant', 'Relevant'])
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title('Confusion Matrix')
                
                # Add text annotations
                for i in range(2):
                    for j in range(2):
                        text = ax.text(j, i, cm[i, j],
                                     ha="center", va="center", color="black")
                
                plt.colorbar(im, ax=ax)
                st.pyplot(fig)
                
                # Feature Importance
                st.subheader("🔝 Top TF-IDF Features")
                top_features = get_top_features(model, n=15)
                
                if top_features:
                    feature_df = pd.DataFrame(top_features, columns=['Feature', 'Coefficient'])
                    
                    fig_features = px.bar(
                        feature_df,
                        x='Coefficient',
                        y='Feature',
                        orientation='h',
                        title='Top 15 Most Important Features',
                        labels={'Coefficient': 'Importance', 'Feature': 'Term'}
                    )
                    st.plotly_chart(fig_features, use_container_width=True)
                else:
                    st.info("Feature importance not available")
                
            except Exception as e:
                st.error(f"Error displaying metrics: {e}")
    
    # ==================== Tab 4: Upload & Test New Resume ====================
    with tab4:
        st.header("Upload & Test Individual Resume")
        
        if st.session_state.model is None:
            st.info("👈 Train a model first in the 'Resume Screening' tab")
        else:
            st.subheader("Upload a Single Resume for Quick Analysis")
            
            test_file = st.file_uploader(
                "Choose a resume file (.txt)",
                type=['txt'],
                key='test_uploader'
            )
            
            if test_file:
                try:
                    # Read the resume
                    resume_text = test_file.read().decode('utf-8', errors='ignore').strip()
                    
                    # Display resume
                    st.subheader(f"📄 {test_file.name}")
                    st.text_area("Resume Content:", resume_text, height=200)
                    
                    # Score the resume
                    test_df = pd.DataFrame([{'candidate_file': test_file.name, 'text': resume_text}])
                    results = score_and_rank(st.session_state.model, job_text, test_df)
                    
                    score = results['relevance_score'].iloc[0]
                    recommendation = categorize_candidate(score)
                    
                    # Display metrics
                    st.subheader("📊 Analysis Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Relevance Score",
                            f"{score * 100:.2f}%",
                            delta=None
                        )
                    
                    with col2:
                        st.metric(
                            "Recommendation",
                            recommendation,
                            delta=None
                        )
                    
                    with col3:
                        color = get_score_color(score)
                        st.markdown(f"**Status:** :{color}[{recommendation}]")
                    
                    # Matched keywords
                    matched_keywords = extract_matched_keywords(resume_text, job_text)
                    
                    if matched_keywords:
                        st.subheader("🔑 Matched Keywords")
                        st.write(", ".join(matched_keywords))
                        
                        # Highlighted resume
                        st.subheader("Highlighted Resume")
                        highlighted = highlight_keywords(resume_text, matched_keywords)
                        with st.expander("Show highlighted version"):
                            st.markdown(highlighted)
                    else:
                        st.warning("No keyword matches found with current job description")
                    
                except Exception as e:
                    st.error(f"Error analyzing resume: {e}")


if __name__ == "__main__":
    main()
