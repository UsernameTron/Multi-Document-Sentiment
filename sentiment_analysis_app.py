import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import openai
import streamlit as st
from transformers import pipeline, AutoTokenizer
import fitz  # PyMuPDF
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from bertopic import BERTopic
import base64
from fpdf import FPDF
from PyPDF2 import PdfMerger
from io import BytesIO

# ------------------------- Configuration -------------------------
st.set_page_config(layout="wide", page_icon="📈")
openai.api_key = st.secrets["OPENAI_API_KEY"]

# ------------------------- Model Initialization -------------------------
@st.cache_resource
def load_models():
    return {
        "sentiment": pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment",
            device=-1,
            return_all_scores=True,
            truncation=True,
            max_length=512
        ),
        "emotion": pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            device=-1,
            truncation=True,
            max_length=512
        ),
        "topic": BERTopic(verbose=True),
        "tokenizer": AutoTokenizer.from_pretrained(
            "cardiffnlp/twitter-roberta-base-sentiment",
            model_max_length=512
        )
    }

models = load_models()

# ------------------------- Text Processing Functions -------------------------
def map_sentiment(label):
    mapping = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}
    return mapping.get(label, "Unknown")

def extract_dates(text):
    date_pattern = r"\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4})\b"
    dates = re.findall(date_pattern, text, re.IGNORECASE)
    valid_dates = []
    for date in dates:
        try:
            parsed_date = datetime.strptime(date, "%Y-%m-%d").date()
            valid_dates.append(parsed_date)
        except ValueError:
            try:
                parsed_date = datetime.strptime(date, "%d-%m-%Y").date()
                valid_dates.append(parsed_date)
            except Exception:
                continue
    return valid_dates

def process_text_chunks(text, max_length=500, overlap=50):
    tokens = models["tokenizer"].encode(
        text, 
        add_special_tokens=True, 
        truncation=True, 
        max_length=512
    )[:512]
    chunks = []
    for i in range(0, len(tokens), max_length - overlap):
        chunk = tokens[i:i + max_length]
        chunks.append(models["tokenizer"].decode(chunk, skip_special_tokens=True))
    return chunks

# ------------------------- Analysis Functions -------------------------
def generate_gpt_insights(report_df, time_series=None):
    prompt = (
        f"Analyze this sentiment data and provide predictive insights:\n"
        f"- Sentiment distribution: {dict(report_df['Sentiment'].value_counts())}\n"
        f"- Average confidence: {report_df['Confidence'].mean():.2f}\n"
        f"- Time trends: {time_series.to_dict() if time_series is not None else 'No temporal data'}\n\n"
        f"Provide:\n"
        f"1. Key observed patterns\n"
        f"2. Potential future trends\n"
        f"3. Recommended actions\n"
        f"4. Risk factors to monitor"
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"GPT-4 analysis failed: {str(e)}")
        return None

def analyze_temporal_trends(texts, dates):
    if not dates or len(texts) == 0:
        return None
    try:
        time_df = pd.DataFrame({"date": dates, "text": texts}).sort_values("date")
        time_df = time_df.set_index("date").resample("W").agg({"text": lambda x: " ".join(x)}).reset_index()
        time_df = time_df[time_df["text"].str.strip().astype(bool)]
        if len(time_df) == 0:
            return None
        # Batch process temporal texts for sentiment
        sentiment_results = models["sentiment"](time_df["text"].tolist())
        sentiments = []
        for result in sentiment_results:
            if not isinstance(result, list):
                result = [result]
            main_score = max(result, key=lambda x: x['score'])['score']
            sentiments.append(main_score)
        time_df["sentiment"] = sentiments
        return time_df
    except Exception as e:
        st.error(f"Temporal analysis failed: {str(e)}")
        return None

def generate_topic_model(texts, n_topics=5):
    try:
        if len(texts) < 10:
            return None
        clean_texts = [txt for txt in texts if len(txt.strip()) > 50]
        if len(clean_texts) < 5:
            return None
        topics, _ = models["topic"].fit_transform(clean_texts)
        return models["topic"].get_topic_info().head(n_topics+1)
    except Exception as e:
        st.error(f"Topic modeling failed: {str(e)}")
        return None

# ------------------------- Visualization Functions -------------------------
def create_interactive_dashboard(report_df, time_df=None, topics=None):
    tab1, tab2, tab3, tab4 = st.tabs(["Sentiment", "Temporal", "Topics", "Emotions"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(report_df, x="Sentiment", color="Sentiment", title="Sentiment Distribution")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.pie(report_df, names="Sentiment", title="Sentiment Proportions")
            st.plotly_chart(fig, use_container_width=True)
        text = " ".join(report_df["Text"])
        if text:
            try:
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text[:10000])
                st.image(wordcloud.to_array(), caption="Frequent Words Cloud", use_container_width=True)
            except Exception as e:
                st.error(f"Word cloud generation failed: {str(e)}")
    
    with tab2:
        if time_df is not None:
            fig = px.line(time_df, x="date", y="sentiment", title="Sentiment Over Time", markers=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No temporal data available")
    
    with tab3:
        if topics is not None:
            fig = px.bar(topics, x="Count", y="Name", orientation='h', title="Key Topics Identified")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No topics identified")
    
    with tab4:
        try:
            truncated_texts = [text[:512] for text in report_df["Text"].tolist()]
            emotions = models["emotion"](truncated_texts)
            emotion_df = pd.DataFrame(emotions)
            fig = px.bar(emotion_df, x="label", y="score", color="label", title="Emotion Distribution")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Emotion analysis failed: {str(e)}")

# ------------------------- PDF Report Generation -------------------------
def generate_pdf_report(report_df, time_df=None, topics=None, title="Document Analysis Report"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Title Section
    pdf.cell(200, 10, txt=title, ln=1, align="C")
    pdf.ln(10)
    
    # Sentiment Summary
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Sentiment Analysis Summary", ln=1, align='L')
    pdf.set_font("Arial", size=12)
    
    sentiment_counts = report_df["Sentiment"].value_counts()
    for sentiment, count in sentiment_counts.items():
        pdf.cell(200, 10, txt=f"{sentiment}: {count} instances", ln=1)
    
    # Key Statistics
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Key Statistics", ln=1, align='L')
    pdf.set_font("Arial", size=12)
    
    pdf.cell(200, 10, txt=f"Dominant Sentiment: {report_df['Sentiment'].mode()[0]}", ln=1)
    pdf.cell(200, 10, txt=f"Average Confidence: {report_df['Confidence'].mean():.1%}", ln=1)
    pdf.cell(200, 10, txt=f"Text Complexity: {np.mean([len(text.split()) for text in report_df['Text']]):.0f} words/section", ln=1)
    
    # Temporal Trends
    if time_df is not None:
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt="Temporal Trends", ln=1, align='L')
        pdf.set_font("Arial", size=12)
        for _, row in time_df.iterrows():
            pdf.cell(200, 10, txt=f"{row['date'].strftime('%Y-%m-%d')}: Sentiment Score {row['sentiment']:.2f}", ln=1)
    
    # Topic Analysis
    if topics is not None:
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt="Key Topics", ln=1, align='L')
        pdf.set_font("Arial", size=12)
        for _, row in topics.head().iterrows():
            pdf.cell(200, 10, txt=f"{row['Name']}: {row['Count']} mentions", ln=1)
    
    return pdf.output(dest='S').encode('latin-1')

# ------------------------- Core Processing Pipeline -------------------------
def analyze_document(uploaded_file):
    try:
        if uploaded_file.type == "application/pdf":
            file_bytes = uploaded_file.read()
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                text = "\n".join([page.get_text() for page in doc])
        else:
            text = uploaded_file.read().decode("utf-8")
        
        dates = extract_dates(text)
        sections = [section.strip() for section in text.split("\n\n") if section.strip()]
        
        if not sections:
            st.warning("No analyzable text found in document")
            return None, None, None

        with st.spinner("Analyzing document..."):
            chunks = []
            section_indices = []
            for section_idx, section in enumerate(sections):
                section_chunks = process_text_chunks(section)
                chunks.extend(section_chunks)
                section_indices.extend([section_idx] * len(section_chunks))
            
            # Batch process sentiment for all chunks
            sentiment_results = models["sentiment"](chunks)
            sentiments = []
            for result in sentiment_results:
                if not isinstance(result, list):
                    result = [result]
                main_label = max(result, key=lambda x: x['score'])
                sentiments.append({
                    "label": map_sentiment(main_label['label']),
                    "score": main_label['score'],
                    "scores": {map_sentiment(item['label']): item['score'] for item in result}
                })

            # Aggregate sentiments back to sections
            from collections import defaultdict
            section_sentiments = defaultdict(list)
            for idx, sentiment in zip(section_indices, sentiments):
                section_sentiments[idx].append(sentiment)
            
            aggregated = []
            for idx in range(len(sections)):
                chunk_sentiments = section_sentiments.get(idx, [])
                if not chunk_sentiments:
                    aggregated.append({
                        "label": "Neutral",
                        "score": 0.5,
                        "scores": {"Negative": 0.0, "Neutral": 1.0, "Positive": 0.0}
                    })
                    continue
                avg_scores = {
                    "Negative": np.mean([c['scores'].get("Negative", 0) for c in chunk_sentiments]),
                    "Neutral": np.mean([c['scores'].get("Neutral", 0) for c in chunk_sentiments]),
                    "Positive": np.mean([c['scores'].get("Positive", 0) for c in chunk_sentiments])
                }
                main_label = max(avg_scores, key=avg_scores.get)
                aggregated.append({
                    "label": main_label,
                    "score": avg_scores[main_label],
                    "scores": avg_scores
                })

            report_df = pd.DataFrame({
                "Text": sections,
                "Sentiment": [s['label'] for s in aggregated],
                "Confidence": [s['score'] for s in aggregated],
                "Scores": [s['scores'] for s in aggregated]
            })
            
            time_df = analyze_temporal_trends(sections, dates)
            topics = generate_topic_model(sections) if len(sections) >= 10 else None
            
            return report_df, time_df, topics
            
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        return None, None, None

# ------------------------- Streamlit Interface -------------------------
def main():
    st.title("📊 Advanced Document Insights Analyzer")
    
    uploaded_files = st.file_uploader(
        "Upload documents (PDF/TXT)", 
        type=["pdf", "txt"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        results = []
        for idx, file in enumerate(uploaded_files):
            with st.expander(f"Document {idx+1} Analysis", expanded=True):
                report_df, time_df, topics = analyze_document(file)
                if report_df is not None:
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        create_interactive_dashboard(report_df, time_df, topics)
                    with col2:
                        st.subheader("AI-Powered Insights")
                        insights = generate_gpt_insights(report_df, time_df)
                        st.markdown(f"```\n{insights}\n```")
                        
                        st.subheader("Key Statistics")
                        st.metric("Overall Sentiment", report_df["Sentiment"].mode()[0])
                        st.metric("Average Confidence", f"{report_df['Confidence'].mean():.1%}")
                        st.metric(
                            "Text Complexity", 
                            f"{np.mean([len(text.split()) for text in report_df['Text']]):.0f} words/section"
                        )
                    results.append((report_df, time_df, topics))

        # PDF Report Generation in Sidebar
        with st.sidebar:
            st.header("Report Options")
            if st.button("Generate Comprehensive PDF Report"):
                if results:
                    merger = PdfMerger()
                    pdf_bytes_list = []
                    
                    for idx, (report_df, time_df, topics) in enumerate(results):
                        pdf_bytes = generate_pdf_report(
                            report_df, 
                            time_df, 
                            topics,
                            title=f"Document {idx+1} Analysis Report"
                        )
                        pdf_bytes_list.append(BytesIO(pdf_bytes))
                    
                    for pdf_bytes in pdf_bytes_list:
                        merger.append(pdf_bytes)
                    
                    combined_pdf = BytesIO()
                    merger.write(combined_pdf)
                    combined_pdf.seek(0)
                    
                    st.subheader("Download Reports")
                    for idx, pdf_bytes in enumerate(pdf_bytes_list):
                        pdf_bytes.seek(0)
                        b64 = base64.b64encode(pdf_bytes.read()).decode()
                        st.markdown(
                            f'<a href="data:application/pdf;base64,{b64}" download="document_{idx+1}_report.pdf">Download Document {idx+1} Report</a>',
                            unsafe_allow_html=True
                        )
                    
                    if len(results) > 1:
                        b64_combined = base64.b64encode(combined_pdf.getvalue()).decode()
                        st.markdown(
                            f'<a href="data:application/pdf;base64,{b64_combined}" download="combined_report.pdf">Download Combined Report</a>',
                            unsafe_allow_html=True
                        )
                else:
                    st.warning("No analysis results available")

if __name__ == "__main__":
    main()
