
📊 Multi-Document Sentiment Analyzer

AI-powered tool for sentiment, emotion, and topic analysis across multiple PDF and text documents. Built with Streamlit, OpenAI, and NLP models to extract insights effortlessly.

🚀 Features

✅ Upload & Analyze PDFs/TXT – Extracts and processes text
✅ AI-Powered Sentiment Analysis – Uses RoBERTa for classification
✅ Emotion Detection – Identifies emotions via fine-tuned DistilRoBERTa
✅ Topic Modeling – BERTopic for unsupervised topic clustering
✅ Interactive Visuals – Sentiment trends, topic distributions, and word clouds
✅ Automated Reports – Export findings as a PDF report
✅ Secure & Optimized – Efficient processing with API security

🔧 Installation

1️⃣ Clone the Repository

git clone https://github.com/UsernameTron/Multi-Document-Sentiment.git
cd Multi-Document-Sentiment

2️⃣ Set Up a Virtual Environment (Optional)

python3 -m venv venv  
source venv/bin/activate  # macOS/Linux  
venv\Scripts\activate     # Windows  

3️⃣ Install Dependencies

pip install -r requirements.txt

4️⃣ Set Up OpenAI API Key
Create a .streamlit/secrets.toml file and add:

[secrets]
OPENAI_API_KEY = "your-api-key-here"

⚠ DO NOT share your API key publicly!

5️⃣ Run the App

streamlit run sentiment_analysis_app.py

📂 Project Structure

📁 Multi-Document-Sentiment/
 ├── 📄 sentiment_analysis_app.py  # Main Streamlit application  
 ├── 📄 requirements.txt           # Dependencies  
 ├── 📄 README.md                   # Documentation  
 ├── 📁 .streamlit/                 # API secrets (not committed)  
 ├── 📁 models/                      # Cached NLP models  

🛠 How It Works

1️⃣ Upload PDFs or TXT files
2️⃣ Sentiment & emotion analysis per document section
3️⃣ BERTopic extracts key topics
4️⃣ Interactive visualization of insights
5️⃣ Download a full PDF report

📸 Screenshots

(Insert images showcasing the interface, graphs, and key results.)

💡 Contributing

Want to improve this project? Fork the repo and submit a pull request! 🚀

📜 License

MIT License – Free to use and modify.

📩 Contact

For questions or feedback, reach out via GitHub Issues or LinkedIn.

⭐ Support the Project

If you find this useful, give it a ⭐ on GitHub to help others discover it!

🔮 Next Steps & Roadmap

🚀 Planned Enhancements:
🔹 Support for additional document formats
🔹 Real-time API integration for automated processing
🔹 Advanced NLP models for improved accuracy

