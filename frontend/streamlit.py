import streamlit as st
import requests
from PIL import Image
import io
from collections import Counter
import pandas as pd

# ----------------------------
# CONFIG
# ----------------------------
FLASK_API_URL = "http://localhost:5000"  # Change to your Flask server URL

st.set_page_config(page_title="YouTube Sentiment Insights", layout="wide")
st.title("📊 YouTube Sentiment Insights")
st.markdown("Paste comments with timestamps, and get predictions, chart, wordcloud, and trend graph in one go.")

# ----------------------------
# USER INPUT
# ----------------------------
comments_with_timestamps = st.text_area(
    "Enter comments with timestamps (comma separated):",
    placeholder="This video is amazing!, 2025-08-15 14:30:00\nNot worth my time, 2025-08-15 14:35:00",
    height=200
)

if st.button("Analyze All"):
    if not comments_with_timestamps.strip():
        st.warning("Please enter at least one comment with a timestamp.")
        st.stop()

    # Parse input into list of dicts
    comments_data = []
    for line in comments_with_timestamps.strip().split("\n"):
        try:
            text, timestamp = line.split(",", 1)
            comments_data.append({"text": text.strip(), "timestamp": timestamp.strip()})
        except ValueError:
            st.error("Each line must have a comment and timestamp separated by a comma.")
            st.stop()

    # ----------------------------
    # 1️⃣ Predict with timestamps
    # ----------------------------
    try:
        response = requests.post(f"{FLASK_API_URL}/predict_with_timestamps", json={"comments": comments_data})
        response.raise_for_status()
        prediction_results = response.json()
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.subheader("🔍 Sentiment Predictions")
    for item in prediction_results:
        st.write(f"**[{item['timestamp']}]** {item['comment']} → **Sentiment:** {item['sentiment']}")

    # ----------------------------
    # 2️⃣ Prepare sentiment counts
    # ----------------------------
    sentiments = [str(item['sentiment']) for item in prediction_results]
    sentiment_counts = Counter(sentiments)

    # ----------------------------
    # 3️⃣ Generate Chart
    # ----------------------------
    try:
        chart_response = requests.post(f"{FLASK_API_URL}/generate_chart", json={"sentiment_counts": sentiment_counts})
        if chart_response.status_code == 200:
            st.subheader("📊 Sentiment Distribution")
            chart_img = Image.open(io.BytesIO(chart_response.content))
            st.image(chart_img, use_container_width=True)
        else:
            st.error(f"Chart generation failed: {chart_response.text}")
    except Exception as e:
        st.error(f"Chart generation failed: {e}")

    # ----------------------------
    # 4️⃣ Generate Wordcloud
    # ----------------------------
    try:
        comments_list = [item['comment'] for item in prediction_results]
        wc_response = requests.post(f"{FLASK_API_URL}/generate_wordcloud", json={"comments": comments_list})
        if wc_response.status_code == 200:
            st.subheader("☁ Wordcloud")
            wc_img = Image.open(io.BytesIO(wc_response.content))
            st.image(wc_img, use_container_width=True)
        else:
            st.error(f"Wordcloud generation failed: {wc_response.text}")
    except Exception as e:
        st.error(f"Wordcloud generation failed: {e}")

    # ----------------------------
    # 5️⃣ Generate Trend Graph (with normalization)
    # ----------------------------
    try:
        # Normalize timestamps & sentiment values before sending
        df = pd.DataFrame(prediction_results)
        if 'timestamp' in df.columns and 'sentiment' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce').astype(str)
            df['sentiment'] = pd.to_numeric(df['sentiment'], errors='coerce').fillna(0).astype(int)

            trend_data = df[['comment', 'sentiment', 'timestamp']].to_dict(orient='records')

            trend_response = requests.post(f"{FLASK_API_URL}/generate_trend_graph", json={"sentiment_data": trend_data})

            if trend_response.status_code == 200:
                st.subheader("📈 Monthly Sentiment Trend")
                trend_img = Image.open(io.BytesIO(trend_response.content))
                st.image(trend_img, use_container_width=True)
            else:
                st.error(f"Trend graph generation failed: {trend_response.text}")
        else:
            st.error("Trend graph generation failed: Missing required columns in data.")
    except Exception as e:
        st.error(f"Trend graph generation failed: {e}")
