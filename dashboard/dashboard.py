"""
dashboard/dashboard.py
-----------------------
Streamlit dashboard for the Defect Classifier API.

Run locally:
    pip install -r dashboard/requirements.txt
    streamlit run dashboard/dashboard.py
"""

import io

import requests
import streamlit as st
from PIL import Image

API_URL = "https://defect-classifier-api-874629550296.europe-west1.run.app"

st.set_page_config(page_title="Defect Classifier Dashboard", page_icon="🔍", layout="wide")

st.title("Defect Classifier")
st.caption(f"ResNet18 + Grad-CAM · MVTec AD · [API docs]({API_URL}/docs)")


@st.cache_data(ttl=300)
def get_categories() -> list[str]:
    try:
        r = requests.get(f"{API_URL}/health", timeout=15)
        r.raise_for_status()
        return r.json()["categories"]
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach API: {e}")
        return []


categories = get_categories()
if not categories:
    st.stop()

category = st.selectbox("Category", categories)
uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp"])

if uploaded:
    col1, col2 = st.columns(2)
    col1.image(uploaded, caption="Input", use_container_width=True)

    if st.button("Analyze", type="primary"):
        file_bytes = uploaded.getvalue()

        with st.spinner("Calling API..."):
            try:
                pred = requests.post(
                    f"{API_URL}/predict",
                    params={"category": category},
                    files={"file": (uploaded.name, file_bytes, uploaded.type)},
                    timeout=30,
                ).json()
                heatmap_resp = requests.post(
                    f"{API_URL}/predict/heatmap",
                    params={"category": category},
                    files={"file": (uploaded.name, file_bytes, uploaded.type)},
                    timeout=30,
                )
            except requests.exceptions.Timeout:
                st.error("Request timed out. The API may be starting up (cold start). Try again.")
                st.stop()
            except requests.exceptions.RequestException as e:
                st.error(f"API error: {e}")
                st.stop()

        if pred.get("defective"):
            col1.error(f"DEFECTIVE (confidence {pred['confidence']:.2%})", icon="🚨")
        else:
            col1.success(f"OK (confidence {pred['confidence']:.2%})", icon="✅")

        heatmap_resp.raise_for_status()
        heatmap_img = Image.open(io.BytesIO(heatmap_resp.content))
        col2.image(heatmap_img, caption="Grad-CAM heatmap", use_container_width=True)
