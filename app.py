import streamlit as st
import pickle
from preprocess import clean_text

# Page setup
st.set_page_config(page_title="TrustCheck", page_icon="🔍")

# Load the trained AI model and vectorizer
@st.cache_resource
def load_model():
    with open('model/classifier.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

try:
    model, vectorizer = load_model()
except FileNotFoundError:
    st.error("Model not found! Please run 'python train_model.py' in your terminal first.")
    st.stop()

# --- MAIN WEBPAGE ---
st.title("🔍 TrustCheck")
st.markdown("### AI-Based Fake Job & Scholarship Advert Detector")
st.markdown("---")
st.markdown("Paste a job or scholarship advert below to check if it is likely fake or genuine.")

# Text input box
advert_text = st.text_area("Advert Text", height=150, placeholder="Example: URGENT! Apply now for fully funded scholarship. Pay only K500 processing fee. WhatsApp 0977XXXXXX...")

# Button
if st.button("🔍 Check Advert", type="primary"):
    if not advert_text.strip():
        st.warning("⚠️ Please paste some advert text first.")
    else:
        # Clean the text and make a prediction
        cleaned = clean_text(advert_text)
        features = vectorizer.transform([cleaned])
        prediction = model.predict(features)[0]
        confidence = model.predict_proba(features)[0].max() * 100
        
        st.markdown("---")
        st.markdown("### 📊 Result")
        
        # Display result
        if prediction == 'fake':
            st.error(f"🚨 **LIKELY FAKE** (Confidence: {confidence:.1f}%)")
            st.write("**Recommendation:** Do NOT pay any fees or share personal information.")
        else:
            st.success(f"✅ **LIKELY GENUINE** (Confidence: {confidence:.1f}%)")
            st.write("**Recommendation:** Still verify through official channels before sharing info.")