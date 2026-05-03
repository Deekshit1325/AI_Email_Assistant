# app.py
import streamlit as st
import joblib
import re

# PAGE CONFIG
st.set_page_config(
    page_title="Smart Email Risk Analyzer",
    page_icon="📧",
    layout="centered")

# LOAD MODEL
@st.cache_resource
def load_files():
    model = joblib.load("spam_model.pkl")
    vectorizer = joblib.load("spam_vectorizer.pkl")
    return model, vectorizer
model, vectorizer = load_files()

# CLEAN TEXT
def clean_text(text):
    text = str(text).lower()
    text = text.replace("escapenumber", " ")
    text = text.replace("escapelong", " ")
    text = text.replace("escapeurl", " ")
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# SMART RULES
def apply_rules(email_text):
    text = email_text.lower()
    trusted_patterns = [
        "we received your application",
        "interview scheduled",
        "recruiting team",
        "candidate self service",
        "otp",
        "verification code",
        "meeting tomorrow",
        "invoice attached"]

    scam_patterns = [
        "win money",
        "claim prize",
        "click now",
        "loan approved",
        "limited offer",
        "earn money fast"]

    # Trusted Legitimate Emails
    for pattern in trusted_patterns:
        if pattern in text:
            return "LEGITIMATE", f"Trusted pattern detected: '{pattern}'"
    # Obvious Spam Emails
    for pattern in scam_patterns:
        if pattern in text:
            return "SPAM", f"Spam pattern detected: '{pattern}'"
    return None, None

# HEADER
st.title("📧 Smart Email Risk Analyzer")
st.write("Machine Learning + Smart Rules")
st.markdown("---")

# SIDEBAR
st.sidebar.header("About")
st.sidebar.info("""
Outputs:

✅ Legitimate  
🚫 Spam

Uses:

• Logistic Regression  
• TF-IDF  
• Explainable AI  
• Smart Rule Overrides
""")

# INPUT
email_text = st.text_area(
    "Paste Email Content:",
    height=280,
    placeholder="""Example:
We received your application for Software Engineering Intern.
OR
Congratulations! You won ₹50,000. Click now!""")

# ANALYZE
if st.button("🔍 Analyze Email"):
    if email_text.strip() == "":
        st.warning("Please enter email content.")
    else:
        st.markdown("---")
        st.subheader("Prediction Result")
        rule_result, reason = apply_rules(email_text)
        if rule_result:
            if rule_result == "LEGITIMATE":
                st.success("✅ LEGITIMATE EMAIL")
            else:
                st.error("🚫 SPAM EMAIL")
            st.write("### Confidence: 95%")
            st.info(reason)
        else:
            # ML MODEL PREDICTION
            cleaned = clean_text(email_text)
            vec = vectorizer.transform([cleaned])
            pred = model.predict(vec)[0]
            probs = model.predict_proba(vec)[0]
            pred_index = list(model.classes_).index(pred)
            confidence = float(probs[pred_index]) * 100
            if str(pred) == "spam" or pred == 1:
                st.error("🚫 SPAM EMAIL")
                st.write(f"### Confidence: {confidence:.2f}%")
                st.info("Predicted by Machine Learning model.")
            else:
                st.success("✅ LEGITIMATE EMAIL")
                st.write(f"### Confidence: {confidence:.2f}%")
                st.info("Predicted by Machine Learning model.")

# EXAMPLES
st.markdown("---")
st.subheader("Try Examples")
with st.expander("📌 Sample Inputs"):
    st.code("We received your application for Software Engineering Intern.")
    st.code("Please attend the project review meeting tomorrow at 3 PM.")
    st.code("Congratulations! You won ₹50,000. Click now!")
    st.code("Limited time loan offer apply immediately.")

# FOOTER
st.markdown("---")
st.caption("Built with Python + Streamlit | Hybrid Email Classifier")