import streamlit as st
from google import genai
from datasets import load_dataset

st.set_page_config(page_title="ToS Auditor", layout="wide")

# Securely retrieve the API key from secrets
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    # Initialize the modern Gemini Client
    client = genai.Client(api_key=api_key)
except Exception as e:
    st.error("API Key not found. Please set GEMINI_API_KEY in your Streamlit secrets.")
    st.stop() # Stops execution if there is no key

LABEL_MAP = { 
    0: "Limitation of liability",
    1: "Unilateral termination",
    2: "Unilateral change",
    3: "Content removal",
    4: "Contract by-using",
    5: "Choice of law",
    6: "Jurisdiction",
    7: "Arbitration"
}

@st.cache_data
def load_tos_dataset():
    # Load the 'test' split of the UNFAIR-ToS dataset from LexGLUE
    dataset = load_dataset("coastalcph/lex_glue", "unfair_tos", split="test")
    return dataset

with st.spinner("Loading legal dataset from Hugging Face..."):
    dataset = load_tos_dataset()

# Helper to format the expert labels
def get_expert_label_text(labels):
    if not labels:
        return "✅ Fair Clause"
    friendly_labels = [LABEL_MAP[l] for l in labels]
    return f"❌ Unfair Clause ({', '.join(friendly_labels)})"

# --- STREAMLIT UI ---
st.title("ToS Fee & Fairness Auditor")
st.subheader("An Interactive AI PM Evaluation Benchmark")

# Setup session state to hold the selected text from the sidebar
if "selected_text" not in st.session_state:
    st.session_state.selected_text = ""

col1, col2 = st.columns(2)

with col1:
    st.header("The Auditor (AI Prediction)")
    
    # Text area populated by session state
    user_input = st.text_area(
        "Paste a Terms of Service clause here:", 
        value=st.session_state.selected_text,
        height=150
    )
    
    analyze_button = st.button("Analyze Clause")

    if analyze_button and user_input:
        with st.spinner("Analyzing with Gemini..."):
            try:
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=(
                        "You are an expert European consumer protection lawyer. "
                        "Analyze this terms of service sentence and tell me if it is fair or unfair under European consumer law. "
                        f"Explain why simply: {user_input}"
                    )
                )
                
                st.success("Analysis Complete!")
                st.write(response.text)
                
            except Exception as e:
                st.error(f"An error occurred: {e}")

with col2:
    st.header("Benchmark Test Cases (LexGLUE)")
    st.write("Click any real-world clause below to load it into the Auditor. Compare Gemini's live prediction against the expert human label.")
    
    # Display the first 5 examples from the dataset as interactive cards
    for idx in range(1,6):
        sample = dataset[idx]
        text_clause = sample["text"]
        labels = sample["labels"]
        expert_assessment = get_expert_label_text(labels)
        
        # UI card container
        with st.container(border=True):
            st.markdown(f"**Sample #{idx + 1}**")
            st.write(f'"{text_clause}"')
            st.markdown(f"**Expert Human Label:** `{expert_assessment}`")
            
            # Button to copy this sample's text to the input area
            if st.button(f"Load Sample #{idx + 1}", key=f"load_{idx}"):
                st.session_state.selected_text = text_clause
                st.rerun()