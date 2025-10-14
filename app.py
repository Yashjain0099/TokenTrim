import os
os.environ["HF_HOME"] = "/tmp/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface"
os.environ["XDG_CACHE_HOME"] = "/tmp"
os.environ["SPACY_CACHE_DIR"] = "/tmp"

import streamlit as st

# MUST be first Streamlit command
st.set_page_config(
    page_title="TokenTrim - AI Prompt Optimizer",
    layout="wide",
    page_icon="✂️",
    initial_sidebar_state="expanded"
)

import spacy
from llmlingua import PromptCompressor
from typing import Optional, Dict, Any
import torch
import gc
import time

# Clear GPU memory at startup
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()

# Custom CSS - Dark Red & Black Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container - Dark theme */
    .main {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d0a0a 100%);
        padding: 2rem;
    }
    
    /* Stacks background */
    .stApp {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d0a0a 100%);
    }
    
    /* Gradient text - Dark Red */
    .gradient-text {
        background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1rem;
        animation: fadeInDown 1s ease-in-out;
    }
    
    .title-box {
        /* background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%); */
        padding: 2rem 1rem;
        margin: -1rem -1rem 2rem -1rem;
        text-align: center;
        border-radius: 20px 20px 20px 20px;
        border: 4px solid #dc2626; /* Red border */
        box-shadow: 0 8px 24px rgba(220, 38, 38, 0.6);
        position: sticky;
        top: 0;
        z-index: 9999;
    }
    
    .title-text {
        font-size: 3rem;
        font-weight: 700;
        color: white;
        margin: 0;
        text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.7);
        letter-spacing: 2px;
    }
    
    .subtitle-text {
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.95);
        margin: 0.8rem 0 0 0;
        font-weight: 500;
    }
    
     @media (max-width: 768px) {
        .title-box {
            padding: 1.5rem 0.5rem;
            margin: 0 0 1.5rem 0;
        }
        
        .title-text {
            font-size: 2rem;
            letter-spacing: 1px;
        }
        
        .subtitle-text {
            font-size: 0.85rem;
        }
        
        * {
            animation: none !important;
            transition: none !important;
        }
    }
    .stat-card {
        background: rgba(30, 30, 30, 0.95);
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.3);
        border: 1px solid rgba(220, 38, 38, 0.3);
        margin: 1rem 0;
        will-change: transform, box-shadow;
    }
    
    .stat-card-glow {
        background: linear-gradient(135deg, rgba(220, 38, 38, 0.25) 0%, rgba(153, 27, 27, 0.25) 100%);
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 6px 20px rgba(220, 38, 38, 0.7);
        border: 3px solid #dc2626;
        margin: 1rem 0;
    }
    
    @media (hover: hover) {
        .stat-card:hover {
            transform: translateY(-3px);
        }
        
        .stat-card-glow {
            animation: glow-pulse 2s ease-in-out infinite;
        }
    }
    
    @keyframes glow-pulse {
        0%, 100% { box-shadow: 0 6px 20px rgba(220, 38, 38, 0.7); }
        50% { box-shadow: 0 8px 28px rgba(220, 38, 38, 1); }
    }
    
    /* Stat cards - Dark with red accents */
    .stat-card {
        background: rgba(30, 30, 30, 0.95);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(220, 38, 38, 0.2);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(220, 38, 38, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin: 1rem 0;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 48px rgba(220, 38, 38, 0.4);
        border-color: rgba(220, 38, 38, 0.6);
    }
    
    /* Pulse animation */
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    .pulse-animation {
        animation: pulse 2s infinite;
    }
    
    /* Fade animations */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Success message - Dark red */
    .success-message {
        background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
        animation: fadeInUp 0.5s ease-in-out;
        margin: 1rem 0;
    }
    
    /* Token badge */
    .token-badge {
        display: inline-block;
        background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        margin: 0.5rem;
        animation: fadeInUp 0.5s ease-in-out;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
    }
    
    /* Button styling - Dark red */
    .stButton > button {
        background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(220, 38, 38, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(220, 38, 38, 0.6);
        background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%);
    }
    @media (max-width: 768px) {
        .stButton > button {
            padding: 0.65rem 1.5rem;
            font-size: 0.95rem;
        }
    }
    
    /* Text area - Dark with red border */
    .stTextArea > div > div > textarea {
        border-radius: 12px;
        border: 2px solid #dc2626;
        background-color: #1f1f1f;
        color: #e5e7eb;
        font-family: 'Inter', sans-serif;
        transition: border-color 0.3s ease;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #ef4444;
        box-shadow: 0 0 0 3px rgba(220, 38, 38, 0.2);
    }
    
    /* Slider */
    .stSlider > div > div > div > div {
        background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
    }
    
    /* Tabs - Dark theme */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        padding: 0.5rem 1.5rem;
        background: rgba(220, 38, 38, 0.1);
        color: #e5e7eb;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
        color: white;
    }
    
    /* Info box - Dark */
    .info-box {
        background: rgba(220, 38, 38, 0.1);
        border-left: 4px solid #dc2626;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        animation: fadeInUp 0.5s ease-in-out;
        color: #e5e7eb;
    }
    
    /* Comparison card - Dark */
    .comparison-card {
        background: #1f1f1f;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(220, 38, 38, 0.2);
        transition: transform 0.3s ease;
        color: #e5e7eb;
        border: 1px solid rgba(220, 38, 38, 0.2);
    }
    
    .comparison-card:hover {
        transform: scale(1.02);
        border-color: rgba(220, 38, 38, 0.4);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a1a 0%, #2d0a0a 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: #e5e7eb !important;
    }
    
    /* Select box - Dark */
    .stSelectbox > div > div {
        background-color: #1f1f1f;
        color: #e5e7eb;
        border: 2px solid #dc2626;
    }
    
    /* Number input - Dark */
    .stNumberInput > div > div > input {
        background-color: #1f1f1f;
        color: #e5e7eb;
        border: 2px solid #dc2626;
    }
    
    /* Code block - Dark */
    code {
        background-color: #1f1f1f !important;
        color: #e5e7eb !important;
        border: 1px solid rgba(220, 38, 38, 0.3) !important;
    }
    
    /* Markdown text color */
    .stMarkdown {
        color: #e5e7eb;
    }
    
    /* Headers color */
    h1, h2, h3, h4, h5, h6 {
        color: #e5e7eb !important;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        color: #e5e7eb;
    }
    
    /* Alert boxes */
    .stAlert {
        background-color: #1f1f1f;
        color: #e5e7eb;
        border-left: 4px solid #dc2626;
    }
    
    @media (max-width: 768px) {
    * {
        animation: none !important;
        transition: none !important;
    }
}

</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_optimizer():
    return PromptOptimizer()

class PromptOptimizer:
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        print("Loading models...")
        
        # Download spaCy model if not available
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"Downloading {spacy_model}...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", spacy_model])
            self.nlp = spacy.load(spacy_model)
        
        self.llm_lingua = PromptCompressor(
            model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
            device_map="cpu"
        )
        print("Models loaded successfully (running on CPU)!")

    def _remove_stopwords(self, text: str) -> str:
        doc = self.nlp(text)
        filtered_tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]
        return " ".join(filtered_tokens)

    def optimize_prompt(self, prompt_text: str, compression_rate: float = 0.5) -> Dict[str, Any]:
        spacy_cleaned_prompt = self._remove_stopwords(prompt_text)
        spacy_tokens = list(self.nlp.tokenizer(spacy_cleaned_prompt))
        target_token_count = max(10, int(len(spacy_tokens) * compression_rate))

        try:
            compressed_output = self.llm_lingua.compress_prompt(
                [spacy_cleaned_prompt],
                instruction="",
                target_token=target_token_count
            )
            final_compressed_prompt = compressed_output['compressed_prompt']
        except Exception as e:
            final_compressed_prompt = spacy_cleaned_prompt

        original_tokens = len(list(self.nlp.tokenizer(prompt_text)))
        compressed_tokens = len(list(self.nlp.tokenizer(final_compressed_prompt)))

        return {
            "original_prompt": prompt_text,
            "spacy_cleaned_prompt": spacy_cleaned_prompt,
            "final_compressed_prompt": final_compressed_prompt,
            "original_token_count": original_tokens,
            "compressed_token_count": compressed_tokens
        }

# Header with custom HTML (shows correctly after set_page_config)
# TITLE - Big and visible
st.markdown("""
<div class="title-box">
    <h1 class="title-text">✂️ TokenTrim</h1>
    <p class="subtitle-text">AI-Powered Prompt Optimization • Save Costs • Preserve Context</p>
</div>
""", unsafe_allow_html=True)

# Feature cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="stat-card">
        <div style="text-align: center;">
            <div style="font-size: 3rem;">🚀</div>
            <h3 style="color: #dc2626;">Smart Compression</h3>
            <p style="color: #9ca3af;">Reduce tokens by up to 90%</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stat-card">
        <div style="text-align: center;">
            <div style="font-size: 3rem;">💰</div>
            <h3 style="color: #dc2626;">Cost Savings</h3>
            <p style="color: #9ca3af;">Save thousands on API costs</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="stat-card">
        <div style="text-align: center;">
            <div style="font-size: 3rem;">🎯</div>
            <h3 style="color: #dc2626;">Context Preserved</h3>
            <p style="color: #9ca3af;">Semantic meaning intact</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)



# Sidebar
with st.sidebar:
    st.markdown("### 📖 How It Works")
    st.markdown("""
    <div class="info-box">
    <strong>Step 1: Enter a prompt </strong><br>
    Input any text prompt you want to optimize
    </div>
    
    <div class="info-box">
    <strong>Step 2: do optimization setting</strong><br>
    AI-powered semantic compression
    </div>
    
    <div class="info-box">
    <strong>Step 3: Get Results</strong><br>
    Use optimized prompt in your apps
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 💡 Pro Tips")
    st.markdown("""
    - Start with 70% compression for safety
    - Test optimized prompts with your model
    - Compare outputs side-by-side
    - Adjust rate based on results
    """)
    
    st.markdown("---")
    st.markdown("### 📊 Supported Models")
    st.markdown("""
    ✅ GPT-4, GPT-3.5  
    ✅ Claude (all versions)  
    ✅ PaLM, Gemini  
    ✅ LLaMA, Mistral  
    ✅ Any text-based LLM
    """)

# Load optimizer
try:
    with st.spinner("🔄 Loading AI models..."):
        optimizer = load_optimizer()
    st.markdown('<div class="success-message">✅ Models loaded successfully! Ready to optimize.</div>', unsafe_allow_html=True)
except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.info("💡 Try: Runtime → Restart runtime, then run all cells again")
    st.stop()

# Main section
st.markdown("---")
st.markdown("## 📝 Enter Your Prompt")

# Example prompts
example_prompts = {
    "Select an example...": "",
    "📊 Data Analysis": "Please analyze the following financial report in detail and provide a comprehensive summary of all the key findings, highlighting any potential risks or opportunities that may be relevant for our investment strategy going forward.",
    "✍️ Content Writing": "I need you to write a detailed blog post about artificial intelligence and machine learning technologies, covering the history, current applications, future potential, ethical considerations, and impact on society.",
    "🔍 Research Query": "Can you help me understand the complex relationship between climate change and ocean acidification, including the chemical processes involved, the impact on marine ecosystems, and potential solutions?",
    "💼 Business Email": "I would like to compose a professional email to our valued clients informing them about the upcoming changes to our service terms and conditions, explaining the reasons for these changes, and reassuring them of our continued commitment to excellent service.",
    "🎓 Educational Content": "Please explain quantum computing in simple terms that a high school student could understand, covering the basic principles, how it differs from classical computing, potential applications, and current limitations in the field."
}

selected_example = st.selectbox("Try an example prompt:", list(example_prompts.keys()))

user_input = st.text_area(
    "Paste or type your prompt here...", 
    value=example_prompts[selected_example],
    height=200, 
    placeholder="Example: Please analyze the following data and provide insights...",
    help="Enter any prompt you want to optimize"
)

# Settings
st.markdown("### ⚙️ Optimization Settings")

col1, col2 = st.columns([3, 1])
with col1:
    compression_rate = st.slider(
        "Target Compression Rate",
        min_value=0.1, max_value=0.9, value=0.7, step=0.05,
        help="Lower = more aggressive compression. Start with 0.7 for best results."
    )

with col2:
    st.markdown("#### 📊 Compression Level")
    if compression_rate >= 0.7:
        st.success("🟢 Conservative")
    elif compression_rate >= 0.5:
        st.warning("🟡 Balanced")
    else:
        st.error("🔴 Aggressive")

# Optimize button
st.markdown("<br>", unsafe_allow_html=True)
if st.button("✨ Optimize My Prompt", use_container_width=True, type="primary"):
    if user_input and user_input.strip():
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("⏳ Step 1/3: Preprocessing text...")
        progress_bar.progress(33)
        time.sleep(0.3)
        
        status_text.text("🔄 Step 2/3: Removing stopwords...")
        progress_bar.progress(66)
        time.sleep(0.3)
        
        with st.spinner("🤖 Step 3/3: AI compression in progress..."):
            try:
                results = optimizer.optimize_prompt(user_input, compression_rate=compression_rate)
                progress_bar.progress(100)
                status_text.text("✅ Optimization complete!")
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()

                st.balloons()
                
                st.markdown("---")
                st.markdown("## 🎉 Optimization Results")

                original_tokens = results["original_token_count"]
                compressed_tokens = results["compressed_token_count"]
                token_reduction = original_tokens - compressed_tokens
                token_savings_percent = (token_reduction / original_tokens) * 100 if original_tokens > 0 else 0

                # Metrics
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    st.markdown(f"""
                    <div class="stat-card pulse-animation">
                        <h4 style="color: #dc2626; margin: 0;">Original Tokens</h4>
                        <h2 style="margin: 0.5rem 0; color: #e5e7eb;">{original_tokens}</h2>
                        <p style="color: #9ca3af; margin: 0; font-size: 0.9rem;">Input length</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with metric_col2:
                    st.markdown(f"""
                    <div class="stat-card pulse-animation">
                        <h4 style="color: #dc2626; margin: 0;">Optimized Tokens</h4>
                        <h2 style="margin: 0.5rem 0; color: #e5e7eb;">{compressed_tokens}</h2>
                        <p style="color: #9ca3af; margin: 0; font-size: 0.9rem;">Output length</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with metric_col3:
                    st.markdown(f"""
                    <div class="stat-card pulse-animation">
                        <h4 style="color: #dc2626; margin: 0;">Tokens Saved</h4>
                        <h2 style="margin: 0.5rem 0; color: #10b981;">{token_reduction}</h2>
                        <p style="color: #9ca3af; margin: 0; font-size: 0.9rem;">Reduction</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with metric_col4:
                    st.markdown(f"""
                    <div class="stat-card pulse-animation">
                        <h4 style="color: #dc2626; margin: 0;">Savings</h4>
                        <h2 style="margin: 0.5rem 0; color: #10b981;">{token_savings_percent:.1f}%</h2>
                        <p style="color: #9ca3af; margin: 0; font-size: 0.9rem;">Efficiency gain</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Tabs
                tab1, tab2, tab3, tab4 = st.tabs(["🎯 Optimized Prompt", "📄 Original Prompt", "🔧 Stopword Removal", "⚖️ Side-by-Side"])
                
                with tab1:
                    st.markdown("### ✨ Your Optimized Prompt")
                    st.success("This is your cost-efficient prompt ready to use!")
                    st.code(results['final_compressed_prompt'], language=None)
                    st.caption("👆 Copy this optimized prompt to use in ChatGPT, Claude, or any LLM")
                
                with tab2:
                    st.markdown("### 📄 Original Prompt")
                    st.info("Your original input for comparison")
                    st.text_area("", value=results['original_prompt'], height=200, disabled=True, key="orig")
                
                with tab3:
                    st.markdown("### 🔧 After Stopword Removal")
                    st.warning("Intermediate step - stopwords removed")
                    st.text_area("", value=results['spacy_cleaned_prompt'], height=200, disabled=True, key="spacy")
                
                with tab4:
                    st.markdown("### ⚖️ Side-by-Side Comparison")
                    compare_col1, compare_col2 = st.columns(2)
                    
                    with compare_col1:
                        st.markdown("#### 📄 Original")
                        st.markdown(f'<div class="comparison-card">{results["original_prompt"]}</div>', unsafe_allow_html=True)
                    
                    with compare_col2:
                        st.markdown("#### ✨ Optimized")
                        st.markdown(f'<div class="comparison-card">{results["final_compressed_prompt"]}</div>', unsafe_allow_html=True)

                # Simple cost calculator - no inputs, just shows savings
                st.markdown("---")
                st.markdown("## 💰 Estimated Cost Savings")
                
                st.info("📊 Based on GPT-4 pricing ($0.03 per 1K tokens) with 100 prompts per day")

                # Calculate for different models
                models = {
                    "GPT-4": 0.03,
                    "GPT-4 Turbo": 0.01,
                    "Claude Opus": 0.015,
                    "Claude Sonnet": 0.003,
                    "GPT-3.5": 0.0015
                }

                daily_prompts = 100

                for model_name, cost_per_1k in models.items():
                    original_cost = (original_tokens / 1000) * cost_per_1k * daily_prompts
                    compressed_cost = (compressed_tokens / 1000) * cost_per_1k * daily_prompts
                    daily_saved = original_cost - compressed_cost
                    monthly_saved = daily_saved * 30
                    yearly_saved = daily_saved * 365

                    col_a, col_b, col_c, col_d = st.columns(4)
                    
                    with col_a:
                        st.markdown(f"**{model_name}**")
                    
                    with col_b:
                        st.markdown(f"""
                        <div class="stat-card">
                            <p style="color: #9ca3af; margin: 0; font-size: 0.8rem;">Daily</p>
                            <h4 style="color: #10b981; margin: 0.3rem 0;">${daily_saved:.2f}</h4>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_c:
                        st.markdown(f"""
                        <div class="stat-card">
                            <p style="color: #9ca3af; margin: 0; font-size: 0.8rem;">Monthly</p>
                            <h4 style="color: #10b981; margin: 0.3rem 0;">${monthly_saved:.2f}</h4>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_d:
                        st.markdown(f"""
                        <div class="stat-card-glow">
                            <p style="color: #d1d5db; margin: 0; font-size: 0.8rem;">Yearly</p>
                            <h4 style="color: #10b981; margin: 0.3rem 0; font-weight: 700;">${yearly_saved:.2f}</h4>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                # Testing section
                st.markdown("---")
                st.markdown("## 🧪 Test Both Prompts")
                st.info("💡 Copy both prompts and test them in your LLM to compare outputs!")
                
                test_col1, test_col2 = st.columns(2)
                
                with test_col1:
                    st.markdown("### 📄 Test Original")
                    if st.button("📋 Copy Original", use_container_width=True):
                        st.code(results['original_prompt'], language=None)
                
                with test_col2:
                    st.markdown("### ✨ Test Optimized")
                    if st.button("📋 Copy Optimized", use_container_width=True):
                        st.code(results['final_compressed_prompt'], language=None)

            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"An error occurred: {e}")
                st.info("💡 Try adjusting the compression rate or check your input text.")
    else:
        st.warning("⚠️ Please enter some text to optimize.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #9ca3af;">
    <p style="font-size: 1.1rem; margin-bottom: 0.5rem;"><strong style="color: #dc2626;">TokenTrim</strong> created with ❤️ by<strong> Yash</strong> </p>
    <p style="font-size: 0.9rem;"> | Running on CPU Mode | </p>
    <p style="font-size: 0.8rem; margin-top: 1rem;">🔒 Your data is processed locally and never stored</p>
</div>
""", unsafe_allow_html=True)