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

# Optimized CSS - Mobile-friendly with reduced animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Remove default margins */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Main container - Solid background for better mobile performance */
    .main {
        background: #1a1a1a;
        padding: 1rem;
    }
    
    .stApp {
        background: #1a1a1a;
    }
    
    /* Fixed header title - Always visible */
    .fixed-header {
        position: sticky;
        top: 0;
        z-index: 999;
        background: linear-gradient(135deg, #1a1a1a 0%, #2d0a0a 100%);
        padding: 1rem;
        margin: -1rem -1rem 1rem -1rem;
        border-bottom: 2px solid #dc2626;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    /* Main title - Large and visible */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin: 0;
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Mobile optimization */
    @media (max-width: 768px) {
        .main-title {
            font-size: 1.8rem;
        }
        
        .fixed-header {
            padding: 0.75rem;
        }
        
        .stat-card {
            margin: 0.5rem 0;
            padding: 1rem;
        }
    }
    
    /* Subtitle */
    .subtitle-text {
        text-align: center;
        font-size: 0.95rem;
        color: #9ca3af;
        margin: 0.5rem 0 0 0;
    }
    
    /* Stat cards - Simplified for mobile performance */
    .stat-card {
        background: rgba(30, 30, 30, 0.95);
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.2);
        border: 1px solid rgba(220, 38, 38, 0.3);
        transition: transform 0.2s ease;
        margin: 1rem 0;
        will-change: transform;
    }
    
    /* Simplified hover - better for mobile */
    .stat-card:hover {
        transform: translateY(-3px);
        border-color: rgba(220, 38, 38, 0.5);
    }
    
    /* Remove heavy animations on mobile */
    @media (max-width: 768px) {
        .stat-card:hover {
            transform: none;
        }
        
        .pulse-animation {
            animation: none !important;
        }
    }
    
    /* Simplified pulse - less intensive */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.9; }
    }
    
    .pulse-animation {
        animation: pulse 3s infinite;
    }
    
    /* Success message */
    .success-message {
        background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 600;
        margin: 1rem 0;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
    }
    
    /* Button styling - Optimized */
    .stButton > button {
        background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.2s ease;
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.4);
        will-change: transform;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(220, 38, 38, 0.6);
    }
    
    /* Mobile button optimization */
    @media (max-width: 768px) {
        .stButton > button {
            padding: 0.6rem 1.5rem;
            font-size: 1rem;
        }
        
        .stButton > button:hover {
            transform: none;
        }
    }
    
    /* Text area - Dark with red border */
    .stTextArea > div > div > textarea {
        border-radius: 12px;
        border: 2px solid #dc2626;
        background-color: #1f1f1f;
        color: #e5e7eb;
        font-family: 'Inter', sans-serif;
        transition: border-color 0.2s ease;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #ef4444;
        box-shadow: 0 0 0 3px rgba(220, 38, 38, 0.2);
    }
    
    /* Smooth scrolling */
    html {
        scroll-behavior: smooth;
        -webkit-overflow-scrolling: touch;
    }
    
    /* Optimize scrolling on mobile */
    @media (max-width: 768px) {
        * {
            -webkit-tap-highlight-color: transparent;
        }
        
        .main {
            padding: 0.5rem;
        }
    }
    
    /* Slider */
    .stSlider > div > div > div > div {
        background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
    }
    
    /* Tabs - Optimized */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 0.5rem 1rem;
        background: rgba(220, 38, 38, 0.1);
        color: #e5e7eb;
        transition: all 0.2s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
        color: white;
    }
    
    /* Mobile tabs */
    @media (max-width: 768px) {
        .stTabs [data-baseweb="tab"] {
            padding: 0.4rem 0.8rem;
            font-size: 0.9rem;
        }
    }
    
    /* Info box */
    .info-box {
        background: rgba(220, 38, 38, 0.1);
        border-left: 4px solid #dc2626;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #e5e7eb;
    }
    
    /* Comparison card - Simplified */
    .comparison-card {
        background: #1f1f1f;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.2);
        color: #e5e7eb;
        border: 1px solid rgba(220, 38, 38, 0.2);
        transition: transform 0.2s ease;
        will-change: transform;
    }
    
    .comparison-card:hover {
        transform: scale(1.01);
        border-color: rgba(220, 38, 38, 0.4);
    }
    
    @media (max-width: 768px) {
        .comparison-card {
            padding: 0.8rem;
        }
        
        .comparison-card:hover {
            transform: none;
        }
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a1a 0%, #2d0a0a 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: #e5e7eb !important;
    }
    
    /* Select box */
    .stSelectbox > div > div {
        background-color: #1f1f1f;
        color: #e5e7eb;
        border: 2px solid #dc2626;
    }
    
    /* Number input */
    .stNumberInput > div > div > input {
        background-color: #1f1f1f;
        color: #e5e7eb;
        border: 2px solid #dc2626;
    }
    
    /* Code block */
    code {
        background-color: #1f1f1f !important;
        color: #e5e7eb !important;
        border: 1px solid rgba(220, 38, 38, 0.3) !important;
    }
    
    /* Markdown text */
    .stMarkdown {
        color: #e5e7eb;
    }
    
    /* Headers */
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
    
    /* Hardware acceleration for smoother scrolling */
    * {
        -webkit-transform: translateZ(0);
        transform: translateZ(0);
    }
    
    /* Reduce motion for users who prefer it */
    @media (prefers-reduced-motion: reduce) {
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

# Fixed Header with Title - Always visible
st.markdown("""
<div class="fixed-header">
    <h1 class="main-title">✂️ TokenTrim</h1>
    <p class="subtitle-text">AI-Powered Prompt Optimization • Save Costs • Preserve Context</p>
</div>
""", unsafe_allow_html=True)

# Feature cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="stat-card">
        <div style="text-align: center;">
            <div style="font-size: 2.5rem;">🚀</div>
            <h3 style="color: #dc2626; font-size: 1.2rem; margin: 0.5rem 0;">Smart Compression</h3>
            <p style="color: #9ca3af; font-size: 0.9rem; margin: 0;">Reduce tokens by up to 90%</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stat-card">
        <div style="text-align: center;">
            <div style="font-size: 2.5rem;">💰</div>
            <h3 style="color: #dc2626; font-size: 1.2rem; margin: 0.5rem 0;">Cost Savings</h3>
            <p style="color: #9ca3af; font-size: 0.9rem; margin: 0;">Save thousands on API costs</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="stat-card">
        <div style="text-align: center;">
            <div style="font-size: 2.5rem;">🎯</div>
            <h3 style="color: #dc2626; font-size: 1.2rem; margin: 0.5rem 0;">Context Preserved</h3>
            <p style="color: #9ca3af; font-size: 0.9rem; margin: 0;">Semantic meaning intact</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📖 How It Works")
    st.markdown("""
    <div class="info-box">
    <strong>Step 1: spaCy Processing</strong><br>
    Removes common stopwords and punctuation
    </div>
    
    <div class="info-box">
    <strong>Step 2: LLMLingua Compression</strong><br>
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
                    <div class="stat-card">
                        <h4 style="color: #dc2626; margin: 0; font-size: 1rem;">Original Tokens</h4>
                        <h2 style="margin: 0.5rem 0; color: #e5e7eb;">{original_tokens}</h2>
                        <p style="color: #9ca3af; margin: 0; font-size: 0.85rem;">Input length</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with metric_col2:
                    st.markdown(f"""
                    <div class="stat-card">
                        <h4 style="color: #dc2626; margin: 0; font-size: 1rem;">Optimized Tokens</h4>
                        <h2 style="margin: 0.5rem 0; color: #e5e7eb;">{compressed_tokens}</h2>
                        <p style="color: #9ca3af; margin: 0; font-size: 0.85rem;">Output length</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with metric_col3:
                    st.markdown(f"""
                    <div class="stat-card">
                        <h4 style="color: #dc2626; margin: 0; font-size: 1rem;">Tokens Saved</h4>
                        <h2 style="margin: 0.5rem 0; color: #10b981;">{token_reduction}</h2>
                        <p style="color: #9ca3af; margin: 0; font-size: 0.85rem;">Reduction</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with metric_col4:
                    st.markdown(f"""
                    <div class="stat-card">
                        <h4 style="color: #dc2626; margin: 0; font-size: 1rem;">Savings</h4>
                        <h2 style="margin: 0.5rem 0; color: #10b981;">{token_savings_percent:.1f}%</h2>
                        <p style="color: #9ca3af; margin: 0; font-size: 0.85rem;">Efficiency gain</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Tabs
                tab1, tab2, tab3, tab4 = st.tabs(["🎯 Optimized", "📄 Original", "🔧 Stopwords", "⚖️ Compare"])
                
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

                # Cost calculator
                st.markdown("---")
                st.markdown("## 💰 Cost Savings Calculator")
                
                cost_col1, cost_col2, cost_col3 = st.columns(3)
                
                with cost_col1:
                    model_prices = {
                        "GPT-4 Turbo": 0.01,
                        "GPT-4": 0.03,
                        "GPT-3.5 Turbo": 0.0015,
                        "Claude Opus": 0.015,
                        "Claude Sonnet": 0.003
                    }
                    selected_model = st.selectbox("Select your LLM:", list(model_prices.keys()))
                    cost_per_1k = model_prices[selected_model]
                
                with cost_col2:
                    daily_prompts = st.number_input("Daily prompts:", min_value=1, value=100, step=10)
                
                with cost_col3:
                    months = st.slider("Calculate for months:", 1, 12, 1)

                original_cost_per_prompt = (original_tokens / 1000) * cost_per_1k
                compressed_cost_per_prompt = (compressed_tokens / 1000) * cost_per_1k
                cost_saved_per_prompt = original_cost_per_prompt - compressed_cost_per_prompt
                
                daily_savings = cost_saved_per_prompt * daily_prompts
                monthly_savings = daily_savings * 30
                total_savings = monthly_savings * months

                savings_col1, savings_col2, savings_col3 = st.columns(3)
                
                with savings_col1:
                    st.markdown(f"""
                    <div class="stat-card">
                        <h4 style="color: #dc2626;">Daily Savings</h4>
                        <h2 style="color: #10b981;">${daily_savings:.2f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with savings_col2:
                    st.markdown(f"""
                    <div class="stat-card">
                        <h4 style="color: #dc2626;">Monthly Savings</h4>
                        <h2 style="color: #10b981;">${monthly_savings:.2f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with savings_col3:
                    st.markdown(f"""
                    <div class="stat-card">
                        <h4 style="color: #dc2626;">{months}-Month Savings</h4>
                        <h2 style="color: #10b981;">${total_savings:.2f}</h2>
                    </div>
                    """, unsafe_allow_html=True)

                st.caption(f"💡 Based on {selected_model} pricing (${cost_per_1k}/1K tokens) with {daily_prompts} prompts/day")

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
<div style="text-align: center; padding: 1.5rem; color: #9ca3af;">
    <p style="font-size: 1rem; margin-bottom: 0.5rem;">Made with ❤️ using <strong style="color: #dc2626;">TokenTrim</strong></p>
    <p style="font-size: 0.85rem;">Powered by LLMLingua & spaCy | Running on CPU Mode</p>
    <p style="font-size: 0.75rem; margin-top: 1rem;">🔒 Your data is processed locally and never stored</p>
</div>
""", unsafe_allow_html=True)