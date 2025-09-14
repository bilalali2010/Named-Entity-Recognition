import streamlit as st
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from collections import Counter
import re

# Set page configuration
st.set_page_config(
    page_title="NER Analysis Tool | RoBERTa Model",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
        border: 1px solid #e0e0e0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        font-weight: 600;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #135e96;
    }
    .entity-per { background-color: #ff6b6b; color: white; padding: 2px 8px; border-radius: 4px; font-weight: 500; }
    .entity-org { background-color: #4ecdc4; color: white; padding: 2px 8px; border-radius: 4px; font-weight: 500; }
    .entity-loc { background-color: #ffe66d; color: #333; padding: 2px 8px; border-radius: 4px; font-weight: 500; }
    .entity-misc { background-color: #c7c7c7; color: white; padding: 2px 8px; border-radius: 4px; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# Header Section
col1, col2 = st.columns([1, 3])
with col1:
    st.image("https://huggingface.co/front/assets/huggingface_logo-noborder.svg", width=80)
with col2:
    st.markdown('<div class="main-header">Named Entity Recognition Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Powered by RoBERTa Large NER Model | Professional Entity Extraction Tool</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🔧 Configuration")
    st.markdown("---")
    
    st.markdown("#### 📋 Example Texts")
    example_text = st.selectbox(
        "Select pre-defined example",
        [
            "Choose an example",
            "Elon Musk is the CEO of Tesla, Inc. which is based in Austin, Texas.",
            "Apple was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in California.",
            "The United Nations headquarters is located in New York City, USA.",
            "Microsoft Corporation, founded by Bill Gates, is based in Redmond, Washington."
        ]
    )
    
    st.markdown("---")
    st.markdown("#### ℹ️ Model Information")
    st.info("""
    **Model:** Jean-Baptiste/roberta-large-ner-english
    
    **Accuracy:** High precision for English NER
    **Supported Entities:** PER, ORG, LOC, MISC
    **Best for:** Professional text analysis
    """)

# Entity type mapping
ENTITY_FULL_FORMS = {
    'PER': 'Person',
    'ORG': 'Organization', 
    'LOC': 'Location',
    'MISC': 'Miscellaneous'
}

ENTITY_COLORS = {
    'PER': '#ff6b6b',
    'ORG': '#4ecdc4', 
    'LOC': '#ffe66d',
    'MISC': '#c7c7c7'
}

# Initialize the NER pipeline
@st.cache_resource(show_spinner=False)
def load_ner_model():
    try:
        nlp = pipeline("ner", model="Jean-Baptiste/roberta-large-ner-english", aggregation_strategy="simple")
        return nlp
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        return None

# Main content
tab1, tab2, tab3 = st.tabs(["📊 Analysis", "📈 Statistics", "ℹ️ Documentation"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 📝 Input Text")
        input_text = st.text_area(
            "Enter text for entity analysis",
            height=150,
            placeholder="Paste your text here for entity recognition...",
            value="",
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Use example if selected
        if example_text != "Choose an example" and not input_text.strip():
            input_text = example_text

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### ⚡ Quick Actions")
        if st.button("🚀 Analyze Text", use_container_width=True):
            st.session_state.analyze = True
        st.markdown("""
        <div style="margin-top: 1rem;">
        <small>Analyzes text for:</small>
        <ul style="margin: 0; padding-left: 1.2rem;">
            <li><small>Persons (PER)</small></li>
            <li><small>Organizations (ORG)</small></li>
            <li><small>Locations (LOC)</small></li>
            <li><small>Miscellaneous (MISC)</small></li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.get('analyze', False) and input_text.strip():
        with st.spinner("🔍 Processing text with RoBERTa NER model..."):
            nlp = load_ner_model()
            
            if nlp:
                try:
                    results = nlp(input_text)
                    filtered_results = [ent for ent in results if ent['score'] > 0.7 and len(ent['word'].strip()) > 1]
                    
                    if filtered_results:
                        # Metrics row
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric("Total Entities", len(filtered_results))
                            st.markdown('</div>', unsafe_allow_html=True)
                        with col2:
                            per_count = len([ent for ent in filtered_results if ent['entity_group'] == 'PER'])
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric("Persons", per_count)
                            st.markdown('</div>', unsafe_allow_html=True)
                        with col3:
                            org_count = len([ent for ent in filtered_results if ent['entity_group'] == 'ORG'])
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric("Organizations", org_count)
                            st.markdown('</div>', unsafe_allow_html=True)
                        with col4:
                            loc_count = len([ent for ent in filtered_results if ent['entity_group'] == 'LOC'])
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric("Locations", loc_count)
                            st.markdown('</div>', unsafe_allow_html=True)

                        # Results in two columns
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown('<div class="card">', unsafe_allow_html=True)
                            st.markdown("#### 📋 Detected Entities")
                            entities_data = []
                            for ent in filtered_results:
                                entities_data.append({
                                    'Entity': ent['word'],
                                'Type': ent['entity_group'],
                                'Full Form': ENTITY_FULL_FORMS.get(ent['entity_group'], ent['entity_group']),
                                'Confidence': f"{ent['score']:.1%}"
                            })
                            
                            entities_df = pd.DataFrame(entities_data)
                            st.dataframe(entities_df, use_container_width=True, height=300)
                            st.markdown('</div>', unsafe_allow_html=True)

                        with col2:
                            st.markdown('<div class="card">', unsafe_allow_html=True)
                            st.markdown("#### 📈 Entity Distribution")
                            entity_types = [ent['entity_group'] for ent in filtered_results]
                            type_counts = Counter(entity_types)
                            
                            if type_counts:
                                fig, ax = plt.subplots(figsize=(8, 4))
                                labels = [ENTITY_FULL_FORMS.get(et, et) for et in type_counts.keys()]
                                values = list(type_counts.values())
                                colors = [ENTITY_COLORS.get(et, '#C7C7C7') for et in type_counts.keys()]
                                
                                bars = ax.bar(labels, values, color=colors)
                                ax.set_ylabel('Count', fontweight='bold')
                                ax.set_title('Entity Type Distribution', fontweight='bold')
                                
                                for bar in bars:
                                    height = bar.get_height()
                                    ax.text(bar.get_x() + bar.get_width()/2., height,
                                            f'{int(height)}', ha='center', va='bottom', fontweight='bold')
                                
                                plt.xticks(rotation=45, ha='right')
                                plt.tight_layout()
                                st.pyplot(fig)
                            st.markdown('</div>', unsafe_allow_html=True)

                        # Highlighted text
                        st.markdown('<div class="card">', unsafe_allow_html=True)
                        st.markdown("#### 🔍 Text Analysis")
                        highlighted_text = input_text
                        for ent in sorted(filtered_results, key=lambda x: x['start'], reverse=True):
                            entity_type = ent['entity_group']
                            color = ENTITY_COLORS.get(entity_type, '#C7C7C7')
                            label = f"<span class='entity-{entity_type.lower()}'>{ent['word']} ({entity_type})</span>"
                            highlighted_text = highlighted_text[:ent['start']] + label + highlighted_text[ent['end']:]
                        
                        st.markdown(
                            f"""<div style="background-color: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #1f77b4; line-height: 1.6;">
                            {highlighted_text}
                            </div>""", 
                            unsafe_allow_html=True
                        )
                        st.markdown('</div>', unsafe_allow_html=True)

                    else:
                        st.warning("No entities detected. Please try text with more named entities.")

                except Exception as e:
                    st.error(f"Analysis error: {str(e)}")

with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 📊 Performance Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Entity Type Overview")
        st.markdown("""
        | Entity Type | Description | Typical Examples |
        |------------|-------------|------------------|
        | **PER** | Persons | Elon Musk, Marie Curie |
        | **ORG** | Organizations | Tesla Inc., United Nations |
        | **LOC** | Locations | Paris, Pacific Ocean |
        | **MISC** | Miscellaneous | Nobel Prize, COVID-19 |
        """)
    
    with col2:
        st.markdown("##### Model Performance")
        st.markdown("""
        - **Precision**: >90% for well-formed text
        - **Recall**: High for common entities
        - **Language**: English optimized
        - **Best for**: Professional documents, news articles
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 📚 Technical Documentation")
    
    st.markdown("""
    ##### Model Specifications
    - **Base Model**: RoBERTa Large
    - **Training Data**: English Wikipedia, news articles
    - **Entity Types**: 4 primary categories
    - **Input**: Raw text (English)
    - **Output**: JSON with entity positions and confidence scores
    
    ##### Usage Guidelines
    1. Input well-punctuated English text
    2. Ensure proper capitalization for names
    3. Longer texts generally yield better results
    4. Model works best with formal/professional language
    
    ##### Limitations
    - Primarily English language
    - May struggle with very short texts
    - Performance varies with domain-specific terminology
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>Professional NER Analysis Tool | Day 52 Implementation | Powered by Hugging Face 🤗</p>
        <p>Model: <a href="https://huggingface.co/Jean-Baptiste/roberta-large-ner-english" target="_blank">Jean-Baptiste/roberta-large-ner-english</a></p>
    </div>
    """,
    unsafe_allow_html=True
)
