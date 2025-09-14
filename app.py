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
    page_title="NER with RoBERTa",
    page_icon="🤖",
    layout="wide"
)

# Title and description
st.title("🔍 Named Entity Recognition (NER) with RoBERTa")
st.markdown("""
This app performs Named Entity Recognition using the **Jean-Baptiste/roberta-large-ner-english** model from Hugging Face.
This model is specifically trained for high-accuracy entity detection in English text.
""")

# Entity type mapping with full forms
ENTITY_FULL_FORMS = {
    'PER': 'Person',
    'ORG': 'Organization', 
    'LOC': 'Location',
    'MISC': 'Miscellaneous',
    'B-PER': 'Beginning of Person',
    'I-PER': 'Inside of Person', 
    'B-ORG': 'Beginning of Organization',
    'I-ORG': 'Inside of Organization',
    'B-LOC': 'Beginning of Location',
    'I-LOC': 'Inside of Location',
    'B-MISC': 'Beginning of Miscellaneous',
    'I-MISC': 'Inside of Miscellaneous'
}

# Text cleaning function
def clean_text(text):
    """Clean and preprocess input text"""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Add example text
example_text = st.sidebar.selectbox(
    "Try an example",
    [
        "Select an example",
        "Elon Musk is the CEO of Tesla, Inc. which is based in Austin, Texas.",
        "Apple was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976.",
        "The Eiffel Tower is located in Paris, France and was constructed in 1889.",
        "Microsoft Corporation, founded by Bill Gates and Paul Allen, is based in Redmond, Washington."
    ]
)

# Initialize the NER pipeline
@st.cache_resource(show_spinner=False)
def load_ner_model():
    try:
        nlp = pipeline("ner", model="Jean-Baptiste/roberta-large-ner-english", aggregation_strategy="simple")
        return nlp
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Filter entities
def filter_entities(entities, confidence_threshold=0.7):
    filtered = []
    for ent in entities:
        if ent['score'] < confidence_threshold:
            continue
        word = ent['word'].strip()
        if not word or len(word) < 2:
            continue
        filtered.append(ent)
    return filtered

# Get full form of entity type
def get_entity_full_form(entity_type):
    return ENTITY_FULL_FORMS.get(entity_type, entity_type)

# Main content area
input_text = st.text_area(
    "Enter text for NER analysis",
    height=200,
    placeholder="Type or paste your text here...",
    value="Elon Musk is the CEO of Tesla, Inc. which is based in Austin, Texas. The company was founded in 2003 and produces electric vehicles."
)

# Use example if selected
if example_text != "Select an example" and not input_text.strip():
    input_text = example_text

# Process the text when the button is clicked
if st.button("Analyze Text") and input_text.strip():
    with st.spinner("Loading RoBERTa model and processing text..."):
        cleaned_text = clean_text(input_text)
        nlp = load_ner_model()
        
        if nlp:
            try:
                results = nlp(cleaned_text)
                filtered_results = filter_entities(results)
                
                if filtered_results:
                    # Display results in two columns
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📋 Detected Entities")
                        entities_data = []
                        for ent in filtered_results:
                            entities_data.append({
                                'Entity': ent['word'],
                                'Type': ent['entity_group'],
                                'Full Form': get_entity_full_form(ent['entity_group']),
                                'Confidence': f"{ent['score']:.2%}"
                            })
                        
                        entities_df = pd.DataFrame(entities_data)
                        st.dataframe(entities_df, use_container_width=True, height=300)
                    
                    with col2:
                        st.subheader("📊 Entity Distribution")
                        entity_types = [ent['entity_group'] for ent in filtered_results]
                        type_counts = Counter(entity_types)
                        
                        if type_counts:
                            fig, ax = plt.subplots(figsize=(8, 5))
                            labels = [get_entity_full_form(et) for et in type_counts.keys()]
                            values = list(type_counts.values())
                            
                            colors = ['#FF6B6B', '#4ECDC4', '#FFE66D', '#C7C7C7']
                            bars = ax.bar(labels, values, color=colors[:len(labels)])
                            
                            ax.set_ylabel('Count')
                            ax.set_xlabel('Entity Type')
                            ax.set_title('Entity Type Distribution')
                            
                            max_count = max(values) if values else 1
                            ax.set_ylim(0, max_count + 0.5)
                            ax.set_yticks(range(0, max_count + 1))
                            
                            for bar in bars:
                                height = bar.get_height()
                                ax.text(bar.get_x() + bar.get_width()/2., height,
                                        f'{int(height)}', ha='center', va='bottom', fontweight='bold')
                            
                            plt.xticks(rotation=45, ha='right')
                            plt.tight_layout()
                            st.pyplot(fig)
                    
                    # Display highlighted text
                    st.subheader("🔦 Text with Highlighted Entities")
                    color_map = {
                        'PER': '#FF6B6B',
                        'ORG': '#4ECDC4', 
                        'LOC': '#FFE66D',
                        'MISC': '#C7C7C7'
                    }
                    
                    highlighted_text = cleaned_text
                    for ent in sorted(filtered_results, key=lambda x: x['start'], reverse=True):
                        entity_type = ent['entity_group']
                        color = color_map.get(entity_type, '#C7C7C7')
                        label = f"<mark style='background-color: {color}; padding: 2px 4px; border-radius: 3px; margin: 2px;'>{ent['word']} ({entity_type})</mark>"
                        highlighted_text = highlighted_text[:ent['start']] + label + highlighted_text[ent['end']:]
                    
                    st.markdown(
                        f"""<div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #4ECDC4;">
                        {highlighted_text}
                        </div>""", 
                        unsafe_allow_html=True
                    )
                    
                    st.success("✅ Analysis complete using Jean-Baptiste/roberta-large-ner-english model")
                    
                else:
                    st.warning("No entities detected. Try text with more named entities like people, organizations, or locations.")
                    
            except Exception as e:
                st.error(f"Error during processing: {str(e)}")
        else:
            st.error("Failed to load the model. Please check your internet connection.")

# Model information
with st.expander("ℹ️ About the RoBERTa NER Model"):
    st.markdown("""
    **Model:** Jean-Baptiste/roberta-large-ner-english
    
    **Capabilities:**
    - High accuracy for English text
    - Detects Persons, Organizations, Locations, and Miscellaneous entities
    - Handles complex entity boundaries well
    - Good with proper nouns and capitalization
    
    **Expected Output for Example Text:**
    - "Elon Musk" → PER (Person)
    - "Tesla, Inc." → ORG (Organization) 
    - "Austin" → LOC (Location)
    - "Texas" → LOC (Location)
    """)

# Footer
st.markdown("---")
st.markdown(
    "**Day 52: Named Entity Recognition** | "
    "Model: [Jean-Baptiste/roberta-large-ner-english](https://huggingface.co/Jean-Baptiste/roberta-large-ner-english)"
)
