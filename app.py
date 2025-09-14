import streamlit as st
import requests
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import json
from collections import Counter
import re

# Set page configuration
st.set_page_config(
    page_title="NER with Hugging Face",
    page_icon="🤖",
    layout="wide"
)

# Title and description
st.title("🔍 Named Entity Recognition (NER) with Hugging Face")
st.markdown("""
This app performs Named Entity Recognition (NER) using pre-trained models from Hugging Face.
Paste your text below or try the example to extract entities like persons, organizations, and locations.
""")

# Sidebar for configuration
st.sidebar.header("Configuration")
model_option = st.sidebar.selectbox(
    "Choose a model",
    [
        "dslim/bert-base-NER", 
        "dbmdz/bert-large-cased-ner-english", 
        "Jean-Baptiste/roberta-large-ner-english",
        "Davlan/bert-base-multilingual-cased-ner-hrl",
        "Babelscape/wikineural-multilingual-ner"
    ]
)

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
    # Remove excessive whitespace but preserve meaningful punctuation
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Add Hugging Face token input (optional)
hf_token = st.sidebar.text_input("Hugging Face Token (optional)", type="password")
st.sidebar.info("A token is only needed for private/gated models. The selected models are public.")

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
def load_ner_model(model_name, token):
    try:
        if token:
            nlp = pipeline("ner", model=model_name, token=token, aggregation_strategy="simple")
        else:
            nlp = pipeline("ner", model=model_name, aggregation_strategy="simple")
        return nlp
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Filter and improve entity recognition
def filter_entities(entities, confidence_threshold=0.7):
    filtered = []
    for ent in entities:
        # Filter by confidence
        if ent['score'] < confidence_threshold:
            continue
            
        # Clean entity word
        word = ent['word'].strip()
        if not word or len(word) < 2:  # Skip very short entities
            continue
            
        # Fix common entity type issues
        if ent['entity_group'] == 'ONG':
            ent['entity_group'] = 'ORG'
        
        # Split merged entities (like "Austin Texas" should be two entities)
        if ' ' in word and ent['entity_group'] in ['LOC', 'PER', 'ORG']:
            # For certain cases, consider splitting but this is complex
            # For now, we'll keep as is but note that better models handle this
            pass
            
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
    with st.spinner("Loading model and processing text..."):
        # Clean the input text
        cleaned_text = clean_text(input_text)
        
        nlp = load_ner_model(model_option, hf_token)
        
        if nlp:
            try:
                # Perform NER
                results = nlp(cleaned_text)
                
                # Filter entities
                filtered_results = filter_entities(results)
                
                if filtered_results:
                    # Display results in two columns
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📋 Detected Entities")
                        
                        # Create a DataFrame with full forms
                        entities_data = []
                        for ent in filtered_results:
                            entities_data.append({
                                'Entity': ent['word'],
                                'Type': ent['entity_group'],
                                'Full Form': get_entity_full_form(ent['entity_group']),
                                'Confidence': ent['score']
                            })
                        
                        entities_df = pd.DataFrame(entities_data)
                        
                        # Format the confidence scores
                        entities_df['Confidence'] = entities_df['Confidence'].apply(lambda x: f"{x:.2%}")
                        
                        # Display the dataframe with full forms
                        st.dataframe(entities_df[['Entity', 'Type', 'Full Form', 'Confidence']], 
                                   use_container_width=True, height=300)
                    
                    with col2:
                        st.subheader("📊 Entity Distribution")
                        
                        # Count entity types (using full forms for better readability)
                        entity_types = [ent['entity_group'] for ent in filtered_results]
                        type_counts = Counter(entity_types)
                        
                        # Create a bar chart with proper scaling
                        if type_counts:
                            fig, ax = plt.subplots(figsize=(8, 5))
                            
                            # Get full forms for labels
                            labels = [get_entity_full_form(et) for et in type_counts.keys()]
                            values = list(type_counts.values())
                            
                            bars = ax.bar(labels, values, 
                                        color=['#FF6B6B', '#4ECDC4', '#FFE66D', '#C7C7C7'][:len(labels)])
                            ax.set_ylabel('Count')
                            ax.set_xlabel('Entity Type')
                            ax.set_title('Entity Type Distribution')
                            
                            # Set proper y-axis scale
                            max_count = max(values)
                            ax.set_ylim(0, max_count + 0.5)
                            ax.set_yticks(range(0, max_count + 1))
                            
                            # Add value labels on bars
                            for bar in bars:
                                height = bar.get_height()
                                ax.text(bar.get_x() + bar.get_width()/2., height,
                                        f'{int(height)}', ha='center', va='bottom', fontweight='bold')
                            
                            plt.xticks(rotation=45, ha='right')
                            plt.tight_layout()
                            st.pyplot(fig)
                        
                        # Show entity type legend
                        st.markdown("**Entity Type Legend:**")
                        legend_html = """
                        <div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin-top: 10px;">
                        <b>PER</b> - Person (Names of people)<br>
                        <b>ORG</b> - Organization (Companies, institutions)<br>
                        <b>LOC</b> - Location (Places, cities, countries)<br>
                        <b>MISC</b> - Miscellaneous (Other entities)
                        </div>
                        """
                        st.markdown(legend_html, unsafe_allow_html=True)
                    
                    # Display the text with highlighted entities
                    st.subheader("🔦 Text with Highlighted Entities")
                    
                    # Color mapping for entity types
                    color_map = {
                        'PER': '#FF6B6B',  # Red for persons
                        'ORG': '#4ECDC4',  # Teal for organizations
                        'LOC': '#FFE66D',  # Yellow for locations
                        'MISC': '#C7C7C7'  # Gray for miscellaneous
                    }
                    
                    # Create HTML with highlighted entities
                    highlighted_text = cleaned_text
                    # Sort by start index in reverse to avoid offset issues when replacing
                    for ent in sorted(filtered_results, key=lambda x: x['start'], reverse=True):
                        entity_type = ent['entity_group']
                        full_form = get_entity_full_form(entity_type)
                        color = color_map.get(entity_type, '#C7C7C7')  # Default to gray
                        label = f"<mark style='background-color: {color}; padding: 2px 4px; border-radius: 3px; margin: 2px;'>{ent['word']} ({entity_type})</mark>"
                        highlighted_text = highlighted_text[:ent['start']] + label + highlighted_text[ent['end']:]
                    
                    # Display the highlighted text in a nice container
                    st.markdown(
                        f"""<div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #4ECDC4;">
                        {highlighted_text}
                        </div>""", 
                        unsafe_allow_html=True
                    )
                    
                    # Show model information
                    st.info(f"Using model: **{model_option}**")
                    
                    # Show raw JSON output
                    with st.expander("View raw output"):
                        st.json(filtered_results)
                
                else:
                    st.warning("No valid entities detected in the text. Try:")
                    st.markdown("- A different model (try 'Jean-Baptiste/roberta-large-ner-english')")
                    st.markdown("- Longer text with more named entities")
                    st.markdown("- Clearer entity mentions (proper names)")
                    
            except Exception as e:
                st.error(f"Error during NER processing: {str(e)}")
                st.info("Try selecting a different model from the sidebar.")
        else:
            st.error("Failed to load the model. Please check your Hugging Face token if using a private model.")

# Add troubleshooting section
with st.expander("🔧 Troubleshooting Tips"):
    st.markdown("""
    **If entities are missing or incorrect:**
    
    1. **Try different models**: Some models work better for specific types of text
    2. **Check text quality**: Ensure proper capitalization and punctuation
    3. **Longer text**: Models often perform better with more context
    4. **Specific entities**: Some models are better at certain entity types
    
    **Recommended models for better accuracy:**
    - `Jean-Baptiste/roberta-large-ner-english` - Good overall performance
    - `dbmdz/bert-large-cased-ner-english` - Better with capitalized entities
    """)

# Footer
st.markdown("---")
st.markdown(
    "**Day 52 of the NLP & AI Beyond GPT series** | "
    "Models provided by [Hugging Face](https://huggingface.co/)"
)
