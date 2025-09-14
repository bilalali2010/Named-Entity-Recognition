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
        "Davlan/bert-base-multilingual-cased-ner-hrl"
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
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters that might confuse the model
    text = re.sub(r'[^\w\s@#]', ' ', text)
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

# Filter out low-confidence or invalid entities
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
            
        filtered.append(ent)
    return filtered

# Get full form of entity type
def get_entity_full_form(entity_type):
    return ENTITY_FULL_FORMS.get(entity_type, entity_type)

# Main content area
input_text = st.text_area(
    "Enter text for NER analysis",
    height=200,
    placeholder="Type or paste your text here..."
)

# Use example if selected
if example_text != "Select an example" and not input_text:
    input_text = example_text

# Process the text when the button is clicked
if st.button("Analyze Text") and input_text:
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
                        st.dataframe(entities_df[['Entity', 'Type', 'Full Form', 'Confidence']], use_container_width=True)
                    
                    with col2:
                        st.subheader("📊 Entity Distribution")
                        
                        # Count entity types (using full forms for better readability)
                        entity_full_forms = [get_entity_full_form(ent['entity_group']) for ent in filtered_results]
                        type_counts = Counter(entity_full_forms)
                        
                        # Create a bar chart
                        if type_counts:
                            fig, ax = plt.subplots(figsize=(10, 5))
                            bars = ax.bar(type_counts.keys(), type_counts.values(), 
                                        color=['#FF6B6B', '#4ECDC4', '#FFE66D', '#C7C7C7'])
                            ax.set_ylabel('Count')
                            ax.set_xlabel('Entity Type')
                            ax.set_title('Entity Type Distribution')
                            plt.xticks(rotation=45, ha='right')
                            
                            # Add value labels on bars
                            for bar in bars:
                                height = bar.get_height()
                                ax.text(bar.get_x() + bar.get_width()/2., height,
                                        f'{int(height)}', ha='center', va='bottom')
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                        
                        # Show entity type legend
                        st.markdown("**Entity Type Legend:**")
                        legend_html = """
                        <div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">
                        <b>PER</b> - Person<br>
                        <b>ORG</b> - Organization<br>
                        <b>LOC</b> - Location<br>
                        <b>MISC</b> - Miscellaneous
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
                        label = f"<mark style='background-color: {color}; padding: 2px 4px; border-radius: 3px;'>{ent['word']} ({entity_type}: {full_form})</mark>"
                        highlighted_text = highlighted_text[:ent['start']] + label + highlighted_text[ent['end']:]
                    
                    st.markdown(highlighted_text, unsafe_allow_html=True)
                    
                    # Show raw JSON output
                    with st.expander("View raw output"):
                        st.json(filtered_results)
                
                else:
                    st.info("No valid entities detected in the text. Try a different model or check your input text.")
                    
            except Exception as e:
                st.error(f"Error during NER processing: {str(e)}")
        else:
            st.error("Failed to load the model. Please check your Hugging Face token if using a private model.")

# Add some information about NER with full forms
with st.expander("ℹ️ About Named Entity Recognition"):
    st.markdown("""
    **Named Entity Recognition (NER)** is a natural language processing technique that identifies and classifies named entities in text into predefined categories:
    
    ### Entity Types and Their Full Forms:
    
    - **PER (Person)**: Names of people, characters, or individuals
      - *Examples: Elon Musk, Albert Einstein, Marie Curie*
    
    - **ORG (Organization)**: Companies, organizations, institutions, agencies
      - *Examples: Tesla Inc., United Nations, Harvard University*
    
    - **LOC (Location)**: Geographical places, addresses, landmarks
      - *Examples: Paris, Mount Everest, Pacific Ocean*
    
    - **MISC (Miscellaneous)**: Other entities that don't fit the above categories
      - *Examples: Nobel Prize, Windows OS, COVID-19*
    
    **Technical Note**: Some models use BIO notation:
    - **B-** prefix: Beginning of an entity
    - **I-** prefix: Inside of an entity (continuation)
    - Example: "B-PER" = Beginning of a Person entity
    
    This application uses transformer-based models from Hugging Face to perform NER. The models have been trained on large datasets to recognize entities in text.
    """)

# Footer
st.markdown("---")
st.markdown(
    "**Day 52 of the NLP & AI Beyond GPT series** | "
    "Models provided by [Hugging Face](https://huggingface.co/)"
)
