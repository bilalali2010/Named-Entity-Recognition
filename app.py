import streamlit as st
import requests
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
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
    ["dslim/bert-base-NER", "dbmdz/bert-large-cased-ner-english", "Jean-Baptiste/roberta-large-ner-english"]
)

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
        "The Eiffel Tower is located in Paris, France and was constructed in 1889."
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
        nlp = load_ner_model(model_option, hf_token)
        
        if nlp:
            try:
                # Perform NER
                results = nlp(input_text)
                
                if results:
                    # Display results in two columns
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📋 Detected Entities")
                        
                        # Create a DataFrame for better display
                        entities_df = pd.DataFrame({
                            'Entity': [ent['word'] for ent in results],
                            'Type': [ent['entity_group'] for ent in results],
                            'Confidence': [ent['score'] for ent in results]
                        })
                        
                        # Format the confidence scores
                        entities_df['Confidence'] = entities_df['Confidence'].apply(lambda x: f"{x:.2%}")
                        
                        st.dataframe(entities_df, use_container_width=True)
                    
                    with col2:
                        st.subheader("📊 Entity Distribution")
                        
                        # Count entity types
                        entity_types = [ent['entity_group'] for ent in results]
                        type_counts = Counter(entity_types)
                        
                        # Create a bar chart
                        if type_counts:
                            fig, ax = plt.subplots()
                            ax.bar(type_counts.keys(), type_counts.values())
                            ax.set_ylabel('Count')
                            ax.set_xlabel('Entity Type')
                            plt.xticks(rotation=45)
                            st.pyplot(fig)
                    
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
                    highlighted_text = input_text
                    # Sort by start index in reverse to avoid offset issues when replacing
                    for ent in sorted(results, key=lambda x: x['start'], reverse=True):
                        entity_type = ent['entity_group']
                        color = color_map.get(entity_type, '#C7C7C7')  # Default to gray
                        label = f"<mark style='background-color: {color}; padding: 2px 4px; border-radius: 3px;'>{ent['word']} ({entity_type})</mark>"
                        highlighted_text = highlighted_text[:ent['start']] + label + highlighted_text[ent['end']:]
                    
                    st.markdown(highlighted_text, unsafe_allow_html=True)
                    
                    # Show raw JSON output
                    with st.expander("View raw output"):
                        st.json(results)
                
                else:
                    st.info("No entities detected in the text.")
                    
            except Exception as e:
                st.error(f"Error during NER processing: {str(e)}")
        else:
            st.error("Failed to load the model. Please check your Hugging Face token if using a private model.")

# Add some information about NER
with st.expander("ℹ️ About Named Entity Recognition"):
    st.markdown("""
    **Named Entity Recognition (NER)** is a natural language processing technique that identifies and classifies named entities in text into predefined categories such as:
    
    - **Person (PER)**: Names of people
    - **Organization (ORG)**: Companies, organizations, institutions
    - **Location (LOC)**: Geographical places like cities, countries
    - **Miscellaneous (MISC)**: Other entities that don't fit the above categories
    
    This application uses transformer-based models from Hugging Face to perform NER. The models have been trained on large datasets to recognize entities in text.
    """)

# Footer
st.markdown("---")
st.markdown(
    "**Day 52 of the NLP & AI Beyond GPT series** | "
    "Models provided by [Hugging Face](https://huggingface.co/)"
)
