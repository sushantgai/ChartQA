import streamlit as st
import torch
from PIL import Image
import io
import requests
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import matplotlib.pyplot as plt
import os
import pandas as pd
import re
import base64

# Set page config
st.set_page_config(
    page_title="Chart Q&A ",
    page_icon="📊",
    layout="wide"
)

# Initialize session state variables
if 'paligemma_model' not in st.session_state:
    st.session_state.paligemma_model = None
if 'paligemma_processor' not in st.session_state:
    st.session_state.paligemma_processor = None
if 'device' not in st.session_state:
    st.session_state.device = None
if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'extracted_data' not in st.session_state:
    st.session_state.extracted_data = None

# Initialize PaliGemma Model
@st.cache_resource
def load_paligemma_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        "ahmed-masry/chartgemma", 
        torch_dtype=torch.float16
    )
    processor = AutoProcessor.from_pretrained("ahmed-masry/chartgemma")
    model = model.to(device)
    return model, processor, device

# Function to download sample chart
def download_sample_chart(url, filename):
    try:
        if not os.path.exists(filename):
            response = requests.get(url)
            if response.status_code == 200:
                with open(filename, 'wb') as f:
                    f.write(response.content)
                return True
            else:
                st.error(f"Failed to download sample chart: {response.status_code}")
                return False
        return True
    except Exception as e:
        st.error(f"Error downloading sample chart: {str(e)}")
        return False

# Function to clean model output from print statements and other artifacts
def clean_model_output(text):
    # Check if the entire response is a print statement and extract its content
    print_match = re.search(r'^print\(["\'](.+?)["\']\)$', text.strip())
    if print_match:
        return print_match.group(1)
    
    # Remove all print statements
    text = re.sub(r'print\(.+?\)', '', text, flags=re.DOTALL)
    
    # Remove Python code formatting artifacts
    text = re.sub(r'```python|```', '', text)
    
    return text.strip()

# Function to analyze chart with PaliGemma
def analyze_chart_with_paligemma(model, processor, device, image, query, use_cot=False):
    try:
        # Add program of thought prefix if CoT is enabled
        if use_cot and not query.startswith("program of thought:"):
            modified_query = f"program of thought: {query}"
        else:
            modified_query = query
            
        inputs = processor(text=modified_query, images=image, return_tensors="pt")
        prompt_length = inputs['input_ids'].shape[1]
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate with progress bar
        progress_bar = st.progress(0)
        
        with torch.no_grad():
            generate_ids = model.generate(
                **inputs, 
                num_beams=4, 
                max_new_tokens=512,
                output_scores=True,
                return_dict_in_generate=True
            )
            
            progress_bar.progress(100)
        
        output_text = processor.batch_decode(
            generate_ids.sequences[:, prompt_length:], 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        # Clean output from print statements and other artifacts
        output_text = clean_model_output(output_text)
        
        return output_text
    except Exception as e:
        st.error(f"Error analyzing chart : {str(e)}")
        return f"Error: {str(e)}"

# Function to extract data points from chart
def extract_data_points(model, processor, device, image):
    try:
        # Special query to extract data points
        extraction_query = "program of thought: Extract all data points from this chart. List each category or series and all its corresponding values in a structured format."
        
        with st.spinner("Extracting data points from chart..."):
            result = analyze_chart_with_paligemma(model, processor, device, image, extraction_query)
            
            # Parse the result into a DataFrame
            df = parse_chart_data(result)
            return df
    except Exception as e:
        st.error(f"Error extracting data points: {str(e)}")
        return None

# Function to parse chart data from model response
def parse_chart_data(text):
    try:
        # Clean the text from print statements first
        text = clean_model_output(text)

        data = {}
        lines = text.split('\n')
        current_category = None

        for line in lines:
            if not line.strip():
                continue

            if ':' in line and not re.search(r'\d+\.\d+', line):
                current_category = line.split(':')[0].strip()
                data[current_category] = []
            elif current_category and (re.search(r'\d+', line) or ',' in line):
                value_match = re.findall(r'[-+]?\d*\.\d+|\d+', line)
                if value_match:
                    data[current_category].extend(value_match)

        if not data:
            table_pattern = r'(\w+(?:\s\w+)*)\s*[:|]\s*((?:\d+(?:\.\d+)?(?:\s*,\s*\d+(?:\.\d+)?)*)|(?:\d+(?:\.\d+)?))'
            matches = re.findall(table_pattern, text)
            for category, values in matches:
                category = category.strip()
                if category not in data:
                    data[category] = []
                if ',' in values:
                    values = [v.strip() for v in values.split(',')]
                else:
                    values = [values.strip()]
                data[category].extend(values)

        df = pd.DataFrame(data)

        if df.empty:
            df = pd.DataFrame({'Extracted_Text': [text]})

        return df
    except Exception as e:
        st.error(f"Error parsing chart data: {str(e)}")
        return pd.DataFrame({'Raw_Text': [text]})

# Function to create a download link for dataframe
def get_csv_download_link(df, filename="chart_data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV File</a>'
    return href

# Main UI
st.title("📊 Chart Analysis ")


# Sidebar for model loading and options
with st.sidebar:
    st.header("Model Setup")
    
    if st.button("Load Model"):
        with st.spinner("Loading model... This may take a moment"):
            model, processor, device = load_paligemma_model()
            st.session_state.paligemma_model = model
            st.session_state.paligemma_processor = processor
            st.session_state.device = device
            st.success(f"✅ Model loaded successfully on {device}!")
    
    st.header("Options")
    use_cot = st.checkbox("Enable Chain-of-Thought reasoning", value=True, 
                         help="Adds 'program of thought:' prefix to prompts for better reasoning")
    
    st.header("Sample Charts")
    if st.button("Load Sample Chart"):
        sample_url = "https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/val/png/multi_col_1229.png"
        sample_filename = "chart_example_1.png"
        if download_sample_chart(sample_url, sample_filename):
            st.session_state.current_image = Image.open(sample_filename).convert('RGB')
            st.success("Sample chart loaded!")

# Main content area
col1, col2 = st.columns([3, 2])

with col1:
    st.header("Upload Chart")
    uploaded_file = st.file_uploader("Choose a chart image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert('RGB')
            st.session_state.current_image = image
            # Reset extracted data when new image is uploaded
            st.session_state.extracted_data = None
        except Exception as e:
            st.error(f"Error opening image: {str(e)}")
    
    # Display current image
    if st.session_state.current_image is not None:
        st.image(st.session_state.current_image, caption="Current Chart", use_column_width=True)
        
        # Add extract data points button
        if st.session_state.paligemma_model is not None:
            if st.button("Extract Data Points from Chart"):
                df = extract_data_points(
                    st.session_state.paligemma_model,
                    st.session_state.paligemma_processor,
                    st.session_state.device,
                    st.session_state.current_image
                )
                if df is not None:
                    st.session_state.extracted_data = df
                    st.success("Data points extracted successfully!")

with col2:
    st.header("Ask Questions")
    
    if st.session_state.paligemma_model is None:
        st.warning("Please load the model first from the sidebar.")
    elif st.session_state.current_image is None:
        st.warning("Please upload a chart image or load a sample chart.")
    else:
        # Query input
        query = st.text_input("Ask a question about the chart:", 
                            placeholder="E.g., What is the highest value in the chart?")
        
        if query:
            if st.button("Analyze Chart"):
                with st.spinner("Analyzing chart "):
                    answer = analyze_chart_with_paligemma(
                        st.session_state.paligemma_model,
                        st.session_state.paligemma_processor,
                        st.session_state.device,
                        st.session_state.current_image,
                        query,
                        use_cot
                    )
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "question": query,
                        "answer": answer
                    })
                    
                    # Display answer
                    st.subheader("Answer")
                    st.write(answer)

# Display extracted data if available
if st.session_state.extracted_data is not None:
    st.header("Extracted Data Points")
    st.dataframe(st.session_state.extracted_data)
    
    # Download button for CSV
    st.markdown(get_csv_download_link(st.session_state.extracted_data), unsafe_allow_html=True)

# Display chat history
if st.session_state.chat_history:
    st.header("Question History")
    for i, qa in enumerate(reversed(st.session_state.chat_history)):
        with st.expander(f"Q: {qa['question']}", expanded=(i==0)):
            st.markdown(f"**A:** {qa['answer']}")