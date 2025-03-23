# ChartQA

# Chart Q&A Application

## Overview
This Chart Q&A application allows users to analyze and extract information from chart images using the PaliGemma model. Users can upload chart images, ask questions about the charts, and extract structured data for further analysis.

## Features
- Upload chart images (PNG, JPG, JPEG)
- Load a sample chart for demonstration
- Ask natural language questions about chart content
- Extract data points from charts into a structured format
- Download extracted data as CSV
- Chain-of-Thought reasoning for improved analysis
- Question history tracking

## Requirements
- Python 3.8+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone this repository:
```bash
git clone https://github.com/sushantgai/ChartQA.git
cd ChartQA
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. Access the application in your web browser at http://localhost:8501

3. Usage steps:
   - Click "Load Model" in the sidebar to initialize the PaliGemma model
   - Upload a chart image or load the sample chart
   - Ask questions about the chart in the text input field
   - Click "Extract Data Points" to convert the chart into tabular data
   - Download the extracted data as CSV if needed

## Model Information

This application uses a fine-tuned version of the PaliGemma model specifically trained for chart understanding:
- Model: ahmed-masry/chartgemma
- The model can analyze various types of charts including bar charts, line charts, pie charts, and more

## Notes
- The first load of the model may take some time depending on your hardware
- GPU acceleration is automatically used if available, otherwise CPU is used
- Chain-of-Thought reasoning can be toggled on/off in the sidebar
- For best results, use clear images of charts with readable text and labels


## Acknowledgements
- This application uses the PaliGemma model fine-tuned for chart analysis
- Based on the transformers library from Hugging Face
