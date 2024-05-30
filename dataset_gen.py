import gradio as gr
from bs4 import BeautifulSoup
import requests
import csv
import json
from groq import Groq
import os

# Specify the CSV file path
csv_file_path = 'dataGroq.csv'

# Function to load and display webpage content
def load_and_display_webpage_content(url, model_name, save_to_csv):
    try:
        html = requests.get(url).text
        text_content = BeautifulSoup(html, "html.parser").get_text(strip=True, separator=' ')
        
        prompt = "Extract the title, price, brand, and retailer from the following context in JSON format."
        script_content = f"{prompt}:\n ```{text_content}```"
        
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful data format assistant that gives outputs only in JSON."},
                {"role": "user", "content": script_content},
            ],
            model=model_name,
            temperature=0,
            response_format={"type": "json_object"},
        )

        output_response = chat_completion.choices[0].message.content
        json_data = json.loads(output_response)
        retailer = json_data.get('retailer', '')

        if save_to_csv:
            # Open the CSV file in append mode
            file_exists = os.path.isfile(csv_file_path)
            with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
                csv_headers = ['input', 'output', 'retailer']
                writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
                if not file_exists or os.stat(csv_file_path).st_size == 0:
                    writer.writeheader()
                writer.writerow({'input': text_content, 'output': output_response, 'retailer': retailer})

        return text_content, output_response

    except Exception as e:
        return str(e), ""

# Create Gradio interface
url_input = gr.Textbox(label="Enter URL", type="text")
output_text = gr.Textbox(label="Webpage Content", show_copy_button=True)
# output_script = gr.Textbox(label="Prompt with content", show_copy_button=True)
final_output = gr.Textbox(label="Final output", show_copy_button=True)
dropdown = gr.Dropdown(
    choices=["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"],
    label="Select Model",
    value="llama3-8b-8192"
)
save_to_csv_checkbox = gr.Checkbox(label="Save response to CSV")

# Create Gradio app
gr.Interface(
    fn=load_and_display_webpage_content,
    inputs=[url_input, dropdown, save_to_csv_checkbox],
    outputs=[output_text, final_output],
    title="Webpage Content Viewer",
    allow_flagging=False  # Disables flagging
).launch()
