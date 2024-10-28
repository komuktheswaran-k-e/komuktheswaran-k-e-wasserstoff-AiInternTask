import json
import requests
import os
from urllib.parse import urlparse

def is_valid_url(url):
    """Check if the URL is valid."""
    parsed = urlparse(url)
    return all([parsed.scheme, parsed.netloc])

def download_pdfs_from_json(json_file, output_dir='downloads'):
    """Download PDFs from URLs specified in a JSON file."""
    
    # Load the JSON data
    with open(json_file, 'r') as file:
        urls_dict = json.load(file)

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Download each PDF
    for pdf_name, url in urls_dict.items():
        if not is_valid_url(url):
            print(f"Invalid URL for {pdf_name}: {url}. Skipping.")
            continue

        try:
            response = requests.get(url)
            if response.status_code == 200:
                file_name = os.path.join(output_dir, f"{pdf_name}.pdf")
                with open(file_name, 'wb') as output_file:
                    output_file.write(response.content)
                print(f"Downloaded: {file_name}")
            else:
                print(f"Failed to download {pdf_name}: Status code {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {pdf_name}: {e}")

if __name__ == "__main__":
    json_file_path = 'C:\\Users\\JL COMPUTERS\\Desktop\\AiInternTask\\Dataset.json'  # Change this to your actual JSON file path
    output_folder = 'C:\\Users\\JL COMPUTERS\\Desktop\\AiInternTask\\pdfs'     # Specify your output folder here
    
    download_pdfs_from_json(json_file_path, output_folder)
