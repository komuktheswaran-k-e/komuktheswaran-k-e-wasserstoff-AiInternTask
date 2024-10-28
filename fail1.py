import os
import fitz  # PyMuPDF for PDF reading
import json
import pymongo
import nltk
import requests
import time
import psutil  # Use psutil for memory usage tracking
from threading import Thread
import random

# Ensure to download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Set of English stop words
stop_words = set(nltk.corpus.stopwords.words('english'))

# MongoDB setup
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["pdf_database"]
collection = db["documents"]

# Define domain-specific keywords
domain_keywords = {
    # Technology Domain
    'technology': [
        'AI', 'neural networks', 'machine learning', 'big data', 'blockchain', 
        'computing', 'algorithms', 'cybersecurity', 'cloud computing', 'IoT',
        'quantum computing', '5G', 'robotics', 'automation', 'data science'
    ],
    
    # Medical Domain
    'medical': [
        'diagnosis', 'treatment', 'clinical', 'patients', 'therapy', 'surgery',
        'disease', 'drug', 'vaccine', 'epidemic', 'pandemic', 'biopsy', 
        'radiology', 'oncology', 'cardiology', 'neurology', 'immunology'
    ],
    
    # Legal Domain
    'legal': [
        'contract', 'litigation', 'legal', 'law', 'court', 'case', 'plaintiff', 
        'defendant', 'jurisdiction', 'verdict', 'appeal', 'intellectual property',
        'patent', 'trademark', 'copyright', 'arbitration', 'mediation', 'lawsuit'
    ],
    
    # Financial Domain
    'finance': [
        'investment', 'stock', 'bond', 'market', 'capital', 'interest', 
        'loan', 'credit', 'bank', 'financial', 'tax', 'revenue', 'expenditure', 
        'balance sheet', 'cryptocurrency', 'blockchain', 'equity', 'dividend'
    ],
    
    # Education Domain
    'education': [
        'curriculum', 'syllabus', 'learning', 'teaching', 'pedagogy', 'students', 
        'assessment', 'grading', 'classroom', 'online education', 'distance learning',
        'scholarship', 'tuition', 'degree', 'certification', 'accreditation'
    ],
    
    # Biology Domain
    'biology': [
        'cell', 'organism', 'genome', 'DNA', 'RNA', 'protein', 'enzymes', 
        'species', 'evolution', 'mutation', 'photosynthesis', 'biodiversity', 
        'ecosystem', 'biochemistry', 'microbiology', 'genetics', 'anatomy'
    ],
    
    # Environmental Science Domain
    'environmental science': [
        'climate change', 'global warming', 'sustainability', 'carbon footprint', 
        'greenhouse gases', 'pollution', 'biodiversity', 'conservation', 
        'deforestation', 'renewable energy', 'solar power', 'wind energy', 
        'ecosystem', 'habitat loss', 'recycling', 'waste management'
    ],
    
    # Sports Domain
    'sports': [
        'tournament', 'match', 'goal', 'team', 'player', 'coach', 'tactics', 
        'score', 'offense', 'defense', 'training', 'fitness', 'league', 
        'championship', 'Olympics', 'athlete', 'competition', 'injury'
    ],
    
    # Business Domain
    'business': [
        'entrepreneurship', 'startup', 'innovation', 'market', 'customer', 
        'revenue', 'profit', 'strategy', 'management', 'leadership', 'corporation', 
        'business model', 'product development', 'marketing', 'branding', 
        'operations', 'supply chain', 'HR', 'growth'
    ],
    
    # Psychology Domain
    'psychology': [
        'cognition', 'behavior', 'emotion', 'mental health', 'therapy', 'anxiety', 
        'depression', 'psychotherapy', 'counseling', 'cognitive-behavioral therapy', 
        'personality', 'perception', 'motivation', 'learning', 'memory', 
        'development', 'neuroscience', 'psychological testing', 'intelligence'
    ],
    
    # History Domain
    'history': [
        'ancient', 'medieval', 'modern', 'empire', 'dynasty', 'colonialism', 
        'war', 'revolution', 'democracy', 'industrialization', 'civilization', 
        'migration', 'trade', 'exploration', 'slavery', 'archaeology', 'monarchy'
    ],
    
    # Politics Domain
    'politics': [
        'government', 'policy', 'election', 'democracy', 'republic', 'constitution', 
        'parliament', 'senate', 'congress', 'legislation', 'voter', 'candidate', 
        'campaign', 'diplomacy', 'international relations', 'treaty', 'sanctions'
    ],
    
    # Economics Domain
    'economics': [
        'inflation', 'GDP', 'unemployment', 'trade', 'market', 'demand', 'supply', 
        'recession', 'economic growth', 'monetary policy', 'fiscal policy', 
        'labor market', 'consumer', 'producer', 'interest rates', 'exchange rates', 
        'globalization', 'capitalism', 'Keynesian economics'
    ],
    
    # Physics Domain
    'physics': [
        'quantum', 'relativity', 'force', 'energy', 'mass', 'gravity', 'thermodynamics', 
        'electromagnetism', 'particle', 'wave', 'nuclear', 'radiation', 'momentum', 
        'velocity', 'acceleration', 'cosmology', 'astrophysics', 'black hole', 
        'quantum mechanics', 'classical mechanics'
    ],
    
    # Chemistry Domain
    'chemistry': [
        'atom', 'molecule', 'reaction', 'bond', 'catalyst', 'acid', 'base', 
        'organic', 'inorganic', 'chemical equation', 'solution', 'compound', 
        'polymer', 'isotope', 'oxidation', 'reduction', 'enthalpy', 'stoichiometry'
    ],
    
    # Mathematics Domain
    'mathematics': [
        'algebra', 'calculus', 'geometry', 'trigonometry', 'probability', 'statistics', 
        'theorem', 'differential equations', 'combinatorics', 'matrix', 
        'set theory', 'topology', 'logic', 'number theory', 'real analysis', 
        'discrete mathematics', 'graph theory', 'linear algebra'
    ],
    
    # Arts and Literature Domain
    'arts and literature': [
        'painting', 'sculpture', 'drawing', 'art movement', 'realism', 'modernism', 
        'abstract', 'Renaissance', 'Baroque', 'Impressionism', 'literary analysis', 
        'poetry', 'novel', 'drama', 'theatre', 'narrative', 'protagonist', 
        'metaphor', 'symbolism', 'rhyme'
    ],
    
    # Music Domain
    'music': [
        'melody', 'harmony', 'rhythm', 'tempo', 'composition', 'genre', 
        'orchestra', 'symphony', 'concerto', 'chord', 'scale', 'octave', 
        'improvisation', 'jazz', 'classical', 'blues', 'opera', 'folk', 
        'rock', 'pop', 'hip hop', 'electronic'
    ],
    
    # Engineering Domain
    'engineering': [
        'design', 'prototype', 'systems', 'circuit', 'electronics', 'mechanical', 
        'electrical', 'software', 'aerospace', 'civil', 'chemical', 'materials', 
        'control systems', 'robotics', 'manufacturing', 'automotive', 'CAD', 
        'infrastructure', 'nanotechnology', 'renewable energy'
    ],
    
    # Geography Domain
    'geography': [
        'landform', 'continent', 'climate', 'oceanography', 'mountain', 
        'river', 'desert', 'plate tectonics', 'earthquake', 'volcano', 
        'latitude', 'longitude', 'cartography', 'urbanization', 'migration', 
        'population', 'environmental impact', 'biomes', 'hydrology'
    ]
}


# Function to log messages to a file
def log_message(message):
    with open('pdf_processing1.log', 'a') as log_file:
        log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

def log_message1(message):
    with open('pdf_processing2.log', 'a') as log_file:
        log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {'text'}-{message}\n")

# Function to download PDF files
def download_pdf(url, target_folder):
    try:
        response = requests.get(url, verify=False)  # Disable SSL verification
        response.raise_for_status()  # Raise an error for bad responses
        file_name = url.split("/")[-1]  # Get the file name from the URL
        file_path = os.path.join(target_folder, file_name)
        with open(file_path, 'wb') as pdf_file:
            pdf_file.write(response.content)
        log_message(f"Downloaded {file_name}")
        return file_path
    except requests.exceptions.HTTPError as http_err:
        log_message(f"HTTP error occurred while downloading {url}: {str(http_err)}")
    except requests.exceptions.RequestException as req_err:
        log_message(f"Request error occurred while downloading {url}: {str(req_err)}")
    except Exception as e:
        log_message(f"Unexpected error while downloading {url}: {str(e)}")
    return None

# Function to read PDF files and extract text
def read_pdf(file_path):
    try:
        text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
        return text
    except Exception as e:
        log_message(f"Error reading {file_path}: {str(e)}")
        return ""

# Function to compute TF (Term Frequency) manually
def compute_tf(text):
    words = nltk.word_tokenize(text.lower())
    total_words = len(words)
    word_freq = {}

    for word in words:
        if word not in stop_words:
            word_freq[word] = word_freq.get(word, 0) + 1

    # Calculate TF for each word
    tf_scores = {word: freq / total_words for word, freq in word_freq.items()}
    return tf_scores

# Function to create a basic keyword extraction based on TF
def extract_keywords(text):
    tf_scores = compute_tf(text)
    sorted_keywords = sorted(tf_scores.items(), key=lambda item: item[1], reverse=True)
    return [word for word, score in sorted_keywords if word not in stop_words][:5]  # Top 5 keywords

# Custom Summarization Function
def custom_summarization(text, page_count):
    sentences = nltk.sent_tokenize(text)
    num_sentences = 3 if page_count < 5 else 5 if page_count < 20 else 7

    tf_scores = compute_tf(text)

    # Score sentences based on TF
    sentence_scores = {}
    for sentence in sentences:
        score = sum(tf_scores.get(word, 0) for word in nltk.word_tokenize(sentence.lower()))
        sentence_scores[sentence] = score

    # Select the top sentences based on scores
    top_sentences = sorted(sentence_scores.items(), key=lambda item: item[1], reverse=True)
    summary = ' '.join(sentence for sentence, score in top_sentences[:num_sentences])
    return summary

# Function to detect the domain based on keywords
def detect_domain(text):
    domain_count = {domain: 0 for domain in domain_keywords.keys()}  # Initialize domain counts

    words = nltk.word_tokenize(text.lower())
    for word in words:
        for domain, keywords in domain_keywords.items():
            if word in keywords:
                domain_count[domain] += 1

    # Determine the domain with the highest count
    detected_domain = max(domain_count, key=domain_count.get)
    return detected_domain if domain_count[detected_domain] > 0 else "Unknown"

# Function to process a single PDF
def process_pdf(pdf_file, pdf_folder, results):
    file_path = download_pdf(pdf_file, pdf_folder)  # Download the PDF

    if not file_path:
        results.append((pdf_file, False, "Download failed"))
        return

    try:
        text = read_pdf(file_path)
        if not text:  # Check if text extraction failed
            results.append((file_path, False, "Text extraction failed"))
            return

        page_count = text.count('\n\n') + 1  # Estimate page count based on paragraph breaks
        domain = detect_domain(text)  # Detect the domain
        summary = custom_summarization(text, page_count)
        log_message1(text)
        keywords = extract_keywords(text)

        # Prepare MongoDB document
        document_data = {
            "document_name": os.path.basename(file_path),
            "path": file_path,
            "size": os.path.getsize(file_path),
            "summary": summary,
            "keywords": keywords,
            "domain": domain  # Add detected domain to the document
        }

        # Store in MongoDB
        collection.insert_one(document_data)

        results.append((os.path.basename(file_path), True))  # Return the filename and success status
    except Exception as e:
        log_message(f"Error processing {file_path}: {str(e)}")
        results.append((os.path.basename(file_path), False, str(e)))  # Return the filename, failure status, and error message

# Custom memory usage tracking function
def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024  # Returns memory usage in kilobytes

# Function to process PDFs concurrently and measure performance
def process_pdfs_concurrently(pdf_urls, pdf_folder):
    start_time = time.time()  # Start timing the processing
    results = []

    # Get initial memory usage
    initial_memory = get_memory_usage()

    threads = []
    for pdf_url in pdf_urls:
        thread = Thread(target=process_pdf, args=(pdf_url, pdf_folder, results))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()  # Wait for all threads to complete

    end_time = time.time()  # End timing the processing
    total_time = end_time - start_time
    final_memory = get_memory_usage()

    memory_used = final_memory - initial_memory

    print(f"Processed {len(pdf_urls)} files in {total_time:.2f} seconds.")
    print(f"Memory used: {memory_used:.2f} KB")

    for result in results:
        if result[1] is False:
            print(f"Failed to process {result[0]}: {result[2] if len(result) > 2 else ''}")

# Example usage
if __name__ == "__main__":
    pdf_folder = "C:\\Users\\II-Year.MCA-HP-156\\Desktop\\AiInternTask\\pdfs"  # Path to the folder where PDFs will be downloaded
    json_file = "C:\\Users\\II-Year.MCA-HP-156\\Downloads\\Dataset.json"  # Update this with your actual JSON file path

    # Load PDF URLs from the JSON file
    with open(json_file, 'r') as f:
        pdf_urls = json.load(f).values()  # Extracting URLs from JSON
        log_message(f"Loaded PDF URLs: {pdf_urls}")

    process_pdfs_concurrently(pdf_urls, pdf_folder)
