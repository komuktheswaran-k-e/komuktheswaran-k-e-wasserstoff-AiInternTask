import os
import logging
import time
from memory_profiler import memory_usage
from pymongo import MongoClient
from rake_nltk import Rake
from PyPDF2 import PdfReader
from collections import Counter
import re
from gensim import corpora
from gensim.models import LdaModel
import nltk
from concurrent.futures import ProcessPoolExecutor

nltk.download('punkt_tab')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# MongoDB setup
client = MongoClient('mongodb://localhost:27017/')
db = client['pdf_database']
collection = db['pdf_documents']

# Initialize keyword extractor
keyword_extractor = Rake()

def process_pdf(file_path):
    """Process the PDF file to extract text, summarize it, extract keywords, and topics."""
    text = extract_text_from_pdf(file_path)
    if text:
        summary = summarize_text(text)
        keywords = extract_keywords(text)
        topics = extract_topics(text)
        store_in_mongo(file_path, summary, keywords, topics)

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file."""
    try:
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text() + ' '
            return text.strip()
    except Exception as e:
        logging.error(f"Error extracting text from {file_path}: {e}")
        return None

def summarize_text(text, num_sentences=3):
    """Summarize the given text using a simple frequency-based method."""
    try:
        sentences = re.split(r'(?<=[.!?]) +', text)
        words = re.findall(r'\w+', text.lower())
        word_freq = Counter(words)
        # Score sentences based on word frequency
        sentence_scores = {}
        for sentence in sentences:
            for word in re.findall(r'\w+', sentence.lower()):
                if word in word_freq:
                    sentence_scores[sentence] = sentence_scores.get(sentence, 0) + word_freq[word]
        # Select the top N sentences
        summarized_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
        return ' '.join(summarized_sentences)
    except Exception as e:
        logging.error(f"Error generating summary: {e}")
        return "Error generating summary"

def extract_keywords(text):
    """Extract keywords from the given text using RAKE."""
    try:
        keyword_extractor.extract_keywords_from_text(text)
        keywords = keyword_extractor.get_ranked_phrases()
        return keywords
    except Exception as e:
        logging.error(f"Error extracting keywords: {e}")
        return []

def extract_topics(text, num_topics=3):
    """Extract topics from the text using LDA."""
    try:
        # Preprocess the text
        tokens = [word for word in re.findall(r'\w+', text.lower()) if len(word) > 3]
        dictionary = corpora.Dictionary([tokens])
        corpus = [dictionary.doc2bow(tokens)]
        # Create the LDA model
        lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)
        # Extract topics
        topics = lda_model.print_topics(num_words=3)
        return [topic for topic in topics]
    except Exception as e:
        logging.error(f"Error extracting topics: {e}")
        return []

def store_in_mongo(file_path, summary, keywords, topics):
    """Store the processed data in MongoDB."""
    doc_metadata = {
        "file_name": os.path.basename(file_path),
        "path": file_path,
        "summary": summary,
        "keywords": keywords,
        "topics": topics,
        "status": "processed"
    }
    try:
        collection.insert_one(doc_metadata)
        logging.info(f"Stored data for {file_path} in MongoDB.")
    except Exception as e:
        logging.error(f"Error storing in MongoDB: {e}")

def monitor_performance(func, *args):
    """Monitor the performance of a function."""
    start_time = time.time()
    mem_usage = memory_usage((func, args), interval=0.1)
    end_time = time.time()
    peak_memory = max(mem_usage) if isinstance(mem_usage, list) else mem_usage
    logging.info(f"Function {func.__name__} executed in {end_time - start_time:.2f} seconds with peak memory usage: {peak_memory:.2f} MB")

def process_all_pdfs(file_paths):
    """Process all PDFs in parallel."""
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_pdf, file_path) for file_path in file_paths]
        for future in futures:
            try:
                future.result()  # Wait for each future to complete
            except Exception as e:
                logging.error(f"Error processing PDF: {e}")

def main():
    """Main function to process all PDF files in the current directory and its subdirectories."""
    current_directory = os.getcwd()
    file_paths = []
    for root, _, files in os.walk(current_directory):
        for file_name in files:
            if file_name.endswith('.pdf'):
                file_paths.append(os.path.join(root, file_name))
    monitor_performance(process_all_pdfs, file_paths)

if __name__ == "__main__":
    main()
