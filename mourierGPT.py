import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from googlesearch import search
import requests
from bs4 import BeautifulSoup
from collections import Counter
from string import punctuation
from nltk.corpus import stopwords
import nltk
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
nltk.download('stopwords')
stopwords = set(stopwords.words('english'))
heuristics = ["fake news", "conspiracy theory", "unverified", "baseless claim"]
min_reliability_score = 0.7 # set a minimum reliability score threshold

def generate_text(prompt, max_length=100, top_k=50):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    sample_output = model.generate(input_ids, do_sample=True, max_length=max_length, top_k=top_k)

    generated_text = tokenizer.decode(sample_output[0], skip_special_tokens=True)
    for heuristic in heuristics:
        if heuristic in generated_text.lower():
            return None
    return generated_text

def get_search_results(query, num_results=5):
    results = []
    for j in search(query, num_results=num_results):
        results.append(j)
    return results

def get_reliability_score(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        reliability_score = 1.0
        # Check for a known unreliable source in the URL
        if "infowars.com" in url:
            reliability_score = 0.1
        # Check for specific words on the page indicating a lack of reliability
        elif "satire" in soup.text.lower() or "parody" in soup.text.lower():
            reliability_score = 0.3
        # Use a more sophisticated algorithm to compute a reliability score based on page content
        else:
            reliability_score = compute_reliability_score(soup)
        return reliability_score
    except:
        return 0.0

def compute_reliability_score(soup):
    text = soup.get_text().lower()
    words = text.split()
    words = [word.strip(punctuation) for word in words]
    words = [word for word in words if word.isalpha()]
    words = [word for word in words if word not in stopwords]

    # Count occurrences of specific words
    word_counts = Counter(words)
    conspiracy_count = word_counts['conspiracy'] + word_counts['conspiracies']
    hoax_count = word_counts['hoax'] + word_counts['hoaxes']
    false_count = word_counts['false'] + word_counts['fake']
    unverified_count = word_counts['unverified'] + word_counts['unsubstantiated']
    baseless_count = word_counts['baseless'] + word_counts['groundless']

    # Assign a reliability score based on the frequency of specific words
    if conspiracy_count > 5:
        return 0.1
    elif hoax_count > 5:
        return 0.3
    elif false_count > 5:
        return 0.5
    elif unverified_count > 5:
        return 0.7
    elif baseless_count > 5:
        return 0.9
    else:
        return 1.0

prompt = input("Enter a prompt: ")
search_query = prompt + " -site:wikipedia.org -site:youtube.com" # exclude certain websites from the search results
search_results = get_search_results(search_query)
for result in search_results:
    reliability_score = get_reliability_score(result)
    if reliability_score >= min_reliability_score:
        generated_text = generate_text(result)
        if generated_text:
            print(result + ":")
            print(generated_text)
            break # show the first generated text that passes the heuristics and reliability filters
