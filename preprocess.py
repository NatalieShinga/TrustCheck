
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    print("NLTK download error:", e)

def clean_text(text):
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove URLs, emails, phone numbers, special characters
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\b\d{10,}\b', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 3. Tokenize
    tokens = word_tokenize(text)
    
    # 4. Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # 5. Join back
    return ' '.join(tokens)

if __name__ == "__main__":
    sample = "URGENT! Apply now for scholarship. Pay K500 processing fee. WhatsApp 0977XXXXXX"
    print("Original:", sample)
    print("Cleaned:", clean_text(sample))