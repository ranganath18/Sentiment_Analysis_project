from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib, re

app = FastAPI()

# --- STOPWORDS ---
STOPWORDS = set([
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','yourselves','he','him','his','himself','she','her','hers',
    'herself','it','its','itself','they','them','their','theirs','themselves',
    'what','which','who','whom','this','that','these','those','am','is','are',
    'was','were','be','been','being','have','has','had','having','do','does',
    'did','doing','a','an','the','and','but','if','or','because','as','until',
    'while','of','at','by','for','with','about','against','between','into',
    'through','during','before','after','above','below','to','from','up',
    'down','in','out','on','off','over','under','again','further','then',
    'once','here','there','when','where','why','how','all','both','each',
    'few','more','most','other','some','such','no','nor','not','only','own',
    'same','so','than','too','very','s','t','can','will','just','don',
    'should','now','d','ll','m','o','re','ve','y','ain','aren','couldn',
    'didn','doesn','hadn','hasn','haven','isn','mightn','mustn','needn',
    'shan','shouldn','wasn','weren','won','wouldn','also','would','could',
    'shall','may','might','must','need','dare','used','br','film','movie'
])

# --- HELPER FUNCTIONS ---
def remove_html_tags(text):
    return re.sub(r'<[^>]+>', ' ', text)

def remove_urls(text):
    return re.sub(r'http\S+|www\.\S+', '', text)

def remove_special_characters(text):
    return re.sub(r'[^a-z\s]', '', text)

def tokenize(text):
    return text.split()

def remove_stopwords(tokens):
    return [t for t in tokens if t not in STOPWORDS]

def simple_stem(word):
    suffixes = ['ing','tion','ness','ment','able','ible','ful','less',
                'ous','ive','er','est','ed','ly','s']
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word

def apply_stemming(tokens):
    return [simple_stem(t) for t in tokens]

# --- MAIN PREPROCESS PIPELINE ---
def preprocess(text):
    text = text.lower()
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_special_characters(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = apply_stemming(tokens)
    tokens = [t for t in tokens if len(t) > 2]
    return ' '.join(tokens)

# --- LOAD MODEL ---
model = joblib.load("LR_model.pkl")
vectorizer = joblib.load("LR_vectorizer.pkl")

# --- SCHEMAS ---
class ReviewRequest(BaseModel):
    text: str

class ReviewResponse(BaseModel):
    sentiment: str
    confidence: float
    label: int

class BulkReviewRequest(BaseModel):
    reviews: List[str]

class SinglePrediction(BaseModel):
    text: str
    sentiment: str
    confidence: float
    label: int

class BulkReviewResponse(BaseModel):
    predictions: List[SinglePrediction]
    summary: dict

# --- ROUTES ---
@app.get("/")
def home():
    return {"message":"API running"}
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=ReviewResponse)
def predict(req: ReviewRequest):
    cleaned = preprocess(req.text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    return {
        "sentiment": "POSITIVE" if pred == 1 else "NEGATIVE",
        "confidence": round(float(max(prob)), 4),
        "label": int(pred)
    }

# OPTIMISED: single batch transform+predict instead of per-review loop
@app.post("/predict/bulk", response_model=BulkReviewResponse)
def predict_bulk(req: BulkReviewRequest):
    reviews = [r.strip() for r in req.reviews if r.strip()]

    cleaned_texts = [preprocess(r) for r in reviews]   # list comprehension
    vecs  = vectorizer.transform(cleaned_texts)          # one sparse matrix
    preds = model.predict(vecs)                          # one call
    probs = model.predict_proba(vecs)                    # one call

    predictions = [
        {
            "text":       reviews[i],
            "sentiment":  "POSITIVE" if preds[i] == 1 else "NEGATIVE",
            "confidence": round(float(max(probs[i])), 4),
            "label":      int(preds[i])
        }
        for i in range(len(reviews))
    ]

    pos_count = sum(1 for p in predictions if p["sentiment"] == "POSITIVE")
    neg_count = len(predictions) - pos_count
    avg_conf  = round(sum(p["confidence"] for p in predictions) / len(predictions), 4)
    return {
        "predictions": predictions,
        "summary": {
            "total": len(predictions),
            "positive_count": pos_count,
            "negative_count": neg_count,
            "positive_percentage": round((pos_count / len(predictions)) * 100, 1),
            "avg_confidence": avg_conf
        }
    }