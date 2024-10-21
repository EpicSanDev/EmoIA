import logging
import sqlite3
import time
import threading
import schedule
import requests
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext, CallbackQueryHandler
import datetime
from openai import OpenAI
from bs4 import BeautifulSoup
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from textblob import TextBlob
import random
import wikipedia
import torch
from transformers import AutoTokenizer, AutoModel, pipeline, GPT2LMHeadModel, GPT2Tokenizer, T5Tokenizer, T5ForConditionalGeneration
import re
from collections import deque
import struct
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine
import ast
import emoji
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import DistilBertTokenizer, DistilBertModel
import sys
import traceback
from langdetect import detect
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import pytz
from fuzzywuzzy import fuzz
import spacy
import yfinance as yf
from newsapi import NewsApiClient
import wolframalpha
from dotenv import load_dotenv
from scipy.special import softmax
import networkx as nx
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from textstat import flesch_kincaid_grade
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VaderSentimentIntensityAnalyzer
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import io
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import pickle
import tensorflow as tf
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import shutil
from huggingface_hub import snapshot_download
from transformers import TrainingArguments, Trainer
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
import math
import PyPDF2
from docx import Document
import csv
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from gensim.utils import simple_preprocess
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from io import BytesIO

from datetime import datetime, timedelta, time as datetime_time
from collections import Counter




# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Chargement des variables d'environnement
load_dotenv()

# Configuration
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
YOUR_CHAT_ID = os.getenv('YOUR_CHAT_ID')
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')
WOLFRAM_ALPHA_APP_ID = os.getenv('WOLFRAM_ALPHA_APP_ID')
ENCRYPTION_KEY = os.getenv('ENCRYPTION_KEY').encode()

# Configuration OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# Connexion à la base de données
conn = sqlite3.connect('ai_companion.db', check_same_thread=False)
c = conn.cursor()

# Initialisation de NLTK
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Chargement des modèles pré-entraînés
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")
sentiment_analyzer = pipeline("sentiment-analysis")
ner_pipeline = pipeline("ner")
nlp = spacy.load("en_core_web_sm")
topic_model = LatentDirichletAllocation(n_components=10, random_state=42)
count_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
vader_analyzer = VaderSentimentIntensityAnalyzer()

# Modèle T5 pour la génération de texte et la summarisation
t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")

# Modèle VGG16 pour l'analyse d'images
vgg_model = VGG16(weights='imagenet', include_top=False)

# Configurations globales
CURRENT_EMBEDDING_VERSION = "distilbert-base-uncased-v2"
MEMORY_BUFFER_SIZE = 1000
LONG_TERM_MEMORY_THRESHOLD = 0.7
CONTEXT_RELEVANCE_THRESHOLD = 0.6
ITEMS_PER_PAGE = 5

# Initialisation du tokenizer et du modèle GPT-2
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")

# Classes principales
class ContextManager:
    def __init__(self, max_size=10):
        self.context = deque(maxlen=max_size)
        self.topic_model = None
        self.vectorizer = None

    def add_to_context(self, text):
        self.context.append(text)
        if len(self.context) == self.context.maxlen:
            self.update_topic_model()

    def update_topic_model(self):
        texts = list(self.context)
        self.vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        dtm = self.vectorizer.fit_transform(texts)
        self.topic_model = LatentDirichletAllocation(n_components=3, random_state=42)
        self.topic_model.fit(dtm)

    def get_current_topics(self):
        if self.topic_model is None or self.vectorizer is None:
            return []
        dtm = self.vectorizer.transform(list(self.context))
        topic_distribution = self.topic_model.transform(dtm)
        return topic_distribution.mean(axis=0).tolist()

context_manager = ContextManager()

class MemoryBuffer:
    def __init__(self, max_size=MEMORY_BUFFER_SIZE):
        self.buffer = deque(maxlen=max_size)

    def add(self, item):
        self.buffer.append(item)

    def get_recent(self, n):
        return list(self.buffer)[-n:]

    def get_all(self):
        return list(self.buffer)

memory_buffer = MemoryBuffer()

class LongTermMemory:
    def __init__(self, capacity=1000):
        self.memory = deque(maxlen=capacity)
        self.importance_threshold = 0.7

    def add_memory(self, text, importance):
        if importance > self.importance_threshold:
            self.memory.append((text, importance, time.time()))

    def get_relevant_memories(self, query, n=5):
        query_embedding = get_embedding(query)
        similarities = []
        for memory, importance, timestamp in self.memory:
            memory_embedding = get_embedding(memory)
            similarity = 1 - cosine(query_embedding, memory_embedding)
            time_factor = 1 / (1 + 0.1 * (time.time() - timestamp) / 86400)  # Decay factor
            adjusted_similarity = similarity * importance * time_factor
            similarities.append((memory, adjusted_similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [memory for memory, _ in similarities[:n]]

long_term_memory = LongTermMemory()

class ConversationManager:
    def __init__(self):
        self.conversations = {}
        self.last_interaction = {}

    def add_message(self, user_id, message, is_user=True):
        if user_id not in self.conversations:
            self.conversations[user_id] = []
        self.conversations[user_id].append(("User" if is_user else "AI", message))
        self.last_interaction[user_id] = time.time()

    def get_recent_conversation(self, user_id, n=5):
        return self.conversations.get(user_id, [])[-n:]

    def should_initiate_conversation(self, user_id):
        last_time = self.last_interaction.get(user_id, 0)
        return time.time() - last_time > 3600  # 1 hour

conversation_manager = ConversationManager()

class ReinforcementLearner:
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.q_table = {}
        self.actions = actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon

    def get_state(self, user_input, user_profile):
        return hash(f"{user_input}_{json.dumps(user_profile)}")

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)
        else:
            if state not in self.q_table:
                self.q_table[state] = {action: 0 for action in self.actions}
            return max(self.q_table[state], key=self.q_table[state].get)

    def update(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = {action: 0 for action in self.actions}
        
        current_q = self.q_table[state][action]
        next_max_q = max(self.q_table.get(next_state, {action: 0 for action in self.actions}).values())
        new_q = current_q + self.lr * (reward + self.gamma * next_max_q - current_q)
        self.q_table[state][action] = new_q

rl_learner = ReinforcementLearner(['empathize', 'inform', 'question', 'suggest'])

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
    
    def add_entity(self, entity, properties=None):
        self.graph.add_node(entity, properties=properties or {})
    
    def add_relationship(self, entity1, entity2, relationship):
        self.graph.add_edge(entity1, entity2, relationship=relationship)
    
    def get_related_entities(self, entity):
        return list(self.graph.neighbors(entity))
    
    def get_entity_properties(self, entity):
        return self.graph.nodes[entity].get('properties', {})

knowledge_graph = KnowledgeGraph()

# Fonctions principales
def setup_database(cursor):
    cursor.execute('''CREATE TABLE IF NOT EXISTS memory
                 (timestamp TEXT, user_id TEXT, user_input TEXT, bot_response TEXT, emotion TEXT, intent TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS user_profile
                 (user_id TEXT, key TEXT, value TEXT, PRIMARY KEY (user_id, key))''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS tasks
                 (id INTEGER PRIMARY KEY, user_id TEXT, task TEXT, due_date TEXT, status TEXT, priority TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS daily_reflections
                 (date TEXT, user_id TEXT, reflection TEXT, PRIMARY KEY (date, user_id))''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS knowledge_base
                 (id INTEGER PRIMARY KEY, timestamp TEXT, title TEXT, content TEXT, source TEXT, category TEXT, tags TEXT, relevance_score REAL DEFAULT 0, rating_count INTEGER DEFAULT 0)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS generated_functions
                 (name TEXT PRIMARY KEY, code TEXT, description TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS long_term_memory
                 (timestamp TEXT, user_id TEXT, text TEXT, vector BLOB, importance REAL)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS context_understanding
                 (timestamp TEXT, user_id TEXT, text TEXT, embedding BLOB)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS personal_goals
                 (id INTEGER PRIMARY KEY, user_id TEXT, goal_description TEXT, target_date TEXT, status TEXT, progress REAL)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS calendar_events
                 (id TEXT PRIMARY KEY, user_id TEXT, summary TEXT, start_time TEXT, 
                  end_time TEXT, description TEXT, location TEXT, last_synced TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS knowledge_base
                 (id INTEGER PRIMARY KEY, 
                  timestamp TEXT, 
                  title TEXT, 
                  content TEXT, 
                  source TEXT, 
                  category TEXT,
                  tags TEXT, 
                  relevance_score REAL DEFAULT 0, 
                  rating_count INTEGER DEFAULT 0)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS journal_entries
                 (id INTEGER PRIMARY KEY, user_id TEXT, date TEXT, content TEXT, mood TEXT, activities TEXT, goals TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS calendar_events
                 (id INTEGER PRIMARY KEY, user_id TEXT, event_name TEXT, start_time TEXT, end_time TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS personal_goals
                 (id INTEGER PRIMARY KEY, user_id TEXT, goal_description TEXT, status TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS proactivity_logs
                 (id INTEGER PRIMARY KEY, user_id TEXT, timestamp TEXT, score REAL, decision INTEGER)''')
 
    conn.commit()


def initialize_gpt2():
    global gpt2_tokenizer, gpt2_model
    
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    # Définir le token de remplissage
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
    gpt2_model.config.pad_token_id = gpt2_tokenizer.eos_token_id

def get_embedding(text):
    tokens = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=512)
    input_ids = torch.tensor([tokens])
    
    with torch.no_grad():
        outputs = model(input_ids)
    
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def update_existing_embeddings():
    tables = [
        ("long_term_memory", "vector"),
        ("context_understanding", "embedding")
    ]
    
    for table, column in tables:
        c.execute(f"SELECT rowid, text, {column} FROM {table}")
        for rowid, text, serialized_data in c.fetchall():
            try:
                old_embedding, version = deserialize_embedding(serialized_data)
                if version != CURRENT_EMBEDDING_VERSION:
                    new_embedding = get_embedding(text)
                    new_serialized_data = serialize_embedding(new_embedding)
                    c.execute(f"UPDATE {table} SET {column} = ? WHERE rowid = ?", (new_serialized_data, rowid))
            except Exception as e:
                logger.error(f"Error updating {table} embedding for rowid {rowid}: {e}")

    conn.commit()
def serialize_embedding(embedding, version=CURRENT_EMBEDDING_VERSION):
    metadata = json.dumps({"version": version, "shape": embedding.shape}).encode('utf-8')
    return struct.pack('<I', len(metadata)) + metadata + embedding.tobytes()

def deserialize_embedding(binary_data):
    metadata_len = struct.unpack('<I', binary_data[:4])[0]
    metadata = json.loads(binary_data[4:4+metadata_len].decode('utf-8'))
    embedding = np.frombuffer(binary_data[4+metadata_len:], dtype=np.float32).reshape(metadata['shape'])
    return embedding, metadata['version']

def store_long_term_memory(user_id, text, importance):
    vector = get_embedding(text)
    serialized_vector = serialize_embedding(vector)
    c.execute("""
        INSERT INTO long_term_memory (timestamp, user_id, text, vector, importance)
        VALUES (?, ?, ?, ?, ?)
    """, (time.time(), user_id, text, serialized_vector, importance))
    conn.commit()

def retrieve_long_term_memory(user_id, query, n=5):
    query_vector = get_embedding(query)
    c.execute("SELECT text, vector, importance FROM long_term_memory WHERE user_id = ?", (user_id,))
    results = c.fetchall()
    
    similarities = []
    for text, serialized_vector, importance in results:
        vector, version = deserialize_embedding(serialized_vector)
        if version != CURRENT_EMBEDDING_VERSION:
            vector = get_embedding(text)
        similarity = 1 - cosine(query_vector, vector)
        similarities.append((text, similarity, importance))
    
    similarities.sort(key=lambda x: x[1] * x[2], reverse=True)
    return [text for text, _, _ in similarities[:n]]

def update_context_understanding(user_id, text):
    embedding = get_embedding(text)
    serialized_embedding = serialize_embedding(embedding)
    c.execute("""
        INSERT INTO context_understanding (timestamp, user_id, text, embedding)
        VALUES (?, ?, ?, ?)
    """, (time.time(), user_id, text, serialized_embedding))
    conn.commit()

def get_relevant_context(user_id, query, n=5):
    query_embedding = get_embedding(query)
    c.execute("SELECT text, embedding FROM context_understanding WHERE user_id = ? ORDER BY timestamp DESC LIMIT 100", (user_id,))
    results = c.fetchall()
    
    similarities = []
    for text, serialized_embedding in results:
        embedding, version = deserialize_embedding(serialized_embedding)
        if version != CURRENT_EMBEDDING_VERSION:
            embedding = get_embedding(text)
        similarity = 1 - cosine(query_embedding, embedding)
        if similarity > CONTEXT_RELEVANCE_THRESHOLD:
            similarities.append((text, similarity))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [text for text, _ in similarities[:n]]

def get_gpt4_response(prompt, max_tokens=2000, temperature=0.7):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a highly intelligent, adaptable, and proactive AI assistant with emotional intelligence and contextual understanding."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error in getting GPT-4 response: {e}")
        return "I apologize, I'm having trouble processing that right now. Could you please try again?"

def update_user_profile(user_id, key, value):
    if isinstance(value, list) or isinstance(value, dict):
        value = json.dumps(value)
    else:
        value = str(value)
    
    c.execute("INSERT OR REPLACE INTO user_profile (user_id, key, value) VALUES (?, ?, ?)", (user_id, key, value))
    conn.commit()

def get_user_profile(user_id):
    c.execute("SELECT key, value FROM user_profile WHERE user_id = ?", (user_id,))
    return dict(c.fetchall())

def add_task(user_id, task, due_date, priority="medium"):
    c.execute("INSERT INTO tasks (user_id, task, due_date, status, priority) VALUES (?, ?, ?, ?, ?)",
              (user_id, task, due_date, "pending", priority))
    conn.commit()

def get_pending_tasks(user_id):
    c.execute("SELECT task, due_date, priority FROM tasks WHERE user_id = ? AND status = 'pending' ORDER BY priority DESC, due_date ASC", (user_id,))
    return c.fetchall()

def add_daily_reflection(user_id, reflection):
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    c.execute("INSERT OR REPLACE INTO daily_reflections (date, user_id, reflection) VALUES (?, ?, ?)", (today, user_id, reflection))
    conn.commit()

def analyze_personality(user_id):
    c.execute("SELECT user_input FROM memory WHERE user_id = ? ORDER BY timestamp DESC LIMIT 1000", (user_id,))
    user_inputs = c.fetchall()
    
    if not user_inputs:
        return {
            'openness': 0,
            'conscientiousness': 0,
            'extraversion': 0,
            'agreeableness': 0,
            'neuroticism': 0
        }
    
    total_inputs = len(user_inputs)
    
    return {
        'openness': sum('new' in inp[0].lower() or 'creative' in inp[0].lower() for inp in user_inputs) / total_inputs,
        'conscientiousness': sum('plan' in inp[0].lower() or 'organize' in inp[0].lower() for inp in user_inputs) / total_inputs,
        'extraversion': sum('friend' in inp[0].lower() or 'party' in inp[0].lower() for inp in user_inputs) / total_inputs,
        'agreeableness': sum('help' in inp[0].lower() or 'kind' in inp[0].lower() for inp in user_inputs) / total_inputs,
        'neuroticism': sum('worry' in inp[0].lower() or 'anxious' in inp[0].lower() for inp in user_inputs) / total_inputs
    }

def analyze_user_behavior(user_input, user_id):
    sentiment = TextBlob(user_input).sentiment
    
    behavior_analysis = {
        "polarity": sentiment.polarity,
        "subjectivity": sentiment.subjectivity,
        "word_count": len(user_input.split()),
        "question": '?' in user_input,
        "exclamation": '!' in user_input,
        "uppercase_ratio": sum(1 for c in user_input if c.isupper()) / len(user_input) if len(user_input) > 0 else 0,
        "emoji_count": sum(c in emoji.EMOJI_DATA for c in user_input),
        "personality": analyze_personality(user_id)
    }
    
    return behavior_analysis

# Modification de la fonction analyze_emotion
def analyze_emotion(text):
    try:
        result = sentiment_analyzer(text)[0]
        sentiment_label = result['label']
        sentiment_score = result['score']

        vader_sentiment = sia.polarity_scores(text)

        blob = TextBlob(text)
        subjectivity = blob.sentiment.subjectivity

        if sentiment_label == 'POSITIVE':
            emotion = 'joy'
        elif sentiment_label == 'NEGATIVE':
            emotion = 'sadness'
        else:
            emotion = 'neutral'

        return {
            'emotion': emotion,
            'sentiment': {
                'label': sentiment_label,
                'score': sentiment_score,
                'vader': vader_sentiment,
                'compound': vader_sentiment['compound']  # Ajout de la valeur 'compound'
            },
            'subjectivity': subjectivity
        }
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse de l'émotion: {str(e)}")
        return {
            'emotion': 'unknown',
            'sentiment': {'label': 'unknown', 'score': 0, 'vader': {'compound': 0}, 'compound': 0},
            'subjectivity': 0
        }

def generate_emotional_response(user_input, user_profile, user_analysis, relevant_context, relevant_memories):
    emotional_analysis = analyze_emotion(user_input)
    
    prompt = f"""
    User Input: {user_input}
    User Profile: {json.dumps(user_profile)}
    User Behavior Analysis: {json.dumps(user_analysis)}
    Emotional Analysis: {json.dumps(emotional_analysis)}
    Relevant Context: {' '.join(relevant_context)}
    Relevant Memories: {' '.join(relevant_memories)}

    Based on the above information, generate a highly emotional, intelligent, and personalized response. 
    Consider the following aspects:
    1. Mirror the user's emotional state, but slightly amplify positive emotions and slightly dampen negative ones.
    2. Use language and tone that aligns with the user's personality traits and emotional state.
    3. Include appropriate emojis based on the emotional context.
    4. Express deep empathy and understanding towards the user's situation.
    5. If appropriate, share a personal anecdote or feeling to deepen the emotional connection.
    6. Incorporate relevant context and memories to make the response more personalized and meaningful.
    7. Suggest proactive actions or topics that might improve the user's emotional state.
    8. If the user seems to be facing a challenge, offer supportive advice or encouragement.
    9. Adapt the complexity of your language to match the user's typical communication style and current emotional state.
    10. If appropriate, use gentle humor or light-hearted comments to enhance engagement and improve mood.

    Ensure the response is genuine, supportive, and encourages further emotional and intellectual engagement.
    
    Response (keep it under 200 words):
    """
    
    response = get_gpt4_response(prompt, max_tokens=250)
    return response

def advanced_semantic_analysis(text):
    doc = nlp(text)
    
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    syntactic_structure = [(token.text, token.dep_, token.head.text) for token in doc]
    
    keywords = [token.text for token in doc if token.is_stop == False and token.is_punct == False]
    
    complexity_score = flesch_kincaid_grade(text)
    
    return {
        'entities': entities,
        'syntactic_structure': syntactic_structure,
        'keywords': keywords,
        'complexity_score': complexity_score
    }

def generate_intelligent_response(user_input, user_profile, user_analysis, relevant_context, relevant_memories, semantic_analysis):
    emotional_analysis = analyze_emotion(user_input)
    current_topics = context_manager.get_current_topics()
    
    prompt = f"""
    User Input: {user_input}
    User Profile: {json.dumps(user_profile)}
    User Behavior Analysis: {json.dumps(user_analysis)}
    Emotional Analysis: {json.dumps(emotional_analysis)}
    Semantic Analysis: {json.dumps(semantic_analysis)}
    Relevant Context: {' '.join(relevant_context)}
    Relevant Memories: {' '.join(relevant_memories)}
    Current Conversation Topics: {json.dumps(current_topics)}

    Based on the above information, generate a highly intelligent, emotionally aware, and contextually relevant response. 
    Consider the following aspects:
    1. Address the user's specific query or intent, taking into account the semantic structure and entities identified.
    2. Incorporate relevant background knowledge and context from the user's profile and conversation history.
    3. Adapt the response complexity to match the user's demonstrated language level and preferences.
    4. Show emotional intelligence by acknowledging and responding appropriately to the user's emotional state.
    5. Make connections between current topics and previously discussed subjects or user interests.
    6. If appropriate, ask insightful follow-up questions to deepen the conversation or clarify ambiguities.
    7. Provide novel insights or perspectives that could expand the user's understanding of the topic.
    8. If relevant, incorporate factual information or data to support your response.
    9. Maintain consistency with previous interactions while introducing new, relevant information.
    10. Tailor the tone and style of the response to the user's preferences and the current context of the conversation.

    Ensure the response is coherent, informative, and encourages further intellectual and emotional engagement.
    
    Response (aim for about 150 words, but adjust based on the complexity of the query):
    """
    
    response = get_gpt4_response(prompt, max_tokens=250)
    return response

def handle_message(update, context):
    user_input = update.message.text
    user_id = update.effective_user.id
    
    try:
        lang = detect(user_input)
        
        user_profile = get_user_profile(user_id)
        if not user_profile:
            logger.warning(f"No profile found for user {user_id}. Initializing new profile.")
            user_profile = initialize_user_profile(user_id)

        user_analysis = analyze_user_behavior(user_input, user_id)
        relevant_context = get_relevant_context(user_id, user_input, n=2)
        relevant_memories = long_term_memory.get_relevant_memories(user_input, n=2)
        semantic_analysis = advanced_semantic_analysis(user_input)
        
        context_manager.add_to_context(user_input)
        
        # Recherche dans la base de connaissances
        relevant_knowledge = search_knowledge_base(user_input)
        
        # Ajout des connaissances pertinentes au contexte
        knowledge_context = "\n".join([f"From {k[2]}: {k[1]}" for k in relevant_knowledge])
        
        # Intégration des connaissances dans la génération de réponse
        prompt = f"""
        User Input: {user_input}
        User Profile: {json.dumps(user_profile)}
        User Behavior Analysis: {json.dumps(user_analysis)}
        Emotional Analysis: {json.dumps(analyze_emotion(user_input))}
        Semantic Analysis: {json.dumps(semantic_analysis)}
        Relevant Context: {' '.join(relevant_context)}
        Relevant Memories: {' '.join(relevant_memories)}
        Relevant Knowledge: {knowledge_context}

        Based on all the above information, including the newly learned knowledge, generate a highly intelligent, emotionally aware, and contextually relevant response.
        
        Response:
        """
        
        response = get_gpt4_response(prompt, max_tokens=250)
        
        update_user_profile_from_interaction(user_id, user_input, response, user_analysis, semantic_analysis)
        train_intelligent_model(user_input, response, user_analysis, semantic_analysis)

        importance = calculate_importance(user_input, response, user_analysis, semantic_analysis)
        long_term_memory.add_memory(f"User: {user_input}\nAI: {response}", importance)

        update_context_understanding(user_id, f"User: {user_input}\nAI: {response}")
        conversation_manager.add_message(user_id, user_input, is_user=True)
        conversation_manager.add_message(user_id, response, is_user=False)

        c.execute("INSERT INTO memory VALUES (?, ?, ?, ?, ?, ?)",
                  (time.time(), user_id, user_input, response,
                   json.dumps(analyze_emotion(user_input)),
                   classify_response(response)))
        conn.commit()

        check_proactive_actions(update, context, user_id, user_profile, user_analysis, semantic_analysis)

        # Générer une entrée de journal quotidienne
        current_date = datetime.now().strftime("%Y-%m-%d")
        c.execute("SELECT id FROM journal_entries WHERE user_id = ? AND date = ?", (user_id, current_date))
        if not c.fetchone():
            journal_entry = generate_journal_entry(user_id)
            context.bot.send_message(chat_id=update.effective_chat.id, text="J'ai généré votre entrée de journal pour aujourd'hui. Vous pouvez la consulter en utilisant la commande /journal.")

    except Exception as e:
        logger.error(f"Error in handle_message: {e}")
        logger.error(traceback.format_exc())
        response = "Je suis désolé, j'ai rencontré une erreur en traitant votre message. Pourriez-vous reformuler ou essayer à nouveau plus tard ?"

    context.bot.send_message(chat_id=update.effective_chat.id, text=response)

    # Vérifier si une réponse proactive est nécessaire
    if should_send_proactive_message(user_id, user_analysis):
        proactive_response = generate_proactive_response(user_id, user_profile, user_analysis)
        context.bot.send_message(chat_id=update.effective_chat.id, text=proactive_response)


def should_send_proactive_message(user_id, user_analysis):
    try:
        # Récupérer le timestamp de la dernière interaction
        c.execute("SELECT MAX(timestamp) FROM memory WHERE user_id = ?", (user_id,))
        last_interaction = c.fetchone()[0]
        
        # Calculer le temps écoulé depuis la dernière interaction
        time_since_last_interaction = time.time() - ensure_float(last_interaction or 0)
        
        # Récupérer l'humeur actuelle de l'utilisateur
        current_mood = user_analysis.get('emotions', {}).get('emotion', 'neutral')
        
        # Récupérer le profil de l'utilisateur
        user_profile = get_user_profile(user_id)
        
        # Calculer le score de proactivité
        proactivity_score = 0
        
        # Facteur 1: Temps écoulé depuis la dernière interaction
        if time_since_last_interaction > 86400:  # Plus de 24 heures
            proactivity_score += 0.3
        elif time_since_last_interaction > 43200:  # Plus de 12 heures
            proactivity_score += 0.2
        elif time_since_last_interaction > 21600:  # Plus de 6 heures
            proactivity_score += 0.1
        
        # Facteur 2: Humeur de l'utilisateur
        if current_mood in ['sad', 'angry', 'frustrated']:
            proactivity_score += 0.2
        elif current_mood in ['happy', 'excited']:
            proactivity_score += 0.1
        
        # Facteur 3: Heure de la journée
        current_hour = datetime.now().hour
        if 9 <= current_hour <= 11 or 14 <= current_hour <= 16:  # Heures de pointe d'activité
            proactivity_score += 0.1
        
        # Facteur 4: Jour de la semaine
        if datetime.now().weekday() < 5:  # Jours de semaine
            proactivity_score += 0.1
        
        # Facteur 5: Préférence de l'utilisateur pour les messages proactifs
        user_proactivity_preference = ensure_float(user_profile.get('proactivity_preference', '0.5'))
        proactivity_score *= user_proactivity_preference
        
        # Facteur 6: Tâches en attente
        pending_tasks = get_pending_tasks(user_id)
        if pending_tasks:
            proactivity_score += 0.1
        
        # Facteur 7: Événements à venir
        upcoming_events = get_upcoming_events(user_id)
        if upcoming_events:
            proactivity_score += 0.1
        
        # Facteur 8: Objectifs non atteints
        unachieved_goals = get_unachieved_goals(user_id)
        if unachieved_goals:
            proactivity_score += 0.1
        
        # Décision finale
        threshold = 0.6  # Seuil de décision
        should_send = proactivity_score > threshold
        
        # Enregistrer la décision pour l'apprentissage futur
        log_proactivity_decision(user_id, proactivity_score, should_send)
        
        return should_send

    except Exception as e:
        logger.error(f"Erreur dans should_send_proactive_message: {str(e)}")
        logger.error(traceback.format_exc())
        return False  # En cas d'erreur, on ne recommande pas d'envoyer un message proactif

def ensure_float(value):
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    elif isinstance(value, (int, float)):
        return float(value)
    else:
        return 0.0


def get_upcoming_events(user_id):
    try:
        # Récupérer les événements à venir dans les 24 prochaines heures
        tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')
        c.execute("SELECT * FROM calendar_events WHERE user_id = ? AND start_time <= ? ORDER BY start_time", (user_id, tomorrow))
        return c.fetchall()
    except Exception as e:
        logger.error(f"Erreur dans get_upcoming_events: {str(e)}")
        return []


def get_unachieved_goals(user_id):
    try:
        # Récupérer les objectifs non atteints
        c.execute("SELECT * FROM personal_goals WHERE user_id = ? AND status != 'achieved'", (user_id,))
        return c.fetchall()
    except Exception as e:
        logger.error(f"Erreur dans get_unachieved_goals: {str(e)}")
        return []

def log_proactivity_decision(user_id, score, decision):
    try:
        # Enregistrer la décision pour l'apprentissage futur
        c.execute("INSERT INTO proactivity_logs (user_id, timestamp, score, decision) VALUES (?, ?, ?, ?)",
                  (user_id, time.time(), score, int(decision)))
        conn.commit()
    except Exception as e:
        logger.error(f"Erreur dans log_proactivity_decision: {str(e)}")



def generate_proactive_response(user_id, user_profile, user_analysis):
    prompt = f"""
    Based on the following user profile and analysis, generate a proactive message to engage the user:
    User Profile: {json.dumps(user_profile)}
    User Analysis: {json.dumps(user_analysis)}

    The message should be:
    1. Relevant to the user's interests or recent activities
    2. Encouraging or supportive
    3. Open-ended to invite further conversation

    Proactive Message:
    """
    return get_gpt4_response(prompt, max_tokens=100)

def update_user_profile_from_interaction(user_id, user_input, response, user_analysis, semantic_analysis):
    for trait, value in user_analysis['personality'].items():
        update_user_profile(user_id, f'personality_{trait}', str(value))
    
    for entity, entity_type in semantic_analysis['entities']:
        if entity_type in ['PERSON', 'ORG', 'PRODUCT', 'EVENT']:
            update_user_profile(user_id, f'interest_{entity.lower()}', 'mentioned')
    
    current_complexity = float(get_user_profile(user_id).get('language_complexity', '0'))
    new_complexity = (current_complexity + semantic_analysis['complexity_score']) / 2
    update_user_profile(user_id, 'language_complexity', str(new_complexity))
    
    update_conversation_topics(user_id, user_input)

def update_conversation_topics(user_id, text):
    c.execute("SELECT value FROM user_profile WHERE user_id = ? AND key = 'conversation_topics'", (user_id,))
    result = c.fetchone()
    if result:
        topics = json.loads(result[0])
    else:
        topics = {}
    
    doc = nlp(text)
    keywords = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct and token.pos_ in ['NOUN', 'VERB', 'ADJ']]
    
    for keyword in keywords:
        if keyword in topics:
            topics[keyword] += 1
        else:
            topics[keyword] = 1
    
    topics = dict(sorted(topics.items(), key=lambda x: x[1], reverse=True)[:20])
    
    update_user_profile(user_id, 'conversation_topics', json.dumps(topics))

# Modification de la fonction calculate_importance
def calculate_importance(user_input, response, user_analysis, semantic_analysis):
    importance_factors = {
        'sentiment_intensity': abs(ensure_float(user_analysis['polarity'])) * 2,
        'subjectivity': ensure_float(user_analysis['subjectivity']),
        'entity_relevance': len(semantic_analysis['entities']) / 10,
        'complexity_match': 1 - abs(semantic_analysis['complexity_score'] - ensure_float(user_analysis.get('language_complexity', '0'))) / 10,
        'keyword_relevance': len(set(semantic_analysis['keywords']).intersection(set(user_analysis.get('interests', [])))) / 10,
        'response_length': min(len(response) / 500, 1),
        'question_asked': 1 if '?' in user_input else 0,
        'emotional_content': max([ensure_float(v) for v in user_analysis.get('emotions', {}).values()]) if 'emotions' in user_analysis else 0
    }
    return sum(importance_factors.values()) / len(importance_factors)

def predict_user_behavior(user_id):
    user_profile = get_user_profile(user_id)
    recent_interactions = get_recent_interactions(user_id, limit=50)
    
    behavior_trends = analyze_behavior_trends(recent_interactions)
    predicted_interests = predict_future_interests(user_profile, recent_interactions)
    predicted_mood = predict_future_mood(recent_interactions)
    
    return {
        'behavior_trends': behavior_trends,
        'predicted_interests': predicted_interests,
        'predicted_mood': predicted_mood
    }

def analyze_behavior_trends(interactions):
    if not interactions:
        return {'time_trend': 0, 'length_trend': 0}
    
    interaction_times = [float(interaction['timestamp']) for interaction in interactions]
    interaction_lengths = [len(interaction['user_input']) for interaction in interactions]
    
    time_trend = np.polyfit(range(len(interaction_times)), interaction_times, 1)[0] if len(interaction_times) > 1 else 0
    length_trend = np.polyfit(range(len(interaction_lengths)), interaction_lengths, 1)[0] if len(interaction_lengths) > 1 else 0
    
    return {
        'time_trend': time_trend,
        'length_trend': length_trend
    }

def predict_future_interests(user_profile, recent_interactions):
    current_interests = set(user_profile.get('interests', []))
    recent_keywords = set()
    for interaction in recent_interactions:
        recent_keywords.update(extract_keywords(interaction['user_input']))
    
    new_interests = recent_keywords - current_interests
    return list(new_interests)[:5]

def predict_future_mood(recent_interactions):
    recent_sentiments = [interaction['sentiment'] for interaction in recent_interactions if 'sentiment' in interaction]
    if not recent_sentiments:
        return 'neutral'
    
    mood_trend = sum(recent_sentiments) / len(recent_sentiments)
    if mood_trend > 0.2:
        return 'positive'
    elif mood_trend < -0.2:
        return 'negative'
    else:
        return 'neutral'

def generate_rl_enhanced_response(user_input, user_profile, semantic_analysis):
    state = rl_learner.get_state(user_input, user_profile)
    action = rl_learner.choose_action(state)
    
    if action == 'empathize':
        response_type = "Generate an empathetic response that acknowledges the user's emotions."
    elif action == 'inform':
        response_type = "Provide informative content related to the user's query or interests."
    elif action == 'question':
        response_type = "Ask a thought-provoking question to deepen the conversation."
    else:  # suggest
        response_type = "Offer a suggestion or recommendation based on the user's interests or needs."
    
    prompt = f"""
    User Input: {user_input}
    User Profile: {json.dumps(user_profile)}
    Semantic Analysis: {json.dumps(semantic_analysis)}
    Response Type: {response_type}

    Generate a response according to the specified response type, taking into account the user's input, profile, and the semantic analysis.
    """
    
    response = get_gpt4_response(prompt, max_tokens=150)
    return response, state, action

def update_rl_model(state, action, user_feedback, next_user_input, next_user_profile):
    reward = 1 if user_feedback > 0 else -1
    next_state = rl_learner.get_state(next_user_input, next_user_profile)
    rl_learner.update(state, action, reward, next_state)

def integrate_external_knowledge(query):
    try:
        wiki_results = wikipedia.search(query)
        if wiki_results:
            page = wikipedia.page(wiki_results[0])
            summary = page.summary
            return f"According to Wikipedia: {summary[:250]}..."
    except:
        pass
    
    try:
        client = wolframalpha.Client(WOLFRAM_ALPHA_APP_ID)
        res = client.query(query)
        if res['@success'] == 'true':
            return next(res.results).text
    except:
        pass
    
    return None

def generate_text_with_t5(prompt, max_length=150):
    input_ids = t5_tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = t5_model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)
    return t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

def summarize_text(text, max_length=150):
    input_text = "summarize: " + text
    summary = generate_text_with_t5(input_text, max_length)
    return summary

def analyze_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    features = vgg_model.predict(img_array)
    flattened_features = features.flatten()
    
    return flattened_features

def speech_to_text(audio_file):
    r = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = r.record(source)
    try:
        text = r.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Sorry, I couldn't understand the audio."
    except sr.RequestError:
        return "Sorry, there was an error processing the audio."

def text_to_speech(text, output_file):
    tts = gTTS(text=text, lang='en')
    tts.save(output_file)
    return output_file

def handle_voice_message(update, context):
    file = context.bot.get_file(update.message.voice.file_id)
    voice_file = io.BytesIO()
    file.download(out=voice_file)
    voice_file.seek(0)
    
    text = speech_to_text(voice_file)
    if text.startswith("Sorry"):
        context.bot.send_message(chat_id=update.effective_chat.id, text=text)
        return
    
    response = handle_message(update, context, text)
    
    audio_file = text_to_speech(response, "response.mp3")
    context.bot.send_voice(chat_id=update.effective_chat.id, voice=open(audio_file, 'rb'))

def handle_image_message(update, context):
    file = context.bot.get_file(update.message.photo[-1].file_id)
    image_stream = io.BytesIO()
    file.download(out=image_stream)
    image_stream.seek(0)
    
    image = Image.open(image_stream)
    
    # Convertir l'image en tableau numpy
    image_array = np.array(image)
    
    # Redimensionner l'image si nécessaire
    if image_array.shape[:2] != (224, 224):
        image = image.resize((224, 224))
        image_array = np.array(image)
    
    # Ajouter une dimension pour le lot (batch)
    image_array = np.expand_dims(image_array, axis=0)
    
    # Prétraiter l'image pour VGG16
    image_array = preprocess_input(image_array)
    
    # Obtenir les caractéristiques de l'image
    image_features = vgg_model.predict(image_array)
    
    # Aplatir les caractéristiques
    flattened_features = image_features.flatten()
    
    prompt = f"Describe the content of an image with the following features: {flattened_features[:10]}..."
    description = generate_text_with_t5(prompt)
    
    context.bot.send_message(chat_id=update.effective_chat.id, text=f"Voici ce que je vois dans l'image : {description}")


def handle_feedback(update, context):
    user_id = update.effective_user.id
    feedback = update.message.text
    
    try:
        feedback_value = int(feedback)
        if 1 <= feedback_value <= 5:
            last_state = context.user_data.get('last_rl_state')
            last_action = context.user_data.get('last_rl_action')
            if last_state and last_action:
                user_profile = get_user_profile(user_id)
                update_rl_model(last_state, last_action, feedback_value, update.message.text, user_profile)
                context.bot.send_message(chat_id=update.effective_chat.id, text="Merci pour votre feedback ! J'utiliserai cette information pour m'améliorer.")
            else:
                context.bot.send_message(chat_id=update.effective_chat.id, text="Désolé, je n'ai pas pu associer votre feedback à une interaction récente.")
        else:
            context.bot.send_message(chat_id=update.effective_chat.id, text="Veuillez donner une note entre 1 et 5.")
    except ValueError:
        context.bot.send_message(chat_id=update.effective_chat.id, text="Désolé, je n'ai pas compris votre feedback. Veuillez donner une note entre 1 et 5.")

def generate_daily_insight(user_id):
    user_profile = get_user_profile(user_id)
    recent_interactions = memory_buffer.get_recent(20)
    personality = analyze_personality(user_id)

    prompt = f"""Based on the following user profile, recent interactions, and personality analysis, generate a personalized daily insight:
    User Profile: {json.dumps(user_profile)}
    Recent Interactions: {json.dumps(recent_interactions)}
    Personality Analysis: {json.dumps(personality)}

    The insight should:
    1. Offer a unique perspective or observation about the user's recent behavior or interests
    2. Provide a motivational message aligned with the user's goals or personality
    3. Suggest a small, actionable task that could improve the user's day
    4. Include a relevant quote or fact that might interest the user

    Daily Insight:"""

    insight = get_gpt4_response(prompt, max_tokens=200)
    return insight

def check_proactive_actions(update, context, user_id, user_profile, user_analysis, semantic_analysis):
    pending_tasks = get_pending_tasks(user_id)
    if pending_tasks:
        nearest_task = min(pending_tasks, key=lambda x: datetime.datetime.strptime(x[1], "%Y-%m-%d"))
        if datetime.datetime.strptime(nearest_task[1], "%Y-%m-%d") - datetime.datetime.now() <= datetime.timedelta(days=1):
            reminder = f"Don't forget, you have a task due soon: {nearest_task[0]}"
            context.bot.send_message(chat_id=update.effective_chat.id, text=reminder)
    
    if user_analysis['polarity'] < -0.5:
        activity_suggestion = suggest_mood_improving_activity(user_profile, user_analysis)
        context.bot.send_message(chat_id=update.effective_chat.id, text=activity_suggestion)
    
    if random.random() < 0.1:
        emotional_insights = generate_emotional_insights(user_id)
        context.bot.send_message(chat_id=update.effective_chat.id, text=emotional_insights)

    if random.random() < 0.05:
        daily_insight = generate_daily_insight(user_id)
        context.bot.send_message(chat_id=update.effective_chat.id, text=daily_insight)

def suggest_mood_improving_activity(user_profile, user_analysis):
    interests = [key.split('_')[1] for key in user_profile.keys() if key.startswith('interest_')]
    prompt = f"""
    The user seems to be in a negative mood. Based on their interests: {', '.join(interests)},
    and their current emotional state: {json.dumps(user_analysis.get('emotions', {}))},
    suggest a mood-improving activity. Make the suggestion empathetic and encouraging.
    """
    return get_gpt4_response(prompt, max_tokens=100)

def generate_emotional_insights(user_id):
    user_profile = get_user_profile(user_id)
    recent_interactions = memory_buffer.get_recent(20)
    emotional_history = [analyze_emotion(interaction[0]) for interaction in recent_interactions]
    
    prompt = f"""
    Based on the following user profile and emotional history, generate insights about the user's emotional patterns:
    
    User Profile: {json.dumps(user_profile)}
    Emotional History: {json.dumps(emotional_history)}
    
    Please provide:
    1. A summary of the user's dominant emotions over time
    2. Any noticeable emotional trends or patterns
    3. Suggestions for improving the user's emotional well-being
    4. Potential triggers for positive and negative emotions
    5. Recommendations for topics or activities that might boost the user's mood
    
    Insights:
    """
    
    insights = get_gpt4_response(prompt, max_tokens=250)
    return insights

def error_handler(update, context):
    """Gère les erreurs rencontrées par le bot."""
    logger.error(f"Une erreur s'est produite : {context.error}")
    
    # Envoie un message à l'utilisateur
    if update and update.effective_message:
        update.effective_message.reply_text("Désolé, une erreur s'est produite lors du traitement de votre message. Veuillez réessayer plus tard.")


def main():
    updater = Updater(token=TELEGRAM_TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))
    dp.add_handler(CommandHandler("feedback", handle_feedback))
    dp.add_handler(MessageHandler(Filters.voice, handle_voice_message))
    dp.add_handler(MessageHandler(Filters.photo, handle_image_message))
    dp.add_handler(CommandHandler("analyze", handle_analysis_command))
    dp.add_handler(CommandHandler("learn", handle_learn_command))
    dp.add_handler(CommandHandler("knowledge", knowledge_base_menu))
    dp.add_handler(CallbackQueryHandler(handle_query))
    dp.add_handler(CommandHandler("journal", view_journal))
    dp.add_handler(MessageHandler(Filters.document, handle_document_upload))
    
    dp.add_error_handler(error_handler)

    # Schedule jobs
    updater.job_queue.run_repeating(lambda _: continuous_learning(), interval=86400, first=0)
    updater.job_queue.run_repeating(initiate_conversation, interval=3600, first=0)
    updater.job_queue.run_repeating(lambda _: update_models_and_knowledge(), interval=604800, first=0)  # Weekly update
    
    # Nouvelle tâche planifiée pour générer et envoyer le PDF quotidien
    updater.job_queue.run_daily(send_daily_pdf, time=datetime_time(hour=23, minute=55))

    updater.start_polling()
    logger.info("Bot started successfully with advanced AI capabilities.")
    updater.idle()


def send_daily_pdf(context: CallbackContext):
    for user_id in get_active_users():
        try:
            date = datetime.datetime.now().strftime("%Y-%m-%d")
            pdf_buffer = generate_daily_pdf(user_id, date)
            context.bot.send_document(
                chat_id=user_id,
                document=pdf_buffer,
                filename=f"rapport_quotidien_{date}.pdf",
                caption="Voici votre rapport quotidien stylisé !"
            )
        except Exception as e:
            logger.error(f"Erreur lors de l'envoi du PDF quotidien à l'utilisateur {user_id}: {str(e)}")

def get_active_users():
    # Récupérer la liste des utilisateurs actifs depuis la base de données
    c.execute("SELECT DISTINCT user_id FROM journal_entries")
    return [row[0] for row in c.fetchall()]

def continuous_learning():
    logger.info("Starting continuous learning process")
    
    new_content, source = fetch_new_content()
    if new_content:
        update_knowledge_base(new_content, source)
        logger.info(f"Added new knowledge from {source}")
    else:
        logger.warning("No new content available for learning")
    
    train_models()
    update_embeddings()
    logger.info("Continuous learning process completed")



def fetch_new_content():
    """
    Fetch new content from a random Wikipedia page.
    Returns a tuple of (text, source) or (None, None) if fetching fails.
    """
    try:
        response = requests.get("https://en.wikipedia.org/wiki/Special:Random")
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.find(id="firstHeading").text
        content = soup.find(id="mw-content-text").find_all("p")
        text = "\n".join([p.text for p in content[:5]])  # Get first 5 paragraphs
        return text, f"Wikipedia - {title}"
    except Exception as e:
        logger.error(f"Error fetching new content: {e}")
        return None, None


def alter_knowledge_base_table(cursor):
    try:
        cursor.execute("ALTER TABLE knowledge_base ADD COLUMN category TEXT")
        conn.commit()
        logger.info("Added category column to knowledge_base table")
    except sqlite3.OperationalError:
        logger.info("Category column already exists in knowledge_base table")



def update_knowledge_base():
    logger.info("Updating knowledge base")
    try:
        response = requests.get("https://en.wikipedia.org/wiki/Special:Random")
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.find(id="firstHeading").text
        content = soup.find(id="mw-content-text").find_all("p")
        content_text = "\n".join([p.text for p in content[:5]])

        c.execute("INSERT INTO knowledge_base (timestamp, title, content, source) VALUES (?, ?, ?, ?)",
                  (time.time(), title, content_text, "Wikipedia"))
        conn.commit()
        logger.info(f"Added new knowledge: {title}")
    except Exception as e:
        logger.error(f"Error updating knowledge base: {e}")

def train_models():
    logger.info("Training models")
    try:
        c.execute("SELECT user_input, intent FROM memory")
        data = c.fetchall()
        if len(data) > 100:
            X = [row[0] for row in data]
            y = [row[1] for row in data]
            
            vectorizer = TfidfVectorizer(max_features=5000)
            X_vectorized = vectorizer.fit_transform(X)
            
            X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)
            
            clf = MultinomialNB()
            clf.fit(X_train, y_train)
            
            accuracy = clf.score(X_test, y_test)
            logger.info(f"Intent classifier accuracy: {accuracy}")
            
            joblib.dump(clf, 'intent_classifier.joblib')
            joblib.dump(vectorizer, 'intent_vectorizer.joblib')
            logger.info("Intent classifier trained and saved")
    except Exception as e:
        logger.error(f"Error training intent classifier: {e}")

def update_embeddings():
    logger.info("Updating embeddings")
    try:
        c.execute("SELECT text FROM long_term_memory")
        texts = c.fetchall()
        
        updated_embeddings = []
        for text in texts:
            embedding = get_embedding(text[0])
            updated_embeddings.append(embedding)
        
        c.executemany("UPDATE long_term_memory SET vector = ? WHERE text = ?",
                      zip([serialize_embedding(emb) for emb in updated_embeddings], [text[0] for text in texts]))
        conn.commit()
        logger.info(f"Updated embeddings for {len(texts)} memories")
    except Exception as e:
        logger.error(f"Error updating embeddings: {e}")

def initiate_conversation(context: CallbackContext):
    for user_id in conversation_manager.last_interaction.keys():
        if conversation_manager.should_initiate_conversation(user_id):
            user_profile = get_user_profile(user_id)
            behavior_prediction = predict_user_behavior(user_id)
            
            prompt = f"""Based on the following user profile and behavior prediction, generate a friendly message to initiate a conversation:

            User Profile: {json.dumps(user_profile)}
            Behavior Prediction: {json.dumps(behavior_prediction)}

            The message should be casual, personalized, and encourage further conversation."""

            message = get_gpt4_response(prompt, max_tokens=100)
            context.bot.send_message(chat_id=user_id, text=message)

def update_models_and_knowledge():
    logger.info("Starting weekly update of models and knowledge")
    new_content, source = fetch_new_content()
    if new_content:
        update_knowledge_base(new_content, source)
    train_models()
    update_embeddings()
    fine_tune_language_model()
    logger.info("Weekly update completed")

def fine_tune_language_model():
    logger.info("Fine-tuning language model")
    try:
        c.execute("SELECT bot_response FROM memory ORDER BY timestamp DESC LIMIT 1000")
        responses = c.fetchall()
        
        train_data = [response[0] for response in responses]
        
        if gpt2_tokenizer.pad_token is None:
            gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
            gpt2_model.config.pad_token_id = gpt2_tokenizer.eos_token_id

        train_encodings = gpt2_tokenizer(train_data, truncation=True, padding=True, max_length=512, return_tensors="pt")
        train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'])
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpt2_model.to(device)

        optimizer = AdamW(gpt2_model.parameters(), lr=5e-5)
        loss_fn = CrossEntropyLoss()

        gpt2_model.train()
        for epoch in range(3):  # 3 epochs
            for batch in train_loader:
                input_ids, attention_mask = [b.to(device) for b in batch]
                outputs = gpt2_model(input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        gpt2_model.save_pretrained("./gpt2_fine_tuned")
        gpt2_tokenizer.save_pretrained("./gpt2_fine_tuned")
        
        logger.info("Language model fine-tuned and saved")
    except Exception as e:
        logger.error(f"Error fine-tuning language model: {str(e)}")

def start(update, context):
    user_id = update.effective_user.id
    welcome_message = "Bonjour ! Je suis votre assistant IA personnel. Comment puis-je vous aider aujourd'hui ?"
    context.bot.send_message(chat_id=update.effective_chat.id, text=welcome_message)
    initialize_user_profile(user_id)

def initialize_user_profile(user_id):
    default_profile = {
        "name": "User",
        "language": "fr",
        "interests": json.dumps([]),
        "communication_style": "neutral",
        "language_complexity": "0"
    }
    for key, value in default_profile.items():
        update_user_profile(user_id, key, value)
    return default_profile

def handle_analysis_command(update, context):
    user_id = update.effective_user.id
    personality = analyze_personality(user_id)
    recent_conversations = [conv[1] for conv in conversation_manager.get_recent_conversation(user_id, n=50)]
    
    topics_result = improved_topic_modeling(recent_conversations)
    
    if "error" in topics_result:
        topic_analysis = f"Erreur lors de l'analyse des sujets : {topics_result['error']}"
    else:
        topic_analysis = json.dumps(topics_result, indent=2)

    analysis_text = f"""Voici une analyse de vos interactions récentes :

Profil de personnalité :
{json.dumps(personality, indent=2)}

Sujets d'intérêt récents :
{topic_analysis}

Cette analyse est basée sur vos conversations récentes et peut aider à personnaliser nos futures interactions."""

    context.bot.send_message(chat_id=update.effective_chat.id, text=analysis_text)

def handle_learn_command(update, context):
    user_input = update.message.text.replace("/learn", "").strip()
    if not user_input:
        context.bot.send_message(chat_id=update.effective_chat.id, text="Veuillez fournir des informations à apprendre. Usage : /learn <votre texte ici>")
        return

    update_knowledge_graph(user_input)
    context.bot.send_message(chat_id=update.effective_chat.id, text="Merci pour l'information ! J'ai mis à jour ma base de connaissances.")

def update_knowledge_graph(text):
    doc = nlp(text)
    for ent in doc.ents:
        knowledge_graph.add_entity(ent.text, {'type': ent.label_})
    
    for token in doc:
        if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
            subject = token.text
            verb = token.head.text
            for child in token.head.children:
                if child.dep_ == "dobj":
                    obj = child.text
                    knowledge_graph.add_relationship(subject, obj, verb)

def improved_topic_modeling(texts, num_topics=5):
    if not texts:
        return {"error": "No texts provided"}

    # Assurez-vous que les textes ne sont pas vides et contiennent des mots non-stop
    non_empty_texts = [text for text in texts if text.strip()]
    if not non_empty_texts:
        return {"error": "All texts are empty or contain only stop words"}

    try:
        vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        doc_term_matrix = vectorizer.fit_transform(non_empty_texts)
        
        if doc_term_matrix.shape[1] == 0:
            return {"error": "No features were extracted. Check if texts contain enough unique words."}

        lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
        lda_output = lda_model.fit_transform(doc_term_matrix)
        
        feature_names = vectorizer.get_feature_names_out()
        topics = {}
        for topic_idx, topic in enumerate(lda_model.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
            topics[f"Topic {topic_idx + 1}"] = top_words
        
        return topics
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

def extract_keywords(text):
    doc = nlp(text)
    keywords = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.pos_ in ['NOUN', 'VERB', 'ADJ']]
    return keywords

def get_recent_interactions(user_id, limit=50):
    c.execute("SELECT * FROM memory WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?", (user_id, limit))
    columns = [description[0] for description in c.description]
    return [dict(zip(columns, row)) for row in c.fetchall()]

def classify_response(response):
    if '?' in response:
        return "question"
    elif any(word in response.lower() for word in ["désolé", "pardon", "excuse"]):
        return "apology"
    elif any(word in response.lower() for word in ["suggestion", "recommandation", "conseil"]):
        return "suggestion"
    else:
        return "statement"

def train_intelligent_model(user_input, response, user_analysis, semantic_analysis):
    features = extract_features(user_input, user_analysis, semantic_analysis)
    label = classify_response(response)
    
    global intelligent_model
    if intelligent_model is None:
        intelligent_model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    intelligent_model.fit(np.array([features]), np.array([label]))
    logger.info("Intelligent model updated")

def extract_features(user_input, user_analysis, semantic_analysis):
    features = []
    
    features.append(len(user_input))
    features.append(len(user_input.split()))
    features.append(int('?' in user_input))
    features.append(int('!' in user_input))
    
    features.append(user_analysis['polarity'])
    features.append(user_analysis['subjectivity'])
    
    for trait, value in user_analysis['personality'].items():
        features.append(value)
    
    features.append(len(semantic_analysis['entities']))
    features.append(semantic_analysis['complexity_score'])
    features.append(len(semantic_analysis['keywords']))
    
    return features

def preprocess_input(x):
    return tf.keras.applications.vgg16.preprocess_input(x)

# Initialisation du modèle intelligent
intelligent_model = None

# Fonctions pour la gestion de la base de connaissances
def process_document(file_path):
    file_extension = file_path.split('.')[-1].lower()
    if file_extension == 'pdf':
        return process_pdf(file_path)
    elif file_extension == 'docx':
        return process_docx(file_path)
    elif file_extension == 'txt':
        return process_txt(file_path)
    elif file_extension in ['csv', 'json']:
        return process_structured_data(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

def process_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def process_docx(file_path):
    doc = Document(file_path)
    return " ".join([paragraph.text for paragraph in doc.paragraphs])

def process_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def process_structured_data(file_path):
    file_extension = file_path.split('.')[-1].lower()
    if file_extension == 'csv':
        with open(file_path, 'r') as file:
            reader = csv.DictReader(file)
            return json.dumps([row for row in reader])
    elif file_extension == 'json':
        with open(file_path, 'r') as file:
            return json.dumps(json.load(file))

def extract_key_information(text):
    sentences = sent_tokenize(text)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    key_sentences = [sentences[i] for i in similarities.argsort()[0][-5:]]
    return " ".join(key_sentences)


def update_knowledge_base(text, source):
    key_info = extract_key_information(text)
    category = categorize_knowledge(key_info)
    tags = extract_tags(key_info)
    
    c.execute("""
    INSERT INTO knowledge_base (timestamp, title, content, source, category, tags, relevance_score, rating_count)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (time.time(), f"Learned from {source}", key_info, source, category, ','.join(tags), 0, 0))
    conn.commit()


def handle_document_upload(update, context):
    if not update.message.document:
        context.bot.send_message(chat_id=update.effective_chat.id, text="Veuillez envoyer un document à apprendre.")
        return

    file = context.bot.get_file(update.message.document.file_id)
    file_name = update.message.document.file_name
    file_path = f"temp_{file_name}"
    file.download(file_path)

    try:
        document_text = process_document(file_path)
        update_knowledge_base(document_text, file_name)
        context.bot.send_message(chat_id=update.effective_chat.id, text=f"J'ai appris de nouvelles informations à partir de {file_name}. Vous pouvez maintenant me poser des questions sur ce contenu!")
    except Exception as e:
        context.bot.send_message(chat_id=update.effective_chat.id, text=f"Désolé, une erreur s'est produite lors du traitement du document : {str(e)}")
    finally:
        os.remove(file_path)

def get_recent_learnings(limit=5):
    c.execute("SELECT id, title, content, source FROM knowledge_base ORDER BY timestamp DESC LIMIT ?", (limit,))
    return c.fetchall()

def search_knowledge_base(query):
    c.execute("SELECT id, title, content, source FROM knowledge_base")
    results = c.fetchall()
    
    documents = [f"{title}\n{content}" for _, title, content, _ in results]
    model = train_word2vec_model(documents)
    
    semantic_results = semantic_search(query, documents, model)
    return [(id, title, content, source, similarity) 
            for (id, title, content, source), (_, similarity) in zip(results, semantic_results)]

def categorize_knowledge(content):
    prompt = f"Catégorisez le texte suivant en une seule catégorie générale : \n\n{content}\n\nCatégorie :"
    category = get_gpt4_response(prompt, max_tokens=20).strip()
    return category

def extract_tags(content):
    prompt = f"Extrayez 3 à 5 tags pertinents du texte suivant, séparés par des virgules :\n\n{content}\n\nTags :"
    tags = get_gpt4_response(prompt, max_tokens=50).strip()
    return [tag.strip() for tag in tags.split(',')]

def add_relevance_rating(knowledge_id, rating):
    c.execute("UPDATE knowledge_base SET relevance_score = relevance_score + ?, rating_count = rating_count + 1 WHERE id = ?",
              (rating, knowledge_id))
    conn.commit()

def handle_relevance_rating(update, context):
    try:
        _, knowledge_id, rating = update.message.text.split()
        knowledge_id = int(knowledge_id)
        rating = float(rating)
        
        if not (1 <= rating <= 5):
            raise ValueError("La note doit être comprise entre 1 et 5.")
        
        add_relevance_rating(knowledge_id, rating)
        context.bot.send_message(chat_id=update.effective_chat.id, 
                                 text="Merci pour votre évaluation ! Elle aidera à améliorer la qualité des réponses.")
    except ValueError as e:
        context.bot.send_message(chat_id=update.effective_chat.id, 
                                 text=f"Erreur : {str(e)}. Utilisez le format '/rate [id] [note de 1 à 5]'.")

# Fonctions pour l'interface utilisateur de la base de connaissances
def knowledge_base_menu(update, context):
    keyboard = [
        [InlineKeyboardButton("Catégories", callback_data='categories')],
        [InlineKeyboardButton("Recherche", callback_data='search')],
        [InlineKeyboardButton("Apprentissages récents", callback_data='recent')],
        [InlineKeyboardButton("Top rated", callback_data='top_rated')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    update.message.reply_text('Naviguer dans la base de connaissances:', reply_markup=reply_markup)

def show_categories(update, context, page=1):
    query = update.callback_query
    query.answer()
    
    c.execute("SELECT DISTINCT category FROM knowledge_base ORDER BY category")
    all_categories = [cat[0] for cat in c.fetchall()]
    
    categories, total_pages = paginate_results(all_categories, page)
    
    keyboard = [[InlineKeyboardButton(cat, callback_data=f'cat_{cat}')] for cat in categories]
    keyboard.extend(create_pagination_keyboard(page, total_pages, 'categories'))
    keyboard.append([InlineKeyboardButton("Retour", callback_data='back_main')])
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    query.edit_message_text(text=f"Choisissez une catégorie (Page {page}/{total_pages}):", reply_markup=reply_markup)

def show_category_items(update, context, category, page=1, sort_by='relevance'):
    query = update.callback_query
    query.answer()
    
    c.execute("SELECT id, title, timestamp, relevance_score, rating_count FROM knowledge_base WHERE category = ?", (category,))
    all_items = c.fetchall()
    sorted_items = sort_items(all_items, sort_by)
    
    items, total_pages = paginate_results(sorted_items, page)
    
    keyboard = [[InlineKeyboardButton(item[1], callback_data=f'item_{item[0]}')] for item in items]
    keyboard.extend(create_pagination_keyboard(page, total_pages, f'cat_{category}_{sort_by}'))
    keyboard.append([
        InlineKeyboardButton("Trier par date", callback_data=f'sort_{category}_date_1'),
        InlineKeyboardButton("Trier par pertinence", callback_data=f'sort_{category}_relevance_1')
    ])
    keyboard.append([InlineKeyboardButton("Retour aux catégories", callback_data='categories')])
    keyboard.append([InlineKeyboardButton("Menu principal", callback_data='back_main')])
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    query.edit_message_text(text=f"Éléments de la catégorie '{category}' (Page {page}/{total_pages}):", reply_markup=reply_markup)

def show_item_details(update, context, item_id):
    query = update.callback_query
    query.answer()
    
    c.execute("SELECT title, content, source, category, relevance_score, rating_count, tags FROM knowledge_base WHERE id = ?", (item_id,))
    item = c.fetchone()
    
    if item:
        title, content, source, category, relevance_score, rating_count, tags = item
        avg_score = relevance_score / rating_count if rating_count > 0 else "Non évalué"
        
        details = f"📚 {title}\n\n"
        details += f"Catégorie : {category}\n"
        details += f"Source : {source}\n"
        details += f"Contenu : {content[:200]}...\n"
        details += f"Score moyen : {avg_score}\n"
        details += f"Tags : {tags}\n"
        details += f"Pour évaluer, utilisez '/rate {item_id} [note de 1 à 5]'\n"
        
        keyboard = [
            [InlineKeyboardButton("Retour à la catégorie", callback_data=f'cat_{category}')],
            [InlineKeyboardButton("Menu principal", callback_data='back_main')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        query.edit_message_text(text=details, reply_markup=reply_markup)
    else:
        query.edit_message_text(text="Élément non trouvé.")

def handle_search(update, context):
    query = update.callback_query
    query.answer()
    
    keyboard = [
        [InlineKeyboardButton("Recherche par mot-clé", callback_data='search_keyword')],
        [InlineKeyboardButton("Recherche par tag", callback_data='search_tag')],
        [InlineKeyboardButton("Menu principal", callback_data='back_main')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    query.edit_message_text(text="Choisissez le type de recherche :", reply_markup=reply_markup)

def show_search_results(update, context, search_term, search_type='keyword', page=1, sort_by='relevance'):
    if search_type == 'keyword':
        results = search_knowledge_base(search_term)
    else:  # search_type == 'tag'
        c.execute("SELECT id, title, timestamp, relevance_score, rating_count FROM knowledge_base WHERE tags LIKE ?", (f'%{search_term}%',))
        results = c.fetchall()
    
    sorted_results = sort_items(results, sort_by)
    paginated_results, total_pages = paginate_results(sorted_results, page)
    
    if paginated_results:
        keyboard = [[InlineKeyboardButton(item[1], callback_data=f'item_{item[0]}')] for item in paginated_results]
        keyboard.extend(create_pagination_keyboard(page, total_pages, f'search_{search_type}_{search_term}_{sort_by}'))
        keyboard.append([
            InlineKeyboardButton("Trier par date", callback_data=f'sort_search_{search_type}_{search_term}_date_1'),
            InlineKeyboardButton("Trier par pertinence", callback_data=f'sort_search_{search_type}_{search_term}_relevance_1')
        ])
        keyboard.append([InlineKeyboardButton("Menu principal", callback_data='back_main')])
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        update.message.reply_text(f"Résultats pour '{search_term}' (Page {page}/{total_pages}):", reply_markup=reply_markup)
    else:
        update.message.reply_text(f"Aucun résultat trouvé pour '{search_term}'.")

def handle_query(update, context):
    query = update.callback_query
    query.answer()
    
    if query.data == 'categories':
        show_categories(update, context)
    elif query.data.startswith('cat_'):
        parts = query.data.split('_')
        category = parts[1]
        page = int(parts[2]) if len(parts) > 2 else 1
        sort_by = parts[3] if len(parts) > 3 else 'relevance'
        show_category_items(update, context, category, page, sort_by)
    elif query.data.startswith('item_'):
        item_id = int(query.data[5:])
        show_item_details(update, context, item_id)
    elif query.data.startswith('search_'):
        handle_search(update, context)
    elif query.data.startswith('sort_'):
        parts = query.data.split('_')
        if parts[1] == 'search':
            search_type, search_term, sort_by, page = parts[2], parts[3], parts[4], int(parts[5])
            show_search_results(update, context, search_term, search_type, page, sort_by)
        else:
            category, sort_by, page = parts[1], parts[2], int(parts[3])
            show_category_items(update, context, category, page, sort_by)
    elif query.data == 'recent':
        show_recent_learnings(update, context)
    elif query.data == 'top_rated':
        show_top_rated(update, context)
    elif query.data == 'back_main':
        knowledge_base_menu(update, context)

def paginate_results(items, page=1):
    start = (page - 1) * ITEMS_PER_PAGE
    end = start + ITEMS_PER_PAGE
    return items[start:end], math.ceil(len(items) / ITEMS_PER_PAGE)

def create_pagination_keyboard(current_page, total_pages, base_callback):
    keyboard = []
    if current_page > 1:
        keyboard.append(InlineKeyboardButton("◀️ Précédent", callback_data=f'{base_callback}_page_{current_page-1}'))
    if current_page < total_pages:
        keyboard.append(InlineKeyboardButton("Suivant ▶️", callback_data=f'{base_callback}_page_{current_page+1}'))
    return keyboard

def sort_items(items, sort_by='relevance'):
    if sort_by == 'date':
        return sorted(items, key=lambda x: x[2], reverse=True)  # x[2] est le timestamp
    elif sort_by == 'relevance':
        return sorted(items, key=lambda x: x[3] / x[4] if x[4] > 0 else 0, reverse=True)  # x[3] est relevance_score, x[4] est rating_count
    else:  # Par défaut, tri par pertinence
        return items

def show_recent_learnings(update, context):
    query = update.callback_query
    query.answer()
    
    c.execute("SELECT id, title FROM knowledge_base ORDER BY timestamp DESC LIMIT 5")
    recent_items = c.fetchall()
    
    keyboard = [[InlineKeyboardButton(item[1], callback_data=f'item_{item[0]}')] for item in recent_items]
    keyboard.append([InlineKeyboardButton("Menu principal", callback_data='back_main')])
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    query.edit_message_text(text="Apprentissages récents:", reply_markup=reply_markup)

def show_top_rated(update, context):
    query = update.callback_query
    query.answer()
    
    c.execute("SELECT id, title FROM knowledge_base WHERE rating_count > 0 ORDER BY (relevance_score / rating_count) DESC LIMIT 5")
    top_items = c.fetchall()
    
    keyboard = [[InlineKeyboardButton(item[1], callback_data=f'item_{item[0]}')] for item in top_items]
    keyboard.append([InlineKeyboardButton("Menu principal", callback_data='back_main')])
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    query.edit_message_text(text="Éléments les mieux notés:", reply_markup=reply_markup)

# Fonctions pour le traitement du langage naturel avancé
def preprocess_text(text):
    return simple_preprocess(text, deacc=True)

def train_word2vec_model(documents):
    preprocessed_docs = [preprocess_text(doc) for doc in documents]
    model = Word2Vec(preprocessed_docs, vector_size=100, window=5, min_count=1, workers=4)
    return model

def get_document_embedding(model, document):
    words = preprocess_text(document)
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if not word_vectors:
        return np.zeros(model.vector_size)
    return np.mean(word_vectors, axis=0)

def semantic_search(query, documents, model):
    query_embedding = get_document_embedding(model, query)
    document_embeddings = [get_document_embedding(model, doc) for doc in documents]
    
    similarities = cosine_similarity([query_embedding], document_embeddings)[0]
    sorted_indexes = np.argsort(similarities)[::-1]
    
    return [(documents[i], similarities[i]) for i in sorted_indexes]


# Nouvelle fonction pour générer une entrée de journal
def generate_journal_entry(user_id):
    # Récupérer les données pertinentes
    user_profile = get_user_profile(user_id)
    recent_interactions = get_recent_interactions(user_id, limit=10)
    daily_tasks = get_pending_tasks(user_id)
    
    # Analyser l'humeur générale
    mood = analyze_overall_mood(recent_interactions)
    
    # Extraire les activités principales
    activities = extract_main_activities(recent_interactions)
    
    # Identifier les objectifs en cours
    goals = get_user_goals(user_id)
    
    # Générer le contenu du journal
    prompt = f"""
    Based on the following information, generate a detailed journal entry for the user:
    
    User Profile: {json.dumps(user_profile)}
    Recent Interactions: {json.dumps(recent_interactions)}
    Daily Tasks: {json.dumps(daily_tasks)}
    Overall Mood: {mood}
    Main Activities: {activities}
    Current Goals: {goals}
    
    The journal entry should include:
    1. A summary of the day's events and interactions
    2. Reflections on the user's mood and emotional state
    3. Progress towards goals and daily tasks
    4. Insights or lessons learned
    5. Plans or intentions for the near future
    
    Write the entry in a personal, reflective tone as if the user wrote it themselves.
    """
    
    journal_content = get_gpt4_response(prompt, max_tokens=500)
    
    # Enregistrer l'entrée dans la base de données
    c.execute("""
    INSERT INTO journal_entries (user_id, date, content, mood, activities, goals)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (user_id, datetime.datetime.now().strftime("%Y-%m-%d"), journal_content, mood, json.dumps(activities), json.dumps(goals)))
    conn.commit()
    
    return journal_content


# Modification de la fonction analyze_overall_mood
def analyze_overall_mood(interactions):
    try:
        sentiments = [analyze_emotion(interaction['user_input'])['sentiment']['compound'] for interaction in interactions]
        average_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        
        if average_sentiment > 0.3:
            return "positive"
        elif average_sentiment < -0.3:
            return "negative"
        else:
            return "neutral"
    except Exception as e:
        logger.error(f"Erreur dans analyze_overall_mood: {str(e)}")
        return "neutral"  # Valeur par défaut en cas d'erreur

# Fonction pour extraire les activités principales
def extract_main_activities(interactions):
    activities = []
    for interaction in interactions:
        doc = nlp(interaction['user_input'])
        for ent in doc.ents:
            if ent.label_ in ["EVENT", "FAC", "GPE", "LOC", "ORG"]:
                activities.append(ent.text)
    return list(set(activities))  # Éliminer les doublons

# Fonction pour récupérer les objectifs de l'utilisateur
def get_user_goals(user_id):
    c.execute("SELECT goal_description FROM personal_goals WHERE user_id = ? AND status = 'in_progress'", (user_id,))
    return [row[0] for row in c.fetchall()]

# Fonction pour récupérer les entrées de journal récentes
def get_recent_journal_entries(user_id, limit=7):
    c.execute("SELECT date, content FROM journal_entries WHERE user_id = ? ORDER BY date DESC LIMIT ?", (user_id, limit))
    return c.fetchall()


# Nouvelle commande pour consulter le journal
def view_journal(update, context):
    user_id = update.effective_user.id
    entries = get_recent_journal_entries(user_id)
    
    if not entries:
        context.bot.send_message(chat_id=update.effective_chat.id, text="Vous n'avez pas encore d'entrées de journal.")
        return
    
    for date, content in entries:
        message = f"📅 {date}\n\n{content[:1000]}..."  # Limiter la longueur pour éviter les messages trop longs
        context.bot.send_message(chat_id=update.effective_chat.id, text=message)



def generate_daily_pdf(user_id, date):
    # Récupérer les données nécessaires
    journal_entry = get_journal_entry(user_id, date)
    emotion_stats = get_emotion_stats(user_id, date)
    daily_summary = get_daily_summary(user_id, date)
    
    # Créer le document PDF
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    story = []
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = styles['Heading1']
    subtitle_style = styles['Heading2']
    normal_style = styles['Normal']
    
    # Titre
    story.append(Paragraph(f"Rapport quotidien - {date}", title_style))
    story.append(Spacer(1, 12))
    
    # Journal
    story.append(Paragraph("Journal du jour", subtitle_style))
    story.append(Paragraph(journal_entry, normal_style))
    story.append(Spacer(1, 12))
    
    # Statistiques émotionnelles
    story.append(Paragraph("Statistiques émotionnelles", subtitle_style))
    
    # Créer un graphique des émotions
    plt.figure(figsize=(6, 4))
    plt.bar(emotion_stats.keys(), emotion_stats.values())
    plt.title("Émotions du jour")
    plt.ylabel("Intensité")
    
    # Sauvegarder le graphique dans un buffer
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    
    # Ajouter le graphique au PDF
    img = Image(img_buffer)
    img.drawHeight = 3*inch
    img.drawWidth = 4*inch
    story.append(img)
    
    # Résumé de la journée
    story.append(Paragraph("Résumé de la journée", subtitle_style))
    story.append(Paragraph(daily_summary, normal_style))
    
    # Construire le PDF
    doc.build(story)
    buffer.seek(0)
    return buffer


def get_journal_entry(user_id, date):
    conn = sqlite3.connect('ai_companion.db')
    c = conn.cursor()
    c.execute("SELECT content FROM journal_entries WHERE user_id = ? AND date = ?", (user_id, date))
    result = c.fetchone()
    conn.close()
    
    if result:
        return result[0]
    else:
        return "Pas d'entrée de journal pour aujourd'hui."

def get_emotion_stats(user_id, date):
    conn = sqlite3.connect('ai_companion.db')
    c = conn.cursor()
    
    # Récupérer toutes les interactions de la journée
    start_of_day = datetime.strptime(date, "%Y-%m-%d")
    end_of_day = start_of_day + timedelta(days=1)
    c.execute("""
        SELECT emotion FROM memory 
        WHERE user_id = ? AND timestamp >= ? AND timestamp < ?
    """, (user_id, start_of_day.timestamp(), end_of_day.timestamp()))
    
    emotions = [eval(row[0])['emotion'] for row in c.fetchall()]
    conn.close()
    
    # Compter les occurrences de chaque émotion
    emotion_counts = Counter(emotions)
    
    # Normaliser les comptes pour obtenir des pourcentages
    total = sum(emotion_counts.values())
    emotion_stats = {emotion: count/total for emotion, count in emotion_counts.items()}
    
    return emotion_stats

def get_daily_summary(user_id, date):
    conn = sqlite3.connect('ai_companion.db')
    c = conn.cursor()
    
    # Récupérer les interactions de la journée
    start_of_day = datetime.strptime(date, "%Y-%m-%d")
    end_of_day = start_of_day + timedelta(days=1)
    c.execute("""
        SELECT user_input, bot_response FROM memory 
        WHERE user_id = ? AND timestamp >= ? AND timestamp < ?
    """, (user_id, start_of_day.timestamp(), end_of_day.timestamp()))
    
    interactions = c.fetchall()
    conn.close()
    
    # Générer un résumé avec GPT-4
    interaction_text = "\n".join([f"User: {input}\nAI: {response}" for input, response in interactions])
    prompt = f"""
    Résumez la journée de l'utilisateur en vous basant sur les interactions suivantes :
    
    {interaction_text}
    
    Fournissez un résumé concis qui met en évidence :
    1. Les principaux sujets de conversation
    2. Les activités ou tâches mentionnées
    3. L'humeur générale de l'utilisateur
    4. Toute décision ou réalisation importante
    
    Résumé :
    """
    
    summary = get_gpt4_response(prompt, max_tokens=200)
    return summary


# Fonction principale

if __name__ == '__main__':
    conn = sqlite3.connect('ai_companion.db', check_same_thread=False)
    c = conn.cursor()
    
    setup_database(c)
    alter_knowledge_base_table(c)
    initialize_gpt2()
    vgg_model = VGG16(weights='imagenet', include_top=False)
    gpt2_tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
    gpt2_model = transformers.GPT2LMHeadModel.from_pretrained("gpt2")
    
    main()
