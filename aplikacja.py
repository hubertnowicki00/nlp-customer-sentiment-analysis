# importy
import sys
import os
import time
import requests
import re
import json
import pandas as pd
import numpy as np
import joblib
import traceback
from datetime import datetime, timedelta
from flair.models import TextClassifier
from flair.data import Sentence
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm.auto import tqdm
from collections import Counter
import pantab
from tableauhyperapi import TableName

# Hyper API
from tableauhyperapi import (
    HyperProcess, Connection, Telemetry, CreateMode,
    TableDefinition, Inserter, Name, SqlType
)

APP_ID = 1086940  # id dla gry - Baldur's Gate 3 - możliwa zmiana dla innej gry, ale to już na kiedy indziej
MAX_REVIEWS_TO_FETCH = 1000  #ilość pobieranych danych
MODELS_DIR = "models"

KEYWORD_CATEGORIES = {
    'grafika': ['graf', 'visual', 'graph', 'art', 'design', 'textures', 'lighting', 'shaders', 'models', 'animation', 'animations', 'resolution', '4k', 'hd', 'detail', 'detailed', 'details', 'aesthetic', 'aesthetics', 'style', 'visuals', 'render', 'rendering'],
    'soundtrack': ['sound', 'muzyk', 'music', 'audio', 'sfx', 'soundtrack', 'voice acting', 'va', 'voices', 'vo', 'sound design', 'ost', 'official soundtrack', 'ambient', 'sound quality', 'volume', 'sound effects'],
    'fabuła': ['fab', 'story', 'plot', 'narrat', 'questline', 'quests', 'main story', 'side story', 'side quests', 'lore', 'dialogue choices', 'scenario', 'writing', 'plot twist', 'plot twists'],
    'mechanika': ['mechan', 'gameplay', 'multiplay', 'singleplay', 'coop', 'co-op', 'combat', 'controls', 'crafting', 'skills', 'builds', 'exploration', 'movement', 'jumping'],
    'postacie': ['posta', 'charact', 'bohater', 'hero', 'cosplay', 'npc', 'villains', 'protagonist', 'antagonist', 'character desing', 'personality'],
    'dialogi': ['dialog', 'rozmow', 'funny', 'humor', 'comedy', 'humour', 'banter', 'conversation', 'jokes', 'sarcasm'],
    'świat': ['świat', 'world', 'setting', 'immers', 'environ', 'open world', 'map', 'locations', 'areas', 'biomes', 'exploration', 'atmosphere'],
    'klimat': ['klimat', 'atmos', 'vibe', 'mood', 'creepy', 'cozy', 'intense', 'tense', 'feeling'],
    'bugi': ['bug', 'glitch', 'crash', 'error', 'problem', 'issue', 'lag', 'freeze', 'unstable', 'not working', 'corrupted', 'bugs', 'buggy'],
    'optymalizacja': ['optymal', 'perform', 'fps', 'performance', 'loading time', 'cpu', 'gpu', 'ram', 'optimization'],
    'sterowanie': ['sterow', 'control', 'keyb', 'pad', 'mouse', 'controls', 'keybinds', 'controller', 'keyboard', 'shortcut', 'shortcuts', 'sensitivity'],
    'balans': ['balan', 'diff', 'eas', 'hard', 'challeng', 'easy', 'difficult', 'scaling', 'overpowered', 'op', 'nerf', 'challenge'],
    'interfejs': ['interfejs', 'interface', 'ui', 'hud', 'menu', 'layout', 'navigation'],
    'cena': ['cen', 'price', 'koszt', 'cost', 'worth it', 'worth', 'expensive', 'cheap', 'overpriced', 'deal', 'discount', 'sale']
}

EXPECTATION_CATEGORIES = {
    "DLC": [
        "dlc", "expansion", "expansion pack", "more content", "new content", 
        "addon", "add-on", "bonus content", "season pass"],
    "Sequel": [
        "sequel", "next game", "next one", "bg4", "baldur's gate 4", 
        "follow-up", "continuation", "trilogy", "future game"],
    "Patchowanie": [
        "fix", "patch", "update", "needs fixing", "needs update", "glitch", 
        "bug fix", "crash fix", "optimization", "optimize", "repair", "broken"],
    "Co-op": [
        "multiplayer", "coop", "co-op", "online", "pvp", "pve", 
        "crossplay", "play with friends"],
    "Mechanika": [
        "new game plus", "ng+", "transmog", "customization", "photo mode", 
        "difficulty", "hard mode", "easy mode", "accessibility", "feature"],
    "Nadzieje": [
        "hope", "wish", "would like", "i'd like", "i want", "please add", 
        "looking forward", "can't wait", "excited for", "plan to", "waiting for"]
}

GENERAL_ASPECT = "general::other"

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS  # PyInstaller - dla zbudowania aplikacji .exe
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def auto_assign_category(word, categories_dict):
    word_lower = word.lower()
    for category, fragments in categories_dict.items():
        for frag in fragments:
            frag_lower = frag.lower()
            if len(frag_lower) < 4:
                continue
            if frag_lower in word_lower:
                return category
    return None

def run_dictionary_expansion(text_series, target_dict):
    print(f"Analiza top 400 słów i poszerzanie słowników")

    all_words = Counter()
    for text in text_series:
        all_words.update(str(text).split())
    most_common_words = all_words.most_common(400)

    new_keywords_map = {}
    for word, count in most_common_words:
        cat = auto_assign_category(word, target_dict)
        if cat:
            if word not in target_dict[cat]:
                new_keywords_map.setdefault(cat, []).append(word)

    count_added = 0
    for cat, words in new_keywords_map.items():
        target_dict[cat].extend(words)
        count_added += len(words)

    print(f"Dodano {count_added} nowych słów")
    return target_dict

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def assign_categories(text, dictionary):
    found = set()
    text = str(text).lower()
    for cat, keywords in dictionary.items():
        for kw in keywords:
            if kw in text:
                found.add(cat.capitalize())
    if not found:
        return ["Inne"]
    return list(found)

# pobieramy opinie ze Steama
def fetch_reviews():
    print(f"Pobieranie recenzji")
    url = f"https://store.steampowered.com/appreviews/{APP_ID}"
    params = {"json": 1, "filter": "recent", "language": "english", "review_type": "all", "purchase_type": "all", "num_per_page": 100, "cursor": "*"}

    reviews = []
    count = 0
    cursor = "*"

    consecutive_errors = 0
    max_retries = 5

    while count < MAX_REVIEWS_TO_FETCH:
        try:
            params["cursor"] = cursor
            time.sleep(1.5)
            resp = requests.get(url, params=params, timeout=25)
            if resp.status_code != 200:
                consecutive_errors += 1
                print(f"\nBłąd HTTP {resp.status_code}. Ponawianie próby")
                time.sleep(4)
                if consecutive_errors > max_retries:
                    print("\nPrzekroczono limit błędów. Koniec pobierania")
                    break
                continue
            consecutive_errors = 0
            data = resp.json()

            if not data or 'reviews' not in data:
                print(f"\n Błąd pliku JSON")
                break
            new_cursor = data.get('cursor')
            new_reviews = data.get('reviews', [])
            if not new_reviews:
                if new_cursor and new_cursor != cursor:
                    cursor = new_cursor
                    continue
                else:
                    print("\nBrak nowych recenzji do pobrania")
                    break

            for r in new_reviews:
                if not any(x['review_id'] == r.get("recommendationid") for x in reviews):    
                    reviews.append({
                        "review_id": r.get("recommendationid"),
                        "review": r.get("review"),
                        "voted_up": 1 if r.get("voted_up") else 0,
                        "date_str": datetime.fromtimestamp(r.get("timestamp_created")).strftime('%Y-%m-%d')
                    })

            count = len(reviews)
            cursor = new_cursor
            print(f"\rPobrano: {count}/{MAX_REVIEWS_TO_FETCH}", end="", flush=True)

        except Exception as e:
            consecutive_errors += 1
            print(f"\nBłąd pobierania: {e}")
            time.sleep(3)
            if consecutive_errors > max_retries:
                print("\nPrzekroczono limit błędów - koniec pobierania")
            break

    print(f"\nZebrano {len(reviews)} recenzji")
    if len(reviews) > MAX_REVIEWS_TO_FETCH:
        reviews = reviews[:MAX_REVIEWS_TO_FETCH]
    return pd.DataFrame(reviews)

# filtrujemy spam
def filter_spam(df):
    print("Filtrowanie spamu")

    vec_path = resource_path(os.path.join(MODELS_DIR, 'spam_vectorizer.pkl'))
    model_path = resource_path(os.path.join(MODELS_DIR, 'spam_model.pkl'))

    try:
        vectorizer = joblib.load(vec_path)
        model = joblib.load(model_path)
    except FileNotFoundError:
        print("BŁĄD: brak modeli spam w models/")
        return df, pd.DataFrame()

    df = df.copy()
    df['clean_review'] = df['review'].apply(clean_text).fillna("")

    X = vectorizer.transform(df['clean_review'])
    try:
        scores = model.decision_function(X)
        df['is_spam'] = (scores > -1.5).astype(int)
    except Exception:
        df['is_spam'] = model.predict(X).astype(int)

    clean_df = df[df['is_spam'] == 0].copy()
    spam_df = df[df['is_spam'] == 1].copy()

    print(f"Odrzucono {len(df) - len(clean_df)} recenzji jako treści spamowe")
    return clean_df, spam_df

# liczymy ogólny sentyment Flairem
def load_flair_classifier():
    model_path = resource_path(os.path.join(MODELS_DIR, 'flair_sentiment.pt'))
    classifier = TextClassifier.load(model_path)
    return classifier

def flair_predict_binary(classifier, text: str):
    if not text:
        return 0
    sent = Sentence(str(text)[:512])
    classifier.predict(sent)
    lab = sent.labels[0]
    return 1 if lab.value == "POSITIVE" else 0

def analyze_sentiment(df, expand_dicts: bool = True):
    print("Analiza sentymentu (Flair)")

    global KEYWORD_CATEGORIES, EXPECTATION_CATEGORIES

    df = df.copy()
    df['clean_review'] = df['review'].apply(clean_text).fillna("")

    if expand_dicts:
        KEYWORD_CATEGORIES = run_dictionary_expansion(df['clean_review'], KEYWORD_CATEGORIES)
        EXPECTATION_CATEGORIES = run_dictionary_expansion(df['clean_review'], EXPECTATION_CATEGORIES)

    try:
        classifier = load_flair_classifier()
    except Exception as e:
        print(f"BŁĄD ładowania Flair: {e}")
        return df

    sentiments = []
    total = len(df)

    print("Przetwarzanie tekstu za pomocą modelu Flair")
    for i, text in enumerate(df['review']):
        sentiments.append(flair_predict_binary(classifier, str(text) if text else ""))
        if i % 200 == 0:
            print(f"\rAnaliza: {i}/{total}", end="")

    df['sentiment_flair'] = sentiments
    print("\nSentyment określony")

    print("Przypisywanie kategorii i oczekiwań do treści recenzji")
    df['review_category'] = df['clean_review'].apply(lambda x: assign_categories(x, KEYWORD_CATEGORIES))
    df['expectation_category'] = df['clean_review'].apply(lambda x: assign_categories(x, EXPECTATION_CATEGORIES))

    return df

# liczymy sentyment ABSA regresją logistyczną
def load_absa_lr_models():
    model_path = resource_path(os.path.join(MODELS_DIR, "bg3_absa_model.pkl"))
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Brak: {model_path}")

    pipe = joblib.load(model_path)

    # szybka walidacja
    if not hasattr(pipe, "predict_proba"):
        raise ValueError("Załadowany model ABSA nie ma metody predict_proba(). Sprawdź artefakt pkl.")
    return pipe

def build_aspect_space():
    aspects = {}
    for k, kws in KEYWORD_CATEGORIES.items():
        aspects[f"theme::{k}"] = kws
    for k, kws in EXPECTATION_CATEGORIES.items():
        aspects[f"expect::{k}"] = kws
    return aspects

def _compile_kw_pattern(kw: str):
    kw = str(kw).strip().lower()
    kw_escaped = re.escape(kw).replace(r"\ ", r"\s+")
    return re.compile(rf"\b{kw_escaped}\b", flags=re.IGNORECASE)

def compile_aspect_patterns(aspects):
    patterns = {}
    for a, kws in aspects.items():
        pats = []
        for kw in kws:
            kw = str(kw).strip().lower()
            if not kw:
                continue
            pats.append(_compile_kw_pattern(kw))
        patterns[a] = pats
    return patterns

def count_aspect_tokens(text: str, aspect_patterns):
    t = str(text).lower()
    counts = {}
    for a, pats in aspect_patterns.items():
        c = 0
        for p in pats:
            c += len(p.findall(t))
        if c > 0:
            counts[a] = c
    return counts

def compute_global_weights(df: pd.DataFrame, aspect_patterns):
    global_counts = {a: 0 for a in aspect_patterns.keys()}
    general_count = 0

    for txt in df["clean_review"].tolist():
        c = count_aspect_tokens(txt, aspect_patterns)
        if not c:
            general_count += 1
        else:
            for a, n in c.items():
                global_counts[a] += int(n)

    global_counts[GENERAL_ASPECT] = int(general_count)

    total = sum(global_counts.values())
    if total <= 0:
        global_w = {GENERAL_ASPECT: 1.0}
    else:
        global_w = {k: v / total for k, v in global_counts.items()}

    df_global = pd.DataFrame(
        sorted(global_w.items(), key=lambda x: x[1], reverse=True),
        columns=["aspect", "global_weight"]
    )
    return global_w, df_global


def compute_local_weights(text: str, aspect_patterns):
    counts = count_aspect_tokens(text, aspect_patterns)
    if not counts:
        return {GENERAL_ASPECT: 1.0}
    s = sum(counts.values())
    return {a: counts[a] / s for a in counts}

def compute_total_weights(text: str, global_w, aspect_patterns):
    local_w = compute_local_weights(text, aspect_patterns)
    total_w = {a: float(global_w.get(a, 0.0) * lw) for a, lw in local_w.items()}
    s = sum(total_w.values())
    if s > 0:
        total_w = {a: w / s for a, w in total_w.items()}
    else:
        total_w = {GENERAL_ASPECT: 1.0}
    return total_w

def extract_aspect_texts(review: str, aspect_patterns):
    text = str(review).strip()
    if not text:
        return {GENERAL_ASPECT: ""}

    sents = re.split(r'(?<=[.!?])\s+', text)
    buckets = {}

    for s in sents:
        s_l = s.lower()
        for a, pats in aspect_patterns.items():
            if any(p.search(s_l) for p in pats):
                buckets.setdefault(a, []).append(s)

    if not buckets:
        return {GENERAL_ASPECT: text}

    return {a: " ".join(v) for a, v in buckets.items()}

def absa_lr(df: pd.DataFrame):

    absa_pipe = load_absa_lr_models()

    aspects = build_aspect_space()
    aspect_patterns = compile_aspect_patterns(aspects)

    # globalny baseline
    global_w, df_global_w = compute_global_weights(df, aspect_patterns)

    # progi trójklasowe (neg/neutr/poz)
    POS_THR = 0.55
    NEG_THR = 0.45

    def proba_to_score(p_pos: float) -> float:
        return float(2.0 * p_pos - 1.0)  # [-1, 1]

    def proba_to_label_012(p_pos: float) -> int:
        # 0=neg, 2=neutral, 1=pos
        if p_pos >= POS_THR:
            return 1
        if p_pos <= NEG_THR:
            return 0
        return 2

    df_in = df.copy()

    # przygotowanie struktur
    review_scores = []
    review_labels = []
    review_aspects_json = []
    review_weights_json = []
    review_detail_json = []

    aspect_rows = []

    total = len(df_in)
    for i, row in df_in.iterrows():
        rid = str(row.get("review_id"))
        date_str = row.get("date_str")
        try:
            date_dt = pd.to_datetime(date_str)
        except Exception:
            date_dt = pd.NaT

        review_text = row.get("review", "")
        clean_rev = row.get("clean_review", "")

        # aspekty + kontekst
        aspect_texts = extract_aspect_texts(review_text, aspect_patterns)
        aspects_here = list(aspect_texts.keys())

        # wagi aspektów
        w_total = compute_total_weights(clean_rev, global_w, aspect_patterns)

        # normalizacja aspektów
        s = sum(w_total.get(a, 0.0) for a in aspects_here)
        if s > 0:
            w_total = {a: w_total.get(a, 0.0) / s for a in aspects_here}
        else:
            aspects_here = [GENERAL_ASPECT]
            aspect_texts = {GENERAL_ASPECT: review_text}
            w_total = {GENERAL_ASPECT: 1.0}

        # sentyment dla każdego aspektu dzięki regresji logistycznej
        detail = {}
        agg_score = 0.0

        for a in aspects_here:
            a_txt = aspect_texts.get(a, "")
            p_pos = float(absa_pipe.predict_proba([a_txt])[0, 1])
            score = proba_to_score(p_pos)
            lbl = proba_to_label_012(p_pos)

            weight = float(w_total.get(a, 0.0))
            agg_score += weight * score

            detail[a] = {
                "p_pos": p_pos,
                "score": score,
                "label": int(lbl),
                "weight": weight
            }

            aspect_rows.append({
                "review_id": rid,
                "date": date_dt,
                "aspect": a,
                "weight": weight,
                "p_pos": p_pos,
                "aspect_score": score
            })

        # labelka
        if agg_score >= 0.1:
            rlab = 1
        elif agg_score <= -0.1:
            rlab = 0
        else:
            rlab = 2

        review_scores.append(float(agg_score))
        review_labels.append(int(rlab))
        review_aspects_json.append(json.dumps(aspects_here, ensure_ascii=False))
        review_weights_json.append(json.dumps(w_total, ensure_ascii=False))
        review_detail_json.append(json.dumps(detail, ensure_ascii=False))

        if (len(review_scores) % 500) == 0:
            print(f"\rObliczanie sentymentu dla każdego aspektu: {len(review_scores)}/{total}", end="")

    print("\nModel określił sentyment dla kategorii tematycznych")

    df_review_level = df_in.copy()
    df_review_level["absa_lr_score"] = review_scores
    df_review_level["absa_lr_label"] = review_labels
    df_review_level["absa_lr_aspects"] = review_aspects_json
    df_review_level["absa_lr_weights"] = review_weights_json
    df_review_level["absa_lr_detail"] = review_detail_json

    df_aspect_level = pd.DataFrame(aspect_rows)

    return df_review_level, df_aspect_level, df_global_w

# liczymy prognozy dla sentymentu ogólnego
def run_predictions(df):
    print("Generowanie prognoz")

    df = df.copy()
    df['date'] = pd.to_datetime(df['date_str'])

    daily = df.groupby('date').agg(
        avg_sentiment=('sentiment_flair', 'mean'),
        pos_count=('voted_up', 'sum'),
        total_count=('voted_up', 'count')
    ).sort_index()

    daily['ratio'] = daily['pos_count'] / daily['total_count']
    daily['ratio_smooth'] = daily['ratio'].rolling(7, min_periods=1).mean()
    daily['count_log'] = np.log1p(daily['total_count'])

    last_date = daily.index.max()
    future_dates = pd.date_range(last_date + timedelta(days=1), periods=120)

    features = pd.DataFrame(index=future_dates)
    features['t'] = range(len(daily), len(daily) + 120)
    features['day_of_week'] = features.index.dayofweek
    features['week_of_year'] = features.index.isocalendar().week.astype(int)
    features['month'] = features.index.month

    last_row = daily.iloc[-1]
    features['lag1'] = last_row['avg_sentiment']
    features['lag7'] = last_row['avg_sentiment']
    features['rolling7'] = last_row['avg_sentiment']
    features['count_log'] = last_row['count_log']

    future_df = pd.DataFrame(index=future_dates)

    try:
        sent_model = joblib.load(resource_path(os.path.join(MODELS_DIR, 'sentiment_forecast_model.pkl')))
        if "Prophet" in str(type(sent_model)):
            p_future = pd.DataFrame({'ds': future_dates})
            pred_sent = sent_model.predict(p_future)['yhat'].values
        else:
            pred_sent = sent_model.predict(features)
        future_df['predicted_sentiment'] = pred_sent
    except Exception as e:
        print(f"Błąd prognozy sentymentu: {e}")

    try:
        ratio_model = joblib.load(resource_path(os.path.join(MODELS_DIR, 'ratio_forecast_model.pkl')))
        feats_ratio = features.drop(columns=['count_log'], errors='ignore')
        if "Prophet" in str(type(ratio_model)):
            p_future = pd.DataFrame({'ds': future_dates})
            pred_ratio = ratio_model.predict(p_future)['yhat'].values
        else:
            pred_ratio = ratio_model.predict(feats_ratio)
        future_df['predicted_ratio'] = np.clip(pred_ratio, 0, 1)
    except Exception as e:
        print(f"Błąd prognozy ratio: {e}")

    future_df = future_df.reset_index().rename(columns={'index': 'date'})
    return future_df


def export_data(reviews_df, forecast_df):
    print("Tworzenie plików .hyper dla Tableau (z metodą EXPLODE)")

    def fix_types_for_hyper(df):
        df = df.copy()
        for col in df.select_dtypes(include=['float32']).columns:
            df[col] = df[col].astype('float64')
        return df

    reviews_out = reviews_df[['review_id', 'review', 'voted_up', 'date_str',
                              'sentiment_flair', 'review_category', 'expectation_category']].copy()

    # rozbijanie list kategorii żeby w Tableau dobrze działało
    reviews_out = reviews_out.explode('review_category')

    reviews_out['review_id'] = reviews_out['review_id'].astype(str)
    reviews_out['date_str'] = pd.to_datetime(reviews_out['date_str'], errors='coerce')
    
    reviews_out['review_category'] = reviews_out['review_category'].astype(str)
    reviews_out['expectation_category'] = reviews_out['expectation_category'].astype(str)

    reviews_out = fix_types_for_hyper(reviews_out)

    # zapis do Hyper
    try:
        pantab.frame_to_hyper(reviews_out, "BG3_Reviews_Analysis.hyper", table=TableName("public", "Reviews"))
        print("zapisano w BG3_Reviews_Analysis.hyper (dane rozbite na kategorie)")
    except Exception as e:
        print(f"BłądHyper (Reviews): {e}")
        reviews_out.to_csv("BG3_Reviews_Analysis.csv", sep=";", index=False, encoding='utf-8')

    forecast_df = fix_types_for_hyper(forecast_df)
    try:
        pantab.frame_to_hyper(forecast_df, "BG3_Forecasts.hyper", table=TableName("public", "Forecast"))
        print("zapisano w BG3_Forecasts.hyper")
    except Exception as e:
        print(f"BłądHyper (Forecast): {e}")

def _sqltype_for_series(s: pd.Series) -> SqlType:
    if pd.api.types.is_integer_dtype(s):
        return SqlType.big_int()
    if pd.api.types.is_float_dtype(s):
        return SqlType.double()
    if pd.api.types.is_datetime64_any_dtype(s):
        return SqlType.timestamp()
    return SqlType.text()

def _normalize_for_hyper(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = out[col].astype(str)
    for col in out.select_dtypes(include=['float32']).columns:
        out[col] = out[col].astype('float64')
    return out

# eksport absy do hypera
def export_absa_hyper(df_review_level: pd.DataFrame,
                      df_aspect_level: pd.DataFrame,
                      df_global_w: pd.DataFrame,
                      out_path: str = "BG3_ABSA.hyper"):
    print(f"Eksportowanie analizy sentymentu dla kategorii (ABSA) do {out_path}")

    r = df_review_level[['review_id', 'date_str', 'voted_up',
                         'absa_lr_score', 'absa_lr_label',
                         'absa_lr_aspects', 'absa_lr_weights', 'absa_lr_detail']].copy()
    
    a = df_aspect_level.copy()
    g = df_global_w.copy()

    r['review_id'] = r['review_id'].astype(str)
    r['date_str'] = pd.to_datetime(r['date_str'], errors='coerce').dt.tz_localize(None)

    if "date" in a.columns:
        a["date"] = pd.to_datetime(a["date"], errors="coerce").dt.tz_localize(None)
    a["review_id"] = a["review_id"].astype(str)

    def clean_aspect_name(val):    # trzeba wyczyścić nazwy bo w Tableau brzydko wygląda
        val = str(val)
        if "general::other" in val:
            return "Ogólne / Inne"
        if "expect::" in val:
            clean = val.replace("expect::", "").capitalize()
            return f"Oczekiwania: {clean}"
        return val.replace("theme::", "").capitalize()

    a['aspect'] = a['aspect'].apply(clean_aspect_name)
    g['aspect'] = g['aspect'].apply(clean_aspect_name)

    # tłumaczenie etykiet (0,1,2 -> Tekst)
    sentiment_map = {0: "Negatywny", 1: "Pozytywny", 2: "Neutralny"}
    r['absa_lr_label'] = r['absa_lr_label'].map(sentiment_map)
    
    steam_map = {0: "Negatywne", 1: "Pozytywne"}
    r['voted_up'] = r['voted_up'].map(steam_map)
    
    r = r.rename(columns={
        'review_id': 'ID_Recenzji',
        'date_str': 'Data',
        'voted_up': 'Ocena_Steam',
        'absa_lr_score': 'Sentyment_Cala_Recenzja_Wynik',  # (-1 do 1)
        'absa_lr_label': 'Sentyment_Cala_Recenzja_Opis',   # (poz/neg/neutr)
        'absa_lr_aspects': 'Tech_Lista_Aspektow',
        'absa_lr_weights': 'Tech_Wagi_JSON',
        'absa_lr_detail': 'Tech_Szczegoly_JSON'
    })

    a = a.rename(columns={
        'review_id': 'ID_Recenzji',
        'date': 'Data',
        'aspect': 'Kategoria_Aspektu',
        'weight': 'Waga_(Waznosc_Tematu)',
        'p_pos': 'Prawdopodobienstwo_Pozytywu',
        'aspect_score': 'Sentyment_Aspektu_Wynik' # (-1 do 1)
    })

    g = g.rename(columns={
        'aspect': 'Kategoria_Aspektu',
        'global_weight': 'Globalna_Popularnosc_Tematu'
    })

    r = _normalize_for_hyper(r)
    a = _normalize_for_hyper(a)
    g = _normalize_for_hyper(g)

    tables = {
        "Recenzje_Podsumowanie": r,
        "Analiza_Aspektow": a,
        "Wagi_Globalne": g
    }

    with HyperProcess(telemetry=Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU) as hyper:
        with Connection(endpoint=hyper.endpoint,
                        database=out_path,
                        create_mode=CreateMode.CREATE_AND_REPLACE) as conn:
            
            conn.execute_command('CREATE SCHEMA "Extract"')
            for tname, df in tables.items():
                cols = [TableDefinition.Column(c, _sqltype_for_series(df[c])) for c in df.columns]
                
                table_def = TableDefinition(table_name=TableName("Extract", tname), columns=cols)
                
                conn.catalog.create_table(table_def)

                rows = df.itertuples(index=False, name=None)
                with Inserter(conn, table_def) as inserter:
                    inserter.add_rows(rows)
                    inserter.execute()

    print(f"Zapisano w {out_path}")


# === MAIN PIPELINE === - wygląd aplikacji .exe

if __name__ == "__main__":
    try:
        print("=" * 92)
        print("Autorskie narzędzie informatyczne do analizy opinii z portalu Steam dla firmy Larian Studios")
        print("=" * 92)
        start_time = time.time()
        EXPAND_DICTIONARIES = True
        df_raw = fetch_reviews()
        if df_raw.empty:
            print("Brak danych do przetworzenia")
            input("Wyjście z aplikacji jest możliwe")
            sys.exit(0)
        df_nospam, df_spam = filter_spam(df_raw)
        if df_nospam.empty:
            print("Po spam-filtrze nie zostały żadne dane.")
            input("Wyjście z aplikacji jest możliwe")
            sys.exit(0)
        df_sent = analyze_sentiment(df_nospam, expand_dicts=EXPAND_DICTIONARIES)
        forecast_df = run_predictions(df_sent)
        export_data(df_sent, forecast_df)
        df_absa_review, df_absa_aspect, df_absa_global = absa_lr(df_sent)
        export_absa_hyper(df_absa_review, df_absa_aspect, df_absa_global, out_path="BG3_ABSA.hyper")
        print("Proces zakończony")
        print(f"Czas wykonania zajął {round(time.time() - start_time, 2)} sekund")
        print("Pliki .hyper są gotowe do otwarcia w Tableau:")
        print("- BG3_Reviews_Analysis.hyper")
        print("- BG3_Forecasts.hyper")
        print("- BG3_ABSA.hyper")
    except Exception:
        print("\nWystąpił błąd")
        print("\nSzczegóły: ")
        traceback.print_exc()
    print("\n")
    input("Wyjście z aplikacji jest możliwe")
