"""
detector/features.py
Extractor de características lingüísticas, estadísticas, entrópicas y estilométricas
de grado forense para la detección de texto generado por Inteligencia Artificial
(ChatGPT, Gemini, Claude, Perplexity, DeepSeek) y detección de intentos de evasión.
"""

import math
import re
from collections import Counter
from typing import Dict, List, Any
import numpy as np
from detector.fingerprints import detect_cliches_and_signatures

STOPWORDS_ES = {
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por", "un", "para",
    "con", "no", "una", "su", "al", "lo", "como", "más", "pero", "sus", "le", "ya", "o",
    "este", "sí", "porque", "esta", "entre", "cuando", "muy", "sin", "sobre", "también",
    "me", "hasta", "hay", "donde", "quien", "desde", "todo", "nos", "durante", "todos",
    "uno", "les", "ni", "contra", "otros", "ese", "eso", "ante", "ellos", "e", "esto",
    "mí", "antes", "algunos", "qué", "unos", "yo", "otro", "otras", "otra", "él", "tanto",
    "esa", "estos", "mucho", "quienes", "nada", "muchos", "cual", "poco", "ella", "estar",
    "estas", "algunas", "algo", "nosotros", "mi", "mis", "tú", "te", "ti", "tu", "tus", "ellas"
}

STOPWORDS_EN = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not",
    "on", "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from",
    "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all", "would",
    "there", "their", "what", "so", "up", "out", "if", "about", "who", "get", "which",
    "go", "me", "when", "make", "can", "like", "time", "no", "just", "him", "know",
    "take", "people", "into", "year", "your", "good", "some", "could", "them", "see",
    "other", "than", "then", "now", "look", "only", "come", "its", "over", "think", "also"
}

ALL_STOPWORDS = STOPWORDS_ES.union(STOPWORDS_EN)

PRONOUNS_SINGULAR_1ST_ES = {"yo", "me", "mi", "mis", "mí", "conmigo"}
PRONOUNS_SINGULAR_1ST_EN = {"i", "me", "my", "mine", "myself"}
ALL_PRONOUNS_SINGULAR_1ST = PRONOUNS_SINGULAR_1ST_ES.union(PRONOUNS_SINGULAR_1ST_EN)

PRONOUNS_1ST_ES = {"yo", "me", "mi", "mis", "mí", "conmigo", "nosotros", "nosotras", "nos"}
PRONOUNS_1ST_EN = {"i", "me", "my", "mine", "myself", "we", "us"}
ALL_PRONOUNS_1ST = PRONOUNS_1ST_ES.union(PRONOUNS_1ST_EN)


def split_into_sentences(text: str) -> List[str]:
    """
    Divide el texto en oraciones respetando abreviaturas comunes y signos ortográficos.
    """
    text = re.sub(r'\n+', ' ', text)
    raw_sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 3]
    return sentences if sentences else [text.strip()]


def tokenize_words(text: str) -> List[str]:
    """
    Extrae palabras individuales en minúsculas ignorando puntuación.
    """
    return re.findall(r'\b[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ0-9\'-]+\b', text.lower())


def calculate_shannon_entropy(items: List[str]) -> float:
    """Calcula la entropía de Shannon en bits: H = -sum(p * log2(p))."""
    if not items:
        return 0.0
    counter = Counter(items)
    n = len(items)
    entropy = 0.0
    for count in counter.values():
        p = count / n
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def calculate_char_ngram_entropy(text: str, n: int = 3) -> float:
    """Calcula la entropía de n-gramas de caracteres (indicador de perplejidad)."""
    clean = re.sub(r'\s+', ' ', text.lower()).strip()
    if len(clean) < n:
        return 0.0
    ngrams = [clean[i:i+n] for i in range(len(clean) - n + 1)]
    return calculate_shannon_entropy(ngrams)


def calculate_yules_k(words: List[str]) -> float:
    """
    Calcula la característica K de Yule (medida estilométrica de riqueza léxica independiente de la longitud).
    K = 10^4 * (sum(i^2 * V(i)) - N) / N^2
    """
    n = len(words)
    if n == 0:
        return 0.0
    counts = Counter(words)
    freq_spectrum = Counter(counts.values())
    s2 = sum((i ** 2) * vi for i, vi in freq_spectrum.items())
    return max(0.0, 10000.0 * (s2 - n) / (n ** 2 + 1e-5))


def calculate_punctuation_entropy(text: str) -> float:
    """Calcula la entropía de los signos de puntuación empleados."""
    puncts = re.findall(r'[,.;:!?—\-()[\]"\'«»¿¡]', text)
    if not puncts:
        return 0.0
    return calculate_shannon_entropy(puncts)


def calculate_adjacent_sentence_similarity(sentences: List[str]) -> float:
    """
    Mide la continuidad semántica entre oraciones consecutivas.
    La IA suele mantener una uniformidad y transición excesivamente regular.
    """
    if len(sentences) < 2:
        return 0.5

    sims = []
    for i in range(len(sentences) - 1):
        w1 = Counter(tokenize_words(sentences[i]))
        w2 = Counter(tokenize_words(sentences[i + 1]))
        
        all_words = set(w1.keys()).union(set(w2.keys()))
        if not all_words:
            continue
            
        dot = sum(w1.get(w, 0) * w2.get(w, 0) for w in all_words)
        norm1 = math.sqrt(sum(v**2 for v in w1.values()))
        norm2 = math.sqrt(sum(v**2 for v in w2.values()))
        
        if norm1 > 0 and norm2 > 0:
            sims.append(dot / (norm1 * norm2))
        else:
            sims.append(0.0)

    return float(np.mean(sims)) if sims else 0.5


def extract_features(text: str) -> Dict[str, Any]:
    """
    Extrae el vector ampliado de 30 dimensiones de características estadísticas,
    estocásticas, de perplejidad, burstiness, estilometría y firmas de IA.
    """
    words = tokenize_words(text)
    total_words = len(words)
    sentences = split_into_sentences(text)
    total_sentences = len(sentences)

    if total_words == 0:
        return {
            "total_words": 0,
            "total_sentences": 0,
            "burstiness": 0.0,
            "perplexity_score": 0.0,
            "ttr": 0.0,
            "char_entropy": 0.0,
            "cliche_count": 0,
            "cliche_density": 0.0,
            "dominant_model": "None",
            "features_vector": np.zeros(30, dtype=np.float32)
        }

    # 1. Longitud y Burstiness de Oraciones
    sent_lengths = [len(tokenize_words(s)) for s in sentences]
    avg_sent_len = float(np.mean(sent_lengths)) if sent_lengths else 0.0
    sent_len_std = float(np.std(sent_lengths)) if len(sent_lengths) > 1 else 0.0
    burstiness = (sent_len_std / (avg_sent_len + 1e-5))

    min_sent_len = float(min(sent_lengths)) if sent_lengths else 0.0
    max_sent_len = float(max(sent_lengths)) if sent_lengths else 0.0
    sent_len_range = max_sent_len - min_sent_len

    # 2. Longitud de Palabras
    word_lengths = [len(w) for w in words]
    avg_word_len = float(np.mean(word_lengths))
    word_len_std = float(np.std(word_lengths)) if len(word_lengths) > 1 else 0.0

    # 3. Entropía y Perplejidad
    char_entropy = calculate_shannon_entropy(list(text.lower()))
    char_trigram_entropy = calculate_char_ngram_entropy(text, n=3)
    word_entropy = calculate_shannon_entropy(words)
    word_perplexity = min(2.0 ** word_entropy, 500.0)

    # Varianza de entropía entre oraciones individuales (alta en humanos, plana en IA)
    per_sent_entropies = [calculate_shannon_entropy(tokenize_words(s)) for s in sentences if len(tokenize_words(s)) > 3]
    sent_entropy_std = float(np.std(per_sent_entropies)) if len(per_sent_entropies) > 1 else 0.0

    # 4. Riqueza Léxica y Estilometría
    unique_words = len(set(words))
    ttr = unique_words / total_words
    root_ttr = unique_words / (math.sqrt(total_words) + 1e-5)
    
    word_counts = Counter(words)
    hapax_count = sum(1 for c in word_counts.values() if c == 1)
    hapax_ratio = hapax_count / (unique_words + 1e-5)
    dislegomena_count = sum(1 for c in word_counts.values() if c == 2)
    dislegomena_ratio = dislegomena_count / (unique_words + 1e-5)

    yules_k = calculate_yules_k(words)

    # Pronombres de primera persona (naturalidad personal humana)
    pronoun_1st_count = sum(1 for w in words if w in ALL_PRONOUNS_1ST)
    pronoun_1st_density = (pronoun_1st_count / total_words) * 100.0

    # Stopwords
    stopword_count = sum(1 for w in words if w in ALL_STOPWORDS)
    stopword_ratio = stopword_count / total_words

    # 5. Puntuación y Sintaxis
    comma_count = text.count(',')
    comma_density = (comma_count / total_words) * 100
    semicolon_count = text.count(';')
    semicolon_density = (semicolon_count / total_words) * 100
    dash_count = text.count('-') + text.count('—')
    dash_density = (dash_count / total_words) * 100
    parenthesis_count = text.count('(') + text.count(')')
    parenthesis_density = (parenthesis_count / total_words) * 100
    question_excl_count = text.count('?') + text.count('¿') + text.count('!') + text.count('¡')
    question_excl_density = (question_excl_count / total_words) * 100

    punct_entropy = calculate_punctuation_entropy(text)

    # Viñetas y listas
    lines = text.split('\n')
    bullet_lines = sum(1 for l in lines if re.match(r'^\s*[-*•\d+.]\s', l))
    bullet_ratio = bullet_lines / max(len(lines), 1)

    # 6. Clichés y Firmas
    fingerprint_result = detect_cliches_and_signatures(text)
    cliche_count = fingerprint_result["total_hits"]
    cliche_density = (cliche_count / (total_words / 100.0)) if total_words >= 100 else (cliche_count * 1.5)

    # 7. Continuidad semántica entre oraciones adyacentes
    adjacent_sim = calculate_adjacent_sentence_similarity(sentences)

    # Vector numérico ordenado de 30 características para la red neuronal
    vector = np.array([
        avg_sent_len,           # 0: Longitud media de oraciones
        sent_len_std,           # 1: Desviación estándar de oraciones
        burstiness,             # 2: Burstiness (CV de oraciones)
        min_sent_len,           # 3: Mínima longitud de oración
        max_sent_len,           # 4: Máxima longitud de oración
        sent_len_range,         # 5: Rango de extensión de oraciones
        avg_word_len,           # 6: Longitud media de palabra
        word_len_std,           # 7: Desviación estándar de palabras
        char_entropy,           # 8: Entropía de caracteres
        char_trigram_entropy,   # 9: Entropía de trigramas
        word_entropy,           # 10: Entropía de palabras
        math.log(word_perplexity + 1.0), # 11: Log-perplejidad
        sent_entropy_std,       # 12: Varianza de entropía entre oraciones
        ttr,                    # 13: Type-Token Ratio
        root_ttr,               # 14: Root TTR
        hapax_ratio,            # 15: Palabras únicas usadas 1 vez
        dislegomena_ratio,      # 16: Palabras usadas exactamente 2 veces
        yules_k,                # 17: Constante K de Yule
        stopword_ratio,         # 18: Ratio de palabras vacías
        pronoun_1st_density,    # 19: Pronombres en primera persona
        comma_density,          # 20: Densidad de comas
        semicolon_density,      # 21: Densidad de punto y coma
        dash_density,           # 22: Densidad de guiones y rayas (—)
        parenthesis_density,    # 23: Densidad de paréntesis
        question_excl_density,  # 24: Signos emocionales / preguntas
        punct_entropy,          # 25: Entropía de signos de puntuación
        bullet_ratio,           # 26: Proporción de viñetas
        cliche_density,         # 27: Densidad de clichés de IA
        adjacent_sim,           # 28: Continuidad entre oraciones
        float(cliche_count),    # 29: Total de clichés detectados
    ], dtype=np.float32)

    return {
        "total_words": total_words,
        "total_sentences": total_sentences,
        "avg_sent_len": round(avg_sent_len, 2),
        "sent_len_std": round(sent_len_std, 2),
        "burstiness": round(burstiness, 3),
        "sent_len_range": round(sent_len_range, 1),
        "avg_word_len": round(avg_word_len, 2),
        "char_entropy": round(char_entropy, 3),
        "word_entropy": round(word_entropy, 3),
        "word_perplexity": round(word_perplexity, 2),
        "ttr": round(ttr, 3),
        "yules_k": round(yules_k, 2),
        "pronoun_1st_density": round(pronoun_1st_density, 2),
        "punct_entropy": round(punct_entropy, 3),
        "hapax_ratio": round(hapax_ratio, 3),
        "stopword_ratio": round(stopword_ratio, 3),
        "cliche_count": cliche_count,
        "cliche_density": round(cliche_density, 2),
        "cliche_matches": fingerprint_result["matches"],
        "model_distribution": fingerprint_result["model_distribution"],
        "dominant_model": fingerprint_result["dominant_signature"],
        "adjacent_sim": round(adjacent_sim, 3),
        "bullet_ratio": round(bullet_ratio, 3),
        "features_vector": vector
    }
