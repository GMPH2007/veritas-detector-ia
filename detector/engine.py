"""
detector/engine.py
Motor central de detección de Inteligencia Artificial.
Combina la red neuronal entrenada, análisis de perplejidad, burstiness,
entropía de Shannon y firmas lingüísticas de ChatGPT, Gemini, Claude y Perplexity.
Genera además un mapa de calor (heatmap) oración por oración.
"""

import os
import joblib
import numpy as np
from typing import Dict, List, Any
from detector.features import (
    extract_features,
    split_into_sentences,
    tokenize_words,
    calculate_char_ngram_entropy,
    calculate_shannon_entropy
)
from detector.fingerprints import detect_cliches_and_signatures

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_weights.joblib")

class AIDetectorEngine:
    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        if os.path.exists(MODEL_PATH):
            try:
                self.model = joblib.load(MODEL_PATH)
            except Exception as e:
                print(f"Advertencia: No se pudo cargar el modelo joblib: {e}")
                self.model = None
        else:
            print(f"Modelo no encontrado en {MODEL_PATH}. Entrenando nuevo modelo...")
            from detector.train_model import train_and_save_model
            self.model = train_and_save_model()

    def analyze_sentence(self, sentence: str, global_burstiness: float) -> Dict[str, Any]:
        """
        Analiza una oración individual para generar el mapa de calor (Heatmap).
        """
        words = tokenize_words(sentence)
        n_words = len(words)
        if n_words < 3:
            return {
                "text": sentence,
                "ai_probability": 10.0,
                "label": "safe",
                "tag": "Muy corta para evaluar",
                "reason": "Frase breve o fragmento."
            }

        # 1. Chequeo de clichés y frases de IA en esta oración
        fp = detect_cliches_and_signatures(sentence)
        has_cliche = fp["total_hits"] > 0

        # 2. Entropía y regularidad de la oración
        char_entropy = calculate_char_ngram_entropy(sentence, n=3)
        word_entropy = calculate_shannon_entropy(words)

        # 3. Puntuación heurística de la oración
        score = 25.0  # Base neutral

        # Clichés pesan fuertemente
        if has_cliche:
            score += 45.0 * fp["total_hits"]

        # Entropía baja = muy predecible
        if char_entropy < 3.2:
            score += 20.0
        elif char_entropy > 4.5:
            score -= 15.0

        # Longitud y simetría típica de LLMs (18-35 palabras, ritmo monótono)
        if 16 <= n_words <= 32:
            score += 15.0
        elif n_words < 8 or n_words > 45:
            score -= 10.0

        # Si el texto global tiene burstiness muy baja, las oraciones son más sospechosas
        if global_burstiness < 0.35:
            score += 10.0
        elif global_burstiness > 0.65:
            score -= 15.0

        score = max(2.0, min(99.0, score))

        # Determinar etiqueta de color
        if score >= 70.0:
            label = "danger"
            tag = "Alta probabilidad de IA"
            reason = f"Frase altamente predecible o con conectores típicos de LLMs ({fp['dominant_signature'] if has_cliche else 'Sintaxis uniforme'})."
        elif score >= 45.0:
            label = "warning"
            tag = "Posible asistencia de IA"
            reason = "Estructura intermedia con vocabulario formal o predecible."
        else:
            label = "safe"
            tag = "Probable autoría humana"
            reason = "Variabilidad natural en vocabulario y cadencia."

        return {
            "text": sentence,
            "ai_probability": round(score, 1),
            "label": label,
            "tag": tag,
            "reason": reason,
            "detected_phrases": [m["phrase"] for m in fp["matches"]]
        }

    def detect(self, text: str) -> Dict[str, Any]:
        """
        Ejecuta el análisis completo del texto combinando la red neuronal,
        estadísticas estocásticas, firmas de IA y análisis granular por oración.
        """
        clean_text = text.strip()
        if not clean_text:
            return {
                "error": "El texto ingresado está vacío.",
                "ai_probability": 0.0,
                "human_probability": 100.0
            }

        # Extraer características
        features = extract_features(clean_text)
        total_words = features["total_words"]

        if total_words < 10:
            return {
                "error": "El texto es demasiado corto (mínimo 10 palabras para un análisis confiable).",
                "ai_probability": 0.0,
                "human_probability": 100.0,
                "total_words": total_words
            }

        # 1. Inferencia de la Red Neuronal
        vector = features["features_vector"].reshape(1, -1)
        if self.model is not None:
            try:
                # Obtener probabilidades calibradas [P(Humano), P(IA)]
                probs = self.model.predict_proba(vector)[0]
                nn_ai_prob = float(probs[1]) * 100.0
            except Exception as e:
                print(f"Error en predicción neuronal: {e}")
                nn_ai_prob = 50.0
        else:
            nn_ai_prob = 50.0

        # Factores determinantes conocidos
        burstiness = features["burstiness"]
        cliche_density = features["cliche_density"]
        cliche_count = features["cliche_count"]

        # 2. Análisis granular oración por oración (Mapa de Calor)
        sentences = split_into_sentences(clean_text)
        sentence_details = [
            self.analyze_sentence(s, burstiness) for s in sentences
        ]

        sentence_scores = [s["ai_probability"] for s in sentence_details]
        avg_sentence_score = float(np.mean(sentence_scores)) if sentence_scores else 50.0
        danger_count = sum(1 for s in sentence_details if s["label"] == "danger")
        danger_ratio = danger_count / max(1, len(sentence_details))

        # 3. Calibración combinada Multicapa
        adjusted_score = (0.6 * nn_ai_prob) + (0.4 * avg_sentence_score)

        # Clichés de IA aumentan fuertemente la probabilidad
        if cliche_count >= 1:
            adjusted_score = max(adjusted_score, min(99.0, 50.0 + (cliche_count * 15.0)))

        # Si no hay clichés, ni oraciones en peligro, y la cadencia (burstiness) es alta:
        if cliche_count == 0 and danger_ratio == 0:
            if burstiness > 0.55:
                adjusted_score = min(adjusted_score * 0.45, avg_sentence_score)
            elif burstiness > 0.40:
                adjusted_score = min(adjusted_score * 0.70, avg_sentence_score * 1.1)

        # Burstiness muy baja (<0.32) con muchas palabras = alta sospecha de IA
        if burstiness < 0.32 and total_words > 40:
            adjusted_score = min(99.5, adjusted_score * 1.15 + 10.0)

        # Delimitar probabilidad final entre 1.0% y 99.2%
        final_ai_prob = round(max(1.0, min(99.2, adjusted_score)), 1)
        final_human_prob = round(100.0 - final_ai_prob, 1)

        # 4. Veredicto y diagnóstico
        if final_ai_prob >= 75.0:
            verdict_badge = "Generado por IA"
            verdict_color = "red"
            verdict_desc = "El texto muestra patrones inequívocos de modelos como ChatGPT, Gemini, Claude o Perplexity: cadencia uniforme, baja perplejidad y conectores sintéticos."
        elif final_ai_prob >= 45.0:
            verdict_badge = "Texto Híbrido / Asistido por IA"
            verdict_color = "yellow"
            verdict_desc = "El texto combina elementos naturales con estructuras o frases típicas de inteligencia artificial. Es posible que haya sido editado o reescrito con IA."
        elif final_ai_prob >= 20.0:
            verdict_badge = "Mayormente Humano"
            verdict_color = "blue"
            verdict_desc = "Predominan características humanas como variabilidad de ritmo y espontaneidad, aunque contiene ligeras expresiones formales."
        else:
            verdict_badge = "100% Humano"
            verdict_color = "green"
            verdict_desc = "El texto exhibe alta variabilidad estilística (burstiness alta), léxico espontáneo y perplejidad natural propia de un autor humano."

        ai_sentences_count = danger_count
        hybrid_sentences_count = sum(1 for s in sentence_details if s["label"] == "warning")
        human_sentences_count = sum(1 for s in sentence_details if s["label"] == "safe")

        # 5. Interpretación de métricas para el usuario
        burstiness_status = "Baja (Típica de IA)" if burstiness < 0.40 else ("Media" if burstiness < 0.65 else "Alta (Típica Humana)")
        perplexity_status = "Baja / Predecible" if features["word_perplexity"] < 40 else ("Moderada" if features["word_perplexity"] < 120 else "Alta / Diversa")

        return {
            "ai_probability": final_ai_prob,
            "human_probability": final_human_prob,
            "verdict": {
                "title": verdict_badge,
                "color": verdict_color,
                "description": verdict_desc
            },
            "metrics": {
                "total_words": total_words,
                "total_sentences": features["total_sentences"],
                "burstiness": {
                    "value": burstiness,
                    "status": burstiness_status,
                    "explanation": "Mide la variación en longitud y ritmo entre oraciones. Los humanos alternan frases cortas y largas; la IA es muy constante."
                },
                "perplexity": {
                    "value": features["word_perplexity"],
                    "status": perplexity_status,
                    "explanation": "Mide qué tan predecible es la elección de palabras. La IA elige las palabras más probables (baja perplejidad)."
                },
                "lexical_diversity": {
                    "value": features["ttr"],
                    "unique_words": int(features["ttr"] * total_words),
                    "yules_k": features.get("yules_k", 0.0),
                    "explanation": "Ratio de palabras únicas respecto al total."
                },
                "forensic_signals": {
                    "sent_len_range": features.get("sent_len_range", 0.0),
                    "punct_entropy": features.get("punct_entropy", 0.0),
                    "pronoun_1st_density": features.get("pronoun_1st_density", 0.0),
                    "yules_k": features.get("yules_k", 0.0)
                },
                "ai_phrases_detected": {
                    "count": features["cliche_count"],
                    "matches": features["cliche_matches"]
                },
                "dominant_model": features["dominant_model"],
                "model_signatures": features["model_distribution"]
            },
            "sentence_heatmap": sentence_details,
            "sentence_summary": {
                "ai_sentences": ai_sentences_count,
                "hybrid_sentences": hybrid_sentences_count,
                "human_sentences": human_sentences_count,
                "total": len(sentence_details)
            }
        }

# Instancia global reutilizable para máximo rendimiento
detector_engine = AIDetectorEngine()
