"""
detector/fingerprints.py
Diccionarios, patrones regex y marcadores estilométricos característicos de
modelos de Inteligencia Artificial (ChatGPT / GPT-4o / o1, Gemini, Claude, Perplexity, DeepSeek)
y artefactos de humanizadores superficiales tanto en Español como en Inglés.
"""

import re
from typing import Dict, List, Tuple

# Expresiones y clichés típicos de ChatGPT / OpenAI (GPT-4, 4o, o1)
CHATGPT_PHRASES_ES = [
    r"\ben un mundo cada vez m[aá]s\b",
    r"\b(?:desempe[nñ]a|juega|cumple)?\s*un papel (?:crucial|fundamental|clave)\b",
    r"\bcabe (?:destacar|mencionar|resaltar)(?: que)?\b",
    r"\bes fundamental (?:destacar|recordar|entender|mencionar|tener en cuenta|se[nñ]alar)\b",
    r"\bes importante (?:destacar|recordar|entender|mencionar|resaltar|se[nñ]alar)\b",
    r"\bes crucial (?:comprender|destacar|entender|considerar|encontrar|se[nñ]alar)\b",
    r"\ben este sentido,?\b",
    r"\ben [uú]ltima instancia,?\b",
    r"\ba su vez,?\b",
    r"\ben resumen,?\b",
    r"\ben conclusi[oó]n,?\b",
    r"\bpara concluir,?\b",
    r"\bde vital importancia\b",
    r"\bun tapiz de\b",
    r"\bun testimonio de\b",
    r"\bun faro de\b",
    r"\badentrarse en\b",
    r"\bprofundizar en\b",
    r"\ben el panorama (?:actual|digital|moderno|contempor[aá]neo)\b",
    r"\bno solo [^.,;]+ sino tambi[eé]n\b",
    r"\bun recordatorio de que\b",
    r"\ben constante evoluci[oó]n\b",
    r"\bofrece una amplia gama de\b",
    r"\bfomentar un ambiente\b",
    r"\buna espada de doble filo\b",
    r"\ben el gran esquema de las cosas\b",
    r"\bvale la pena se[nñ]alar(?: que)?\b",
    r"\bun equilibrio delicado entre\b",
    r"\buna pieza angular\b",
    r"\buna piedra angular\b",
    r"\bcamino por recorrer\b",
    r"\bqueda mucho por hacer\b",
    r"\ben nuestro planeta\b",
    r"\ben nuestra sociedad\b"
]

CHATGPT_PHRASES_EN = [
    r"\bdelve(?:s|d)? into\b",
    r"\ba testament to\b",
    r"\brich tapestry\b",
    r"\bbeacon of\b",
    r"\bin today's fast-paced world\b",
    r"\bin today's digital age\b",
    r"\bit is important to remember that\b",
    r"\bit is worth noting that\b",
    r"\bit's worth noting that\b",
    r"\bplays a crucial role\b",
    r"\bplays a pivotal role\b",
    r"\bplays a vital role\b",
    r"\bserves as a reminder\b",
    r"\bnavigating the (?:complexities|landscape|intricacies)\b",
    r"\bever-evolving landscape\b",
    r"\bmultifaceted\b",
    r"\bparamount\b",
    r"\bunderscores the importance\b",
    r"\bholistic approach\b",
    r"\bin conclusion,\b",
    r"\bto sum up,\b",
    r"\bmoreover,\b",
    r"\bfurthermore,\b",
    r"\bnot only [^.,;]+ but also\b",
    r"\ba double-edged sword\b",
    r"\bfosters? a sense of\b",
    r"\bin the realm of\b",
    r"\ba beacon of hope\b",
    r"\bcornerstone of\b"
]

# Expresiones de Gemini (Google)
GEMINI_PHRASES_ES = [
    r"\baqu[ií] tienes un desglose\b",
    r"\bexploremos\b",
    r"\bpuntos clave:\b",
    r"\ba continuaci[oó]n se presentan\b",
    r"\ben t[eé]rminos generales,?\b",
    r"\bveamos m[aá]s de cerca\b",
    r"\bcomo modelo de lenguaje\b",
    r"\ben resumen:\b",
    r"\bdesglose detallado\b",
    r"\baspectos destacados:\b",
    r"\ben resumidas cuentas,?\b"
]

GEMINI_PHRASES_EN = [
    r"\bhere's a breakdown\b",
    r"\blet's explore\b",
    r"\bkey takeaways:\b",
    r"\bhere is what you need to know\b",
    r"\bdiving deeper\b",
    r"\bin terms of\b",
    r"\boverall, it is\b",
    r"\blet's take a closer look\b"
]

# Expresiones de Claude / Anthropic
CLAUDE_PHRASES_ES = [
    r"\bes importante reconocer que\b",
    r"\bsi bien existen argumentos v[aá]lidos\b",
    r"\buna perspectiva matizada\b",
    r"\bconviene ser cauteloso\b",
    r"\bes crucial sopesar\b",
    r"\bdesde una perspectiva equilibrada\b",
    r"\bresulta prudente se[nñ]alar\b"
]

CLAUDE_PHRASES_EN = [
    r"\bit is important to recognize that\b",
    r"\bwhile there are valid arguments on both sides\b",
    r"\ba nuanced perspective\b",
    r"\bit is worth being cautious\b",
    r"\bbalanced approach\b",
    r"\bfrom a broader perspective\b",
    r"\bit's reasonable to conclude\b"
]

# Expresiones de Perplexity
PERPLEXITY_PHRASES_ES = [
    r"\bseg[uú]n los hallazgos recientes\b",
    r"\blos estudios sugieren que\b",
    r"\bla evidencia se[nñ]ala que\b",
    r"\blos datos indican que\b",
    r"\bde acuerdo con las fuentes\b",
    r"\ben base a investigaciones recientes\b"
]

PERPLEXITY_PHRASES_EN = [
    r"\baccording to recent findings\b",
    r"\bstudies suggest that\b",
    r"\bevidence indicates that\b",
    r"\bdata shows that\b",
    r"\bbased on available research\b",
    r"\bsources indicate that\b"
]

# Expresiones de DeepSeek
DEEPSEEK_PHRASES_ES = [
    r"\banalicemos paso a paso\b",
    r"\bdesglosando el razonamiento\b",
    r"\bde manera exhaustiva\b",
    r"\bexaminando minuciosamente\b"
]

DEEPSEEK_PHRASES_EN = [
    r"\blet's analyze step by step\b",
    r"\bbreaking down the reasoning\b",
    r"\bcomprehensive analysis\b",
    r"\bexamining thoroughly\b"
]

ALL_AI_PATTERNS = {
    "chatgpt_es": [re.compile(p, re.IGNORECASE) for p in CHATGPT_PHRASES_ES],
    "chatgpt_en": [re.compile(p, re.IGNORECASE) for p in CHATGPT_PHRASES_EN],
    "gemini_es": [re.compile(p, re.IGNORECASE) for p in GEMINI_PHRASES_ES],
    "gemini_en": [re.compile(p, re.IGNORECASE) for p in GEMINI_PHRASES_EN],
    "claude_es": [re.compile(p, re.IGNORECASE) for p in CLAUDE_PHRASES_ES],
    "claude_en": [re.compile(p, re.IGNORECASE) for p in CLAUDE_PHRASES_EN],
    "perplexity_es": [re.compile(p, re.IGNORECASE) for p in PERPLEXITY_PHRASES_ES],
    "perplexity_en": [re.compile(p, re.IGNORECASE) for p in PERPLEXITY_PHRASES_EN],
    "deepseek_es": [re.compile(p, re.IGNORECASE) for p in DEEPSEEK_PHRASES_ES],
    "deepseek_en": [re.compile(p, re.IGNORECASE) for p in DEEPSEEK_PHRASES_EN],
}

def detect_cliches_and_signatures(text: str) -> Dict[str, any]:
    """
    Busca coincidencias con frases cliché y firmas de IA.
    Retorna conteo total, lista de coincidencias con posiciones y atribución por modelo.
    """
    matches = []
    model_hits = {
        "ChatGPT / OpenAI": 0,
        "Google Gemini": 0,
        "Claude (Anthropic)": 0,
        "Perplexity AI": 0,
        "DeepSeek": 0
    }

    mapping = {
        "chatgpt_es": "ChatGPT / OpenAI",
        "chatgpt_en": "ChatGPT / OpenAI",
        "gemini_es": "Google Gemini",
        "gemini_en": "Google Gemini",
        "claude_es": "Claude (Anthropic)",
        "claude_en": "Claude (Anthropic)",
        "perplexity_es": "Perplexity AI",
        "perplexity_en": "Perplexity AI",
        "deepseek_es": "DeepSeek",
        "deepseek_en": "DeepSeek"
    }

    for key, regex_list in ALL_AI_PATTERNS.items():
        model_name = mapping[key]
        for rgx in regex_list:
            for m in rgx.finditer(text):
                phrase = m.group(0)
                matches.append({
                    "phrase": phrase,
                    "model": model_name,
                    "start": m.start(),
                    "end": m.end()
                })
                model_hits[model_name] += 1

    dominant_model = None
    max_hits = 0
    for model, count in model_hits.items():
        if count > max_hits:
            max_hits = count
            dominant_model = model

    return {
        "total_hits": len(matches),
        "matches": matches,
        "model_distribution": model_hits,
        "dominant_signature": dominant_model if max_hits > 0 else "Neutral / Sin firma específica"
    }
