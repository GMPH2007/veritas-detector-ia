"""
detector/pdf_reporter.py
Generador de Informes Forenses en PDF Oficial de VeritasAI.
Crea reportes profesionales con veredictos, métricas estocásticas y mapa de calor.
Compatible con codificación segura Latin-1 / UTF-8.
"""

from datetime import datetime
from fpdf import FPDF
from typing import Dict, Any

def clean_latin1(text: str) -> str:
    """Reemplaza caracteres no soportados por las fuentes estándar de PDF."""
    if not text:
        return ""
    # Reemplazos específicos comunes
    text = text.replace("•", "-").replace("—", "-").replace("–", "-")
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = text.replace("…", "...").replace("«", '"').replace("»", '"')
    return text.encode('latin-1', 'replace').decode('latin-1')

class VeritasPDF(FPDF):
    def header(self):
        self.set_fill_color(15, 23, 42) # Slate 900
        self.rect(0, 0, 210, 24, 'F')
        
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(255, 255, 255)
        self.set_xy(12, 6)
        self.cell(0, 8, "VERITAS AI | INFORME FORENSE DE AUTORIA Y DETECCION", ln=False)
        
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(148, 163, 184)
        self.set_xy(12, 14)
        self.cell(0, 6, "Auditoria de Inteligencia Artificial (ChatGPT, Gemini, Claude, Perplexity, DeepSeek)", ln=False)
        self.ln(18)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(148, 163, 184)
        self.cell(0, 10, f"Pagina {self.page_no()}/{{nb}}  |  Generado por VeritasAI Pro v3.0  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="C")

def generate_pdf_report(analysis_result: Dict[str, Any], raw_text: str = "") -> bytes:
    """
    Genera un informe PDF profesional en bytes a partir del resultado del detector.
    """
    pdf = VeritasPDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=18)

    # 1. Resumen Ejecutivo / Tarjeta de Veredicto
    ai_prob = analysis_result.get("ai_probability", 0.0)
    human_prob = analysis_result.get("human_probability", 100.0)
    verdict = analysis_result.get("verdict", {})
    metrics = analysis_result.get("metrics", {})
    sentences = analysis_result.get("sentence_heatmap", [])
    summary = analysis_result.get("sentence_summary", {})

    # Caja del Veredicto
    if ai_prob >= 75.0:
        box_bg = (254, 242, 242)      # Rose light
        box_border = (239, 68, 68)    # Red
        text_color = (185, 28, 28)
    elif ai_prob >= 38.0:
        box_bg = (254, 252, 232)      # Amber light
        box_border = (245, 158, 11)   # Amber
        text_color = (180, 83, 9)
    else:
        box_bg = (240, 253, 244)      # Emerald light
        box_border = (16, 185, 129)   # Green
        text_color = (4, 120, 87)

    pdf.set_fill_color(*box_bg)
    pdf.set_draw_color(*box_border)
    pdf.set_line_width(0.5)
    pdf.rect(12, 28, 186, 32, 'FD')

    pdf.set_xy(16, 31)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 5, "VEREDICTO DE AUTORIA:", ln=True)

    pdf.set_x(16)
    pdf.set_font("Helvetica", "B", 15)
    pdf.set_text_color(*text_color)
    title_clean = clean_latin1(verdict.get('title', 'Evaluado').upper())
    pdf.cell(0, 8, f"{title_clean}  ({ai_prob}% IA / {human_prob}% HUMANO)", ln=True)

    pdf.set_x(16)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(51, 65, 85)
    desc = clean_latin1(verdict.get('description', ''))
    pdf.multi_cell(178, 4.5, desc)

    pdf.ln(8)

    # 2. Tabla de Indicadores Científicos
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 6, "METRICAS FORENSES Y ESTOCASTICAS", ln=True)
    pdf.ln(1)

    pdf.set_fill_color(241, 245, 249)
    pdf.set_draw_color(203, 213, 225)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(30, 41, 59)
    
    col_w = [46, 46, 47, 47]
    pdf.cell(col_w[0], 7, "Indicador", 1, 0, 'C', fill=True)
    pdf.cell(col_w[1], 7, "Valor Calculado", 1, 0, 'C', fill=True)
    pdf.cell(col_w[2], 7, "Interpretacion", 1, 0, 'C', fill=True)
    pdf.cell(col_w[3], 7, "Patron de Referencia", 1, 1, 'C', fill=True)

    pdf.set_font("Helvetica", "", 8.5)
    data_rows = [
        ("Cadencia (Burstiness)", f"{metrics.get('burstiness', {}).get('value', 0)}", f"{metrics.get('burstiness', {}).get('status', '--')}", "Baja (<0.40) = IA | Alta (>0.60) = Humano"),
        ("Perplejidad Lexica", f"{metrics.get('perplexity', {}).get('value', 0)}", f"{metrics.get('perplexity', {}).get('status', '--')}", "Baja = Predecible IA | Alta = Espontaneo"),
        ("Riqueza Lexica (TTR)", f"{metrics.get('lexical_diversity', {}).get('value', 0)}", f"{metrics.get('lexical_diversity', {}).get('unique_words', 0)} pal. unicas", "Vocabulario diversificado"),
        ("Cliches de IA Detectados", f"{metrics.get('ai_phrases_detected', {}).get('count', 0)} frases", "Patrones roboticos", "0 = Lenguaje fluido natural"),
        ("Firma de Modelo", f"{metrics.get('dominant_model', 'Neutral')}", "Atribucion de IA", "ChatGPT / Gemini / Claude / DeepSeek"),
    ]

    for row in data_rows:
        pdf.cell(col_w[0], 6.5, clean_latin1(f" {row[0]}"), 1, 0, 'L')
        pdf.cell(col_w[1], 6.5, clean_latin1(f" {row[1]}"), 1, 0, 'C')
        pdf.cell(col_w[2], 6.5, clean_latin1(f" {row[2]}"), 1, 0, 'C')
        pdf.cell(col_w[3], 6.5, clean_latin1(f" {row[3]}"), 1, 1, 'L')

    pdf.ln(6)

    # 3. Desglose Oración por Oración (Mapa de Calor)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 6, clean_latin1(f"ANALISIS POR ORACION ({summary.get('total', len(sentences))} oraciones evaluadas)"), ln=True)
    pdf.ln(1)

    pdf.set_font("Helvetica", "", 8)
    for idx, sent in enumerate(sentences[:45], 1):
        prob = sent.get("ai_probability", 0.0)
        label = sent.get("label", "safe")
        text = clean_latin1(sent.get("text", ""))
        reason = clean_latin1(sent.get("reason", ""))

        if label == "danger":
            tag_color = (220, 38, 38)
            tag_str = f"[{prob}% IA]"
        elif label == "warning":
            tag_color = (217, 119, 6)
            tag_str = f"[{prob}% HIB]"
        else:
            tag_color = (5, 150, 105)
            tag_str = f"[{prob}% HUM]"

        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*tag_color)
        pdf.cell(20, 5, tag_str, ln=False)

        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(30, 41, 59)
        pdf.multi_cell(166, 4.5, f"{text} ({reason})")
        pdf.ln(1)

    return bytes(pdf.output())
