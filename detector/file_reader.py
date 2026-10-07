"""
detector/file_reader.py
Módulo para extraer texto desde archivos subidos: .txt, .pdf, .docx
"""

import io
from typing import Tuple

def extract_text_from_file(filename: str, file_bytes: bytes) -> Tuple[bool, str]:
    """
    Extrae texto limpio desde bytes según la extensión del archivo.
    Retorna (éxito, texto_o_error).
    """
    ext = filename.lower().split('.')[-1] if '.' in filename else ''
    
    try:
        if ext == 'txt':
            # Intentar decodificar en utf-8 o latin-1
            try:
                text = file_bytes.decode('utf-8')
            except UnicodeDecodeError:
                text = file_bytes.decode('latin-1', errors='ignore')
            return True, text.strip()

        elif ext == 'pdf':
            # Intentar primero con pymupdf (fitz) si está disponible, o con pypdf
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                pages = [page.get_text() for page in doc]
                text = "\n".join(pages).strip()
                if text:
                    return True, text
            except Exception:
                pass

            # Fallback a pypdf
            try:
                import pypdf
                reader = pypdf.PdfReader(io.BytesIO(file_bytes))
                pages = [page.extract_text() or '' for page in reader.pages]
                text = "\n".join(pages).strip()
                if text:
                    return True, text
            except Exception as e:
                return False, f"No se pudo leer el archivo PDF: {str(e)}"

            return False, "El archivo PDF no contiene texto legible o es un documento escaneado."

        elif ext in ('docx', 'doc'):
            try:
                import docx
                doc = docx.Document(io.BytesIO(file_bytes))
                paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
                text = "\n".join(paragraphs).strip()
                return True, text
            except Exception as e:
                return False, f"No se pudo leer el archivo Word (.docx): {str(e)}"

        else:
            return False, f"Formato no soportado (. {ext}). Formatos permitidos: .txt, .pdf, .docx"

    except Exception as e:
        return False, f"Error al procesar el archivo: {str(e)}"
