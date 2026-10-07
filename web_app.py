"""
web_app.py
Servidor Web FastAPI para VeritasAI Detector de Inteligencia Artificial.
Provee API REST e Interfaz Gráfica interactiva con mapa de calor.
"""

import os
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
from detector.engine import detector_engine
from detector.file_reader import extract_text_from_file
from detector.humanizer import humanize_text
from detector.pdf_reporter import generate_pdf_report

app = FastAPI(
    title="VeritasAI - Detector y Humanizador de IA",
    description="Detector avanzado y Humanizador inteligente de texto IA (ChatGPT, Gemini, Claude, Perplexity)",
    version="3.0.0"
)

# Servir archivos estáticos si existen
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

INDEX_HTML_PATH = os.path.join(os.path.dirname(__file__), "templates", "index.html")

class DetectRequest(BaseModel):
    text: str

class HumanizeRequest(BaseModel):
    text: str
    mode: Optional[str] = "stealth"
    api_key: Optional[str] = None

# Ejemplos precargados para pruebas inmediatas
SAMPLE_TEXTS = {
    "chatgpt": (
        "En un mundo cada vez más interconectado y digitalizado, la inteligencia artificial desempeña "
        "un papel fundamental en la evolución de nuestras sociedades. Desde la optimización de procesos "
        "industriales hasta la personalización de la educación, estas tecnologías están redefiniendo el panorama global. "
        "Cabe destacar que, si bien estos avances conllevan beneficios incuestionables, también suscitan interrogantes "
        "éticas de vital importancia respecto a la privacidad de los datos y el empleo. A su vez, es crucial fomentar "
        "un diálogo inclusivo y multidisciplinario para garantizar un desarrollo armonioso. "
        "En conclusión, el futuro dependerá de nuestra capacidad para aprovechar este potencial de manera responsable y ética."
    ),
    "gemini": (
        "Aquí tienes un desglose detallado sobre los principales pilares de la transición energética:\n\n"
        "1. Energías Renovables: La adopción masiva de energía solar y eólica resulta indispensable para reducir emisiones.\n"
        "2. Almacenamiento Eficiente: Las baterías de litio y nuevas tecnologías químicas juegan un papel crucial para equilibrar la red eléctrica.\n"
        "3. Eficiencia en el Consumo: Optimizar la demanda industrial y urbana permite maximizar cada gigavatio producido.\n\n"
        "En términos generales, lograr la neutralidad de carbono exige una inversión sostenida y políticas claras. "
        "En resumen, cada sector debe asumir compromisos concretos para acelerar este cambio."
    ),
    "human": (
        "El sábado pasado se me ocurrió ponerme a cambiar los parlantes viejos de la sala y terminé armando "
        "un desastre monumental. Resulta que los cables estaban todos resecos y pelados por detrás del mueble, "
        "así que en cuanto tiré un poco de uno se cortó al ras. Me pasé toda la tarde buscando el alicate pelacables "
        "que creí haber guardado en la caja de herramientas, pero mi hermano se lo había llevado sin avisarme. "
        "Al final tuve que pelar la punta con un cuchillo de cocina y me corté el pulgar. Una tarde perdida."
    )
}

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    if not os.path.exists(INDEX_HTML_PATH):
        raise HTTPException(status_code=404, detail="Plantilla index.html no encontrada.")
    with open(INDEX_HTML_PATH, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/api/detect")
async def detect_text(payload: DetectRequest):
    text = payload.text.strip()
    if not text:
        return JSONResponse(status_code=400, content={"error": "Por favor proporciona un texto."})
    
    result = detector_engine.detect(text)
    return JSONResponse(content=result)

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    contents = await file.read()
    success, text_or_error = extract_text_from_file(file.filename, contents)
    
    if not success:
        return JSONResponse(status_code=400, content={"error": text_or_error})
    
    result = detector_engine.detect(text_or_error)
    return JSONResponse(content={
        "filename": file.filename,
        "extracted_text": text_or_error,
        "analysis": result
    })

@app.post("/api/humanize")
async def humanize_endpoint(payload: HumanizeRequest):
    text = payload.text.strip()
    if not text:
        return JSONResponse(status_code=400, content={"error": "Por favor proporciona un texto para humanizar."})
    
    result = humanize_text(text, mode=payload.mode or "stealth", api_key=payload.api_key)
    return JSONResponse(content=result)

@app.post("/api/export-pdf")
async def export_pdf(payload: DetectRequest):
    text = payload.text.strip()
    if not text:
        return JSONResponse(status_code=400, content={"error": "Por favor proporciona un texto."})
    
    result = detector_engine.detect(text)
    pdf_bytes = generate_pdf_report(result, text)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=reporte_forense_veritas.pdf"}
    )

@app.get("/api/sample/{sample_type}")
async def get_sample(sample_type: str):
    sample = SAMPLE_TEXTS.get(sample_type.lower())
    if not sample:
        return JSONResponse(status_code=404, content={"error": "Tipo de ejemplo desconocido."})
    return JSONResponse(content={"sample": sample, "type": sample_type})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    print("\n" + "="*60)
    print(" 🚀 INICIANDO VERITAS AI DETECTOR DE INTELIGENCIA ARTIFICIAL")
    print(f" 🌐 Servidor disponible en: http://{host}:{port}")
    print("="*60 + "\n")
    uvicorn.run("web_app:app", host=host, port=port, reload=False)
