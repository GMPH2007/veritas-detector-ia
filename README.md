# 🧠 VeritasAI Pro - Suite Avanzada de Detección & Humanización de IA

**VeritasAI** es una plataforma de vanguardia con doble función integrada:
1. **🔍 Detector de Inteligencia Artificial Multicapa**: Detecta con precisión textos generados por **ChatGPT (GPT-4 / GPT-4o), Google Gemini, Claude y Perplexity**, blindado además contra intentos de camuflaje o "humanización superficial".
2. **✨ Humanizador de Texto Antidetección**: Transforma cualquier texto de IA en prosa auténticamente humana (inyectando cadencia, burstiness orgánica y eliminando clichés) con autoverificación en tiempo real para garantizar un resultado indetectable (**< 20% de IA**).

---

## 🚀 Inicio Rápido (1 Clic)

Simplemente haz doble clic en cualquiera de los siguientes archivos en la carpeta:

1. **`iniciar.bat`**: Menú interactivo principal (Elige Web o Escritorio).
2. **`iniciar_detector_web.bat`**: Abre la interfaz web en tu navegador (`http://127.0.0.1:8000`).
3. **`iniciar_detector_escritorio.bat`**: Abre la aplicación de ventana nativa para Windows (CustomTkinter en Modo Oscuro).

---

## 🌟 Las Dos Herramientas Integradas

### 1. 🔍 Detector de IA con Red Neuronal y Mapa de Calor
- **Red Neuronal Antievasión (MLP + Gradient Boosting)**: Entrenada no solo con textos puros de IA y humanos, sino también con textos parafraseados con "humanizadores" tradicionales para evitar que pasen desapercibidos.
- **Mapa de Calor Oración por Oración**: Resalta en rojo, amarillo y verde cada frase con su probabilidad individual y motivos del diagnóstico.
- **Métricas Científicas en Vivo**: Perplejidad estocástica, Burstiness (coeficiente de variación), Riqueza Léxica (TTR) y detección de firmas de laboratorio.
- **Soporte de Documentos**: Analiza directamente archivos `.pdf`, `.docx` y `.txt`.

### 2. ✨ Humanizador de Texto Antidetección
- **Purga Total de Clichés**: Erradica expresiones robóticas como *"cabe destacar"*, *"desempeña un papel fundamental"*, *"en un mundo cada vez más..."*, *"delve into"*, *"a su vez"*, *"en conclusión"*.
- **Inyección de Burstiness Orgánica**: Rompe la monotonía métrica de los LLMs alternando micro-oraciones de 3 a 5 palabras con pensamientos extensos y preguntas retóricas vivas.
- **3 Modos de Humanización**:
  - 🔥 **Stealth (100% Antidetección)**: Máxima reestructuración para eludir Turnitin, GPTZero y VeritasAI.
  - 🎓 **Académico Natural**: Conserva rigor formal y vocabulario culto sin sonar a robot.
  - 💬 **Conversacional / Cercano**: Ideal para artículos, correos, blogs y redes sociales.
- **Bucle de Autoverificación Cerrado**: Al pulsar *"Humanizar"*, el sistema verifica automáticamente el resultado en el detector y te muestra la comparativa (*Antes: 99% IA ➔ Ahora: 18% IA*).
- **Botón "⚡ Probar en el Detector"**: Transfiere el texto humanizado al detector para comprobar en vivo cómo cae al verde seguro.
- **Modo Opcional con Groq (Llama-3.3 70B)**: Si dispones de una clave API de Groq gratuita, puedes activar la reescritura neuronal ultra-rápida.

### 1. 🧬 Red Neuronal y Ensamble Calibrado (MLP + Gradient Boosting)
- Clasificador neuronal entrenado con vectores de más de 20 dimensiones estadísticas y lingüísticas.
- Emplea activación no lineal ReLU y ensamble con ponderación suave (*soft voting*) para arrojar probabilidades calibradas entre 0% y 100%.

### 2. ⚡ Análisis de Perplejidad y Entropía Estocástica
- Los modelos de IA eligen sistemáticamente las palabras con mayor probabilidad estadística (baja perplejidad y baja entropía de Shannon).
- Los autores humanos eligen giros creativos, anécdotas, irregularidades y vocabulario atípico (alta perplejidad).

### 3. 🌊 Burstiness (Variación Rítmica de Oraciones)
- Mide el coeficiente de variación ($\sigma / \mu$) de la longitud de las oraciones.
- **Texto IA**: Presenta longitud simétrica y constante entre 18 y 30 palabras por oración (burstiness baja $< 0.35$).
- **Texto Humano**: Combina oraciones de 4 palabras con pensamientos largos de 45 palabras (burstiness alta $> 0.65$).

### 4. 🕵️ Detección de Firmas Lingüísticas y Clichés de Modelos
Identifica los sesgos característicos de cada laboratorio:
- **ChatGPT / OpenAI**: *"En un mundo cada vez más...", "desempeña un papel fundamental", "cabe destacar", "a su vez", "en conclusión", "delve into", "rich tapestry", "a testament to"*.
- **Google Gemini**: *"Aquí tienes un desglose", "puntos clave:", "en términos generales", "here's a breakdown"*.
- **Claude (Anthropic)**: *"Es importante reconocer que", "una perspectiva matizada", "nuanced perspective"*.
- **Perplexity**: Estilo de síntesis enciclopédica neutral con citas factuales condensadas.

### 5. 🎯 Mapa de Calor Interactivo Oración por Oración (Sentence Heatmap)
Cada oración es aislada y evaluada individualmente:
- 🟥 **Rojo (75-100%)**: Alta probabilidad de IA (muy predecible o con conectores de LLM).
- 🟧 **Amarillo (45-74%)**: Posible asistencia, reescritura o lenguaje híbrido.
- 🟩 **Verde (0-44%)**: Cadencia y léxico típicamente humano.
- *Al hacer clic en cualquier oración en la web, se despliega una ventana con la justificación del veredicto.*

---

## 📂 Formatos de Archivo Soportados
Puedes arrastrar o subir directamente:
- Archivos de texto plano (**`.txt`**)
- Documentos de Microsoft Word (**`.docx`**, **`.doc`**)
- Documentos PDF (**`.pdf`**)

---

## 📊 Comparativa de Diagnóstico

| Característica | Texto Generado por IA (ChatGPT / Gemini) | Texto Escrito por Humano |
| :--- | :--- | :--- |
| **Burstiness** | Muy baja (cadencia monótona y simétrica) | Alta (picos y valles de extensión) |
| **Perplejidad** | Baja (palabras altamente predecibles) | Alta / Variable (giros inesperados) |
| **Conectores** | Exceso de transiciones formales (*cabe destacar, a su vez*) | Conectores espontáneos o coloquiales |
| **Estructura** | Introducción, listas balanceadas, conclusión fija | Flujo narrativo irregular y orgánico |

---

## 💻 Requisitos
- Python 3.10 o superior (compatible con Python 3.11 instalado en tu equipo).
- Librerías: `fastapi`, `uvicorn`, `scikit-learn`, `numpy`, `customtkinter`, `pymupdf` / `pypdf`, `python-docx`. (Todas ya configuradas en el entorno).
