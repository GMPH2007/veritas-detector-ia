@echo off
title VeritasAI - Detector de Inteligencia Artificial
color 0B
:menu
cls
echo ======================================================================
echo       VERITAS AI - DETECTOR Y HUMANIZADOR DE TEXTO DE IA
echo     (ChatGPT 4o/o1, Gemini, Claude, Perplexity, DeepSeek y Humanos)
echo ======================================================================
echo.
echo   Selecciona el modo en el que deseas iniciar:
echo.
echo     [1] Interfaz Web Moderna (Recomendado - Abre en tu Navegador con Mapa de Calor)
echo     [2] Aplicacion de Escritorio Nativa (Ventana Windows)
echo     [3] Salir
echo.
echo ======================================================================
set /p opcion="Elige una opcion (1, 2 o 3): "

if "%opcion%"=="1" goto web
if "%opcion%"=="2" goto desktop
if "%opcion%"=="3" goto salir

echo Opcion invalida. Intenta nuevamente.
timeout /t 2 >nul
goto menu

:web
cls
echo Iniciando VeritasAI Web en http://127.0.0.1:8000 ...
start "" "http://127.0.0.1:8000"
python web_app.py
pause
goto menu

:desktop
cls
echo Iniciando VeritasAI Escritorio...
python desktop_app.py
pause
goto menu

:salir
exit
