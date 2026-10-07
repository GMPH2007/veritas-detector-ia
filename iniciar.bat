@echo off
title VeritasAI - Detector y Humanizador de Inteligencia Artificial
color 0B
:menu
cls
echo ======================================================================
echo       VERITAS AI - DETECTOR Y HUMANIZADOR DE TEXTO DE IA
echo     (ChatGPT 4o/o1, Gemini, Claude, Perplexity, DeepSeek y Humanos)
echo ======================================================================
echo.
echo   Selecciona una opcion:
echo.
echo     [1] Interfaz Web Local (Abre en tu Navegador http://127.0.0.1:8000)
echo     [2] Aplicacion de Escritorio Nativa (Ventana Windows)
echo     [3] Abrir con Enlace Publico en Internet (Para Celular o Amigos)
echo     [4] Subir a Google Cloud Run (Servidor en la Nube de Google)
echo     [5] Salir
echo.
echo ======================================================================
set /p opcion="Elige una opcion (1, 2, 3, 4 o 5): "

if "%opcion%"=="1" goto web
if "%opcion%"=="2" goto desktop
if "%opcion%"=="3" goto publico
if "%opcion%"=="4" goto cloud
if "%opcion%"=="5" goto salir

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

:publico
cls
call iniciar_con_enlace_publico_internet.bat
goto menu

:cloud
cls
call subir_a_google_cloud.bat
goto menu

:salir
exit
