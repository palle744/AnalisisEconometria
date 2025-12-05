@echo off
echo ==========================================
echo   Iniciando Plataforma de Econometria
echo ==========================================

echo Verificando Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python no esta instalado o no esta en el PATH.
    echo Por favor instala Python desde https://www.python.org/downloads/
    echo y asegurate de marcar "Add Python to PATH" durante la instalacion.
    pause
    exit /b
)

echo Instalando dependencias del Backend...
python -m pip install -r backend/requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Fallo la instalacion de dependencias.
    pause
    exit /b
)

echo Iniciando Servidor Backend...
echo La API estara disponible en http://localhost:8000/docs
cd backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

pause
