@echo off
echo ==========================================
echo   Descarga de Datos Financieros
echo ==========================================

echo Instalando dependencias...
python -m pip install -r backend/requirements.txt

echo.
set /p TICKERS="Ingresa los tickers separados por coma (ej. META,AVGO,GLD): "
set /p START_DATE="Fecha de inicio (YYYY-MM-DD, ej. 2020-01-01): "

echo.
echo Descargando datos...
cd backend
python scripts/fetch_prices.py --tickers %TICKERS% --from_date %START_DATE%

echo.
echo Proceso finalizado.
pause
