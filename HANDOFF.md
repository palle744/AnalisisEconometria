# Entrega del Proyecto: Plataforma de Econometría

## Resumen Ejecutivo
Se ha completado el desarrollo de la Plataforma Web de Econometría. La solución permite la ingestión de datos financieros reales, análisis técnico y fundamental, y la optimización de portafolios utilizando el modelo de Markowitz.

## Estado Final
- **Backend**: Python (FastAPI) funcionando correctamente.
- **Frontend**: Integrado en `backend/static` (HTML/JS puro).
- **Datos**: Conexión real a Yahoo Finance (yfinance).
- **Tests**: Pruebas unitarias pasando (2/2).

## Instrucciones de Uso

1.  **Ejecutar la Aplicación**:
    - Doble clic en `run_app.bat`.
    - Accede a `http://localhost:8000`.

2.  **Ingestión de Datos (Opcional)**:
    - Doble clic en `run_ingestion.bat` si deseas descargar CSVs localmente (aunque la app ya descarga en tiempo real).

## Solución de Problemas
- Si ves errores de módulos faltantes, ejecuta `run_app.bat` nuevamente, ya que intenta instalar dependencias al inicio.
- Si `frontend/` aparece en tu carpeta, puedes borrarla. La interfaz real está dentro de `backend/static`.

¡Listo para el análisis financiero!
