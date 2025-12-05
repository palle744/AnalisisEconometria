# Plataforma de Econometría

Esta es una plataforma web para el análisis econométrico y optimización de portafolios utilizando el modelo de Markowitz.

## Características

- Optimización de portafolios (Frontera Eficiente de Markowitz)
- Análisis de portafolios manuales
- Descarga de reportes en PDF y Excel
- Análisis fundamental y técnico
- Noticias financieras en tiempo real

## Ejecución con Docker

Para ejecutar la aplicación utilizando Docker, asegúrate de tener Docker y Docker Compose instalados en tu sistema.

1.  Construir y levantar el contenedor:

    ```bash
    docker-compose up --build
    ```

2.  Acceder a la aplicación en tu navegador:

    ```
    http://localhost:8000
    ```

## Ejecución Local (Sin Docker)

1.  Crear un entorno virtual e instalar dependencias:

    ```bash
    cd backend
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

2.  Ejecutar el servidor:

    ```bash
    uvicorn app.main:app --reload
    ```

## Estructura del Proyecto

- `backend/`: Código fuente del backend (FastAPI) y frontend estático.
- `docker-compose.yml`: Configuración de Docker Compose.
