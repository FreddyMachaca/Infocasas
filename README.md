# Infocasas — Predicción de Precios de Propiedades

Aplicación web desarrollada para estimar el precio de inmuebles a partir de variables estructurales y de ubicación. El proyecto integra una interfaz en Flask con un modelo de Bosque Aleatorio de regresión implementado manualmente.

## Resumen técnico

- Backend web con Flask para captura de datos y despliegue de predicción en tiempo real.
- Modelo de aprendizaje automático implementado desde cero en Python (árbol de decisión + ensamble tipo Random Forest).
- Sin uso de librerías de *machine learning* (por ejemplo, scikit-learn) para el entrenamiento o la inferencia del modelo.
- Preprocesamiento y manejo de datos con pandas y numpy.
- Dataset estructurado con variables de superficie, dormitorios, baños, garaje, cercanía a servicios y antigüedad.

## Arquitectura del proyecto

- `app.py`: servidor Flask y rutas (`/` y `/predecir`) para recibir entradas del formulario y devolver el precio estimado.
- `main.py`: lógica de entrenamiento, definición de `ArbolDecision` y `BosqueAleatorio`, y función de predicción.
- `templates/index.html`: interfaz principal del sistema.
- `static/styles.css`: estilos de la vista.
- `dataset.csv`: muestra de datos utilizados para entrenar y evaluar el modelo.

## Enfoque del algoritmo

El modelo fue construido manualmente para demostrar dominio de fundamentos de *machine learning*:

- División de datos en entrenamiento y prueba con función `train_test_split` propia.
- Construcción recursiva del árbol de decisión para regresión.
- Criterio de partición basado en reducción de varianza (ganancia de información).
- Muestreo *bootstrap* por árbol para generar diversidad en el ensamble.
- Agregación final por promedio de predicciones de múltiples árboles.

## Tecnologías utilizadas

- Python
- Flask
- pandas
- numpy
- HTML y CSS

## Ejecución local

1. Instalar dependencias:
	- `pip install flask pandas numpy`
2. Ejecutar la aplicación:
	- `python app.py`
3. Abrir en navegador:
	- `http://127.0.0.1:5000`