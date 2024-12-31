from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from main import modelo_rf

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  

@app.route('/predecir', methods=['POST'])
def predecir():
    data = request.form
    nueva_propiedad = pd.DataFrame({
        'Superficie_m2': [float(data['superficie'])],
        'Dormitorios': [int(data['dormitorios'])],
        'Baños': [int(data['banos'])],
        'Garaje': [1 if data['garaje'] == 'Si' else 0],
        'Cercania_Escuelas_km': [float(data['escuelas'])],
        'Cercania_Hospitales_km': [float(data['hospitales'])],
        'Cercania_Teleferico_km': [float(data['teleferico'])],
        'Edad_Propiedad_años': [int(data['edad'])]
    })

    precio_predicho = modelo_rf.predecir(nueva_propiedad)[0]
    
    return render_template('index.html', prediccion=f"Precio Predicho: ${precio_predicho:.2f}")

if __name__ == '__main__':
    app.run(debug=True)
