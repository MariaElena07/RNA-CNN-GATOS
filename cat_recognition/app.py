"""
app.py — Servidor Flask para reconocimiento de razas de gatos
Ubicación: cat_recognition/app.py

Cómo correr:
    (venv) > cd cat_recognition
    (venv) > python app.py
    Luego abre: http://localhost:5000
"""

import os
import json
import base64
import datetime
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image
import io
import tensorflow as tf
import cv2

# ── Configuración ──────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join("models/best_model.keras", compile=False)
NAMES_PATH    = os.path.join(BASE_DIR, 'models', 'class_names.npy')
HISTORIAL_PATH= os.path.join(BASE_DIR, 'historial', 'historial.json')
IMG_SIZE      = 224
UMBRAL        = 0.70   # Por debajo de esto → mestizo

app = Flask(__name__)

# ── Cargar modelo y clases al iniciar ──────────────────────────
print("Cargando modelo...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
class_names = np.load(NAMES_PATH, allow_pickle=True).tolist()
print(f"Modelo listo — {len(class_names)} razas: {class_names}")

# Detector de gatos con Haar Cascades (incluido en OpenCV)
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalcatface.xml'
cat_detector = cv2.CascadeClassifier(CASCADE_PATH)


# ── Funciones auxiliares ───────────────────────────────────────

def preprocesar_imagen(img_pil):
    """Convierte una imagen PIL al tensor que espera el modelo."""
    img = img_pil.convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def predecir(imagen_pil):
    """
    Predice la raza del gato en la imagen.
    - Si confianza >= UMBRAL → devuelve raza
    - Si confianza < UMBRAL  → devuelve perfil de mezcla
    """
    tensor = preprocesar_imagen(imagen_pil)
    probs  = model.predict(tensor, verbose=0)[0]

    idx_top    = int(np.argmax(probs))
    conf_top   = float(probs[idx_top])

    # Top 3 razas siempre
    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = [
        {'raza': class_names[i], 'confianza': round(float(probs[i]) * 100, 1)}
        for i in top3_idx
    ]

    # Todas las razas para el gráfico de barras
    todas = [
        {'raza': class_names[i], 'confianza': round(float(probs[i]) * 100, 1)}
        for i in np.argsort(probs)[::-1]
    ]

    if conf_top >= UMBRAL:
        return {
            'tipo':      'raza_pura',
            'raza':      class_names[idx_top],
            'confianza': round(conf_top * 100, 1),
            'top3':      top3,
            'todas':     todas
        }
    else:
        return {
            'tipo':      'mestizo',
            'raza':      'Mestizo',
            'confianza': round(conf_top * 100, 1),
            'nota':      f"Podría tener rasgos de {top3[0]['raza']} y {top3[1]['raza']}",
            'top3':      top3,
            'todas':     todas
        }


def guardar_historial(resultado, imagen_b64):
    """Guarda la predicción en historial.json."""
    historial = []
    if os.path.exists(HISTORIAL_PATH):
        with open(HISTORIAL_PATH, 'r') as f:
            historial = json.load(f)

    entrada = {
        'id':         len(historial) + 1,
        'fecha':      datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'raza':       resultado['raza'],
        'confianza':  resultado['confianza'],
        'tipo':       resultado['tipo'],
        'top3':       resultado['top3'],
        'imagen_b64': imagen_b64[:100] + '...'  # solo guardamos preview
    }
    historial.insert(0, entrada)
    historial = historial[:50]  # máximo 50 entradas

    with open(HISTORIAL_PATH, 'w') as f:
        json.dump(historial, f, ensure_ascii=False, indent=2)

    return entrada


def detectar_gato_opencv(imagen_pil):
    """Detecta si hay un gato en la imagen con Haar Cascades."""
    img_cv  = cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)
    gris    = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    caras   = cat_detector.detectMultiScale(gris, scaleFactor=1.1,
                                             minNeighbors=5, minSize=(50, 50))
    return len(caras) > 0, caras


# ── Rutas Flask ────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html', clases=class_names)


@app.route('/predecir', methods=['POST'])
def ruta_predecir():
    """Recibe una imagen, predice la raza y devuelve JSON."""
    if 'imagen' not in request.files:
        return jsonify({'error': 'No se recibió ninguna imagen'}), 400

    archivo = request.files['imagen']
    if archivo.filename == '':
        return jsonify({'error': 'Archivo vacío'}), 400

    try:
        img_bytes = archivo.read()
        imagen_pil = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # Detectar si hay gato (opcional, no bloquea)
        hay_gato, boxes = detectar_gato_opencv(imagen_pil)

        # Si detectó gato, recortar esa zona para mejor precisión
        if hay_gato and len(boxes) > 0:
            x, y, w, h = boxes[0]
            margen = 20
            x0 = max(0, x - margen)
            y0 = max(0, y - margen)
            x1 = min(imagen_pil.width,  x + w + margen)
            y1 = min(imagen_pil.height, y + h + margen)
            imagen_pil = imagen_pil.crop((x0, y0, x1, y1))

        resultado = predecir(imagen_pil)
        resultado['gato_detectado'] = bool(hay_gato)

        # Imagen en base64 para mostrar en la web
        buffer   = io.BytesIO()
        imagen_pil.save(buffer, format='JPEG', quality=85)
        img_b64  = base64.b64encode(buffer.getvalue()).decode('utf-8')

        guardar_historial(resultado, img_b64)

        return jsonify({**resultado, 'imagen_b64': img_b64})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/historial')
def ruta_historial():
    """Devuelve el historial de predicciones."""
    if not os.path.exists(HISTORIAL_PATH):
        return jsonify([])
    with open(HISTORIAL_PATH, 'r') as f:
        return jsonify(json.load(f))


@app.route('/camara_predecir', methods=['POST'])
def ruta_camara():
    """Recibe un frame de la cámara en base64 y predice."""
    data = request.get_json()
    if not data or 'frame' not in data:
        return jsonify({'error': 'No se recibió frame'}), 400

    try:
        # Decodificar base64
        img_data   = base64.b64decode(data['frame'].split(',')[1])
        imagen_pil = Image.open(io.BytesIO(img_data)).convert('RGB')

        hay_gato, boxes = detectar_gato_opencv(imagen_pil)

        if hay_gato and len(boxes) > 0:
            x, y, w, h = boxes[0]
            margen = 20
            x0 = max(0, x - margen)
            y0 = max(0, y - margen)
            x1 = min(imagen_pil.width,  x + w + margen)
            y1 = min(imagen_pil.height, y + h + margen)
            imagen_recortada = imagen_pil.crop((x0, y0, x1, y1))
            resultado = predecir(imagen_recortada)
            resultado['box'] = {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}
        else:
            resultado = {'tipo': 'sin_gato', 'raza': None, 'confianza': 0, 'top3': []}

        resultado['gato_detectado'] = bool(hay_gato)
        return jsonify(resultado)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/clases')
def ruta_clases():
    """Devuelve las clases disponibles."""
    return jsonify(class_names)


# ── Iniciar servidor ───────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*50)
    print("  Cat Recognizer — servidor iniciado")
    print("  Abre tu navegador en: http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
