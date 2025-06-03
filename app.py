from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from collections import Counter
import pickle
import os

app = Flask(__name__)

# ========== IMPLEMENTASI KNN ALGORITHM ==========
class KNNClassifier:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def euclidean_distance(self, point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))
    
    def predict_single(self, test_point):
        distances = []
        
        # Hitung jarak ke semua titik training
        for i in range(len(self.X_train)):
            dist = self.euclidean_distance(test_point, self.X_train[i])
            distances.append((dist, self.y_train[i]))
        
        # Sorting ascending berdasarkan jarak
        distances.sort(key=lambda x: x[0])

        # Ambil k tetangga terdekat
        k_nearest = distances[:self.k]

        # Ambil label dari k tetangga terdekat
        k_labels = [label for _, label in k_nearest]
        
        # Voting majority untuk prediksi
        prediction = Counter(k_labels).most_common(1)[0][0]

        # Hitung confidence score
        confidence = Counter(k_labels).most_common(1)[0][1] / self.k
        
        return prediction, confidence, k_nearest
    
    def predict(self, X_test):
        if len(X_test.shape) == 1:
            X_test = X_test.reshape(1, -1)
        
        predictions = []
        confidences = []
        prediction_details = []
        
        for i, test_point in enumerate(X_test):
            pred, conf, k_nearest = self.predict_single(test_point)
            predictions.append(pred)
            confidences.append(conf)
            prediction_details.append({
                'test_index': i,
                'prediction': pred,
                'confidence': conf,
                'k_nearest_neighbors': k_nearest
            })
        
        return np.array(predictions), confidences, prediction_details

# ========== Dataset Processing ==========
# kategori pekerjaan menjdi 3
def categorize_job(job):
    formal = ['PNS', 'TNI/POLRI', 'Guru']
    informal = ['Buruh', 'Petani', 'Nelayan', 'Wiraswasta']
    if job in formal:
        return 'Formal'
    elif job in informal:
        return 'Informal'
    else:
        return 'Tidak_Bekerja'

def preprocess_data():
    # membaca dataset
    data = pd.read_csv("C:/Users/mbak/Documents/GitHub/algoritma-knn-kelayakan-ukt/klasifikasimhs_dataset.csv")
    
    # proses normalisasi (normalisasi data dan one hot encoding)
    data_normalized = data.copy()
    
    # One hot encoding
    data_normalized['Kategori_Pekerjaan'] = data_normalized['Pekerjaan Orang Tua'].apply(categorize_job)
    data_normalized = pd.get_dummies(data_normalized, columns=['Kategori_Pekerjaan'], prefix='Pekerjaan', dtype=int)
    
    # normalisasi data
    scaler = MinMaxScaler()
    cols_to_normalize = ['Tempat Tinggal', 'Penghasilan Orang Tua', 'Jumlah Tanggungan Orang Tua', 'Kendaraan']
    data_normalized[cols_to_normalize] = scaler.fit_transform(data_normalized[cols_to_normalize])
    
    return data, data_normalized, scaler

def setup_model():
    data, data_normalized, scaler = preprocess_data()
    
    # proses split data dan smoye
    feature_cols = ['Tempat Tinggal', 'Penghasilan Orang Tua', 'Jumlah Tanggungan Orang Tua', 'Kendaraan', 
                   'Pekerjaan_Formal', 'Pekerjaan_Informal', 'Pekerjaan_Tidak_Bekerja']
    target_col = 'Kelayakan Keringanan UKT'
    
    # split data
    X = data_normalized[feature_cols]
    y = data_normalized[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # smote
    smote = SMOTE(random_state=42)
    X_train_processed, y_train_processed = smote.fit_resample(X_train, y_train)
    
    # Inisialisasi KNN dengan k=5
    k_value = 5
    knn = KNNClassifier(k=k_value)
    knn.fit(X_train_processed.values, y_train_processed.values)
    
    return knn, scaler, data, feature_cols

#initialize model dan scaler
MODEL, SCALER, ORIGINAL_DATA, FEATURE_COLS = setup_model()

# ========== INPUT DATA USER ==========
def format_currency(amount):
    #format angka menjdi rupiah
    return f"Rp {amount:,.0f}".replace(',', '.')

def preprocess_input(tempat_tinggal, pekerjaan, penghasilan, tanggungan, kendaraan):
    input_data = pd.DataFrame({
        'Tempat Tinggal': [1 if tempat_tinggal == 'Punya' else 0],
        'Penghasilan Orang Tua': [float(penghasilan)],
        'Jumlah Tanggungan Orang Tua': [int(tanggungan)],
        'Kendaraan': [int(kendaraan)]
    })
    
    # kategorisasi pekerjaan
    kategori = categorize_job(pekerjaan)
    input_data['Pekerjaan_Formal'] = 1 if kategori == 'Formal' else 0
    input_data['Pekerjaan_Informal'] = 1 if kategori == 'Informal' else 0
    input_data['Pekerjaan_Tidak_Bekerja'] = 1 if kategori == 'Tidak_Bekerja' else 0
    
    # normalisasi kolom numerik input user menggunakan scaler yang sudah di fit ke data training
    cols_to_normalize = ['Tempat Tinggal', 'Penghasilan Orang Tua', 'Jumlah Tanggungan Orang Tua', 'Kendaraan']
    input_data[cols_to_normalize] = SCALER.transform(input_data[cols_to_normalize])
    
    return input_data[FEATURE_COLS].values[0]

# ========== ROUTES ==========
# tampilan awal/home
@app.route('/')
def index():
    return render_template('index.html')

# tampilan dataset
@app.route('/dataset')
def dataset():
    data_dict = ORIGINAL_DATA.to_dict('records')
    stats = {
        'total_data': len(ORIGINAL_DATA),
        'layak': len(ORIGINAL_DATA[ORIGINAL_DATA['Kelayakan Keringanan UKT'] == 1]),
        'tidak_layak': len(ORIGINAL_DATA[ORIGINAL_DATA['Kelayakan Keringanan UKT'] == 0]),
    }
    return render_template('dataset.html', data=data_dict, stats=stats)

# tampilan kelompok 6
@app.route('/about')
def about():
    return render_template('about.html')

# route untuk prediks dariinput user dan akan ditampilakdi halaman result
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ambil input dari form
        tempat_tinggal = request.form['tempat_tinggal']
        pekerjaan = request.form['pekerjaan']
        penghasilan = request.form['penghasilan']
        tanggungan = request.form['tanggungan']
        kendaraan = request.form['kendaraan']
        
        #validasi input
        if not all([tempat_tinggal, pekerjaan, penghasilan, tanggungan, kendaraan]):
            return jsonify({'error': 'Semua field harus diisi'}), 400
        
        #preprocess input
        processed_input = preprocess_input(tempat_tinggal, pekerjaan, penghasilan, tanggungan, kendaraan)
        
        #prediksi
        predictions, confidences, details = MODEL.predict(processed_input.reshape(1, -1))
        
        prediction = predictions[0]
        confidence = confidences[0]
        k_nearest = details[0]['k_nearest_neighbors']
        
        #hasil
        result = {
            'input_data': {
                'tempat_tinggal': tempat_tinggal,
                'pekerjaan': pekerjaan,
                'penghasilan': format_currency(int(penghasilan)),
                'tanggungan': tanggungan,
                'kendaraan': kendaraan
            },
            'prediction': 'Layak' if prediction == 1 else 'Tidak Layak',
            'prediction_value': int(prediction),
            'confidence': f"{confidence * 100:.1f}%",
            'confidence_value': confidence,
            'k_nearest_neighbors': [
                {
                    'distance': f"{dist:.4f}",
                    'label': 'Layak' if label == 1 else 'Tidak Layak',
                    'label_value': int(label)
                }
                for dist, label in k_nearest
            ],
        }
        
        return render_template('result.html', result=result)
        
    except Exception as e:
        return jsonify({'error': f'Terjadi kesalahan: {str(e)}'}), 500

# api untuk get data dataset, ringkasan statik di halaman dataset
@app.route('/api/stats')
def api_stats():
    stats = {
        'total_data': len(ORIGINAL_DATA),
        'layak': int(len(ORIGINAL_DATA[ORIGINAL_DATA['Kelayakan Keringanan UKT'] == 1])),
        'tidak_layak': int(len(ORIGINAL_DATA[ORIGINAL_DATA['Kelayakan Keringanan UKT'] == 0])),
    }
    return jsonify(stats)

if __name__ == '__main__':
    app.run(debug=True)