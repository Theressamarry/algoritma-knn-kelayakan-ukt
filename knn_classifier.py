import numpy as np
from collections import Counter

class KNNClassifier:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X_train, y_train):
        """Menyimpan data training"""
        self.X_train = X_train
        self.y_train = y_train
    
    def euclidean_distance(self, point1, point2):
        """Menghitung jarak Euclidean antara dua titik"""
        return np.sqrt(np.sum((point1 - point2) ** 2))
    
    def predict_single(self, test_point):
        """Prediksi untuk satu data testing"""
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
        
        return prediction, k_nearest
    
    def predict(self, X_test):
        """Prediksi untuk semua data testing"""
        predictions = []
        prediction_details = []
        
        for i, test_point in enumerate(X_test):
            pred, k_nearest = self.predict_single(test_point)
            predictions.append(pred)
            prediction_details.append({
                'test_index': i,
                'prediction': pred,
                'k_nearest_neighbors': k_nearest
            })
        
        return np.array(predictions), prediction_details
