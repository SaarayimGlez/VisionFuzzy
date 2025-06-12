import numpy as np
from collections import Counter
from scipy.spatial import distance
from collections import defaultdict


class KNN():
    
    def clasificador_(self, entrenamiento, test, clase, k):
        predicciones = []
        
        X_train = np.array(entrenamiento)
        X_test = np.array(test)
        y_train = np.array(clase)
        
        for x_test in X_test:
            distancias = np.linalg.norm(X_train - x_test, axis=1)

            vecinos_idx = np.argsort(distancias)[:k]

            vecinos_clase = y_train[vecinos_idx]

            pred = Counter(vecinos_clase).most_common(1)[0][0]
            predicciones.append(pred)

        return predicciones
    
    
    
    def clasificador_fuzzy_knn(self, X_train, y_train, X_test, k=5, m=2):
        clases = np.unique(y_train)
        n_clases = len(clases)
        n_train = len(y_train)

        u_train = np.zeros((n_train, n_clases))
        for i, c in enumerate(y_train):
            u_train[i, np.where(clases == c)] = 1
    
        predicciones = []
        pertenencias = []
    
        for x in X_test:
            distancias = np.array([distance.euclidean(x, xi) for xi in X_train])
            distancias = np.maximum(distancias, 1e-8)
            vecinos_idx = np.argsort(distancias)[:k]
            
            pesos = 1.0 / (distancias[vecinos_idx] ** (2 / (m - 1)))
            denom = np.sum(pesos)
            
            u_x = {
                c: np.sum(u_train[vecinos_idx, j] * pesos) / denom
                for j, c in enumerate(clases)
            }
            
            pred_clase = max(u_x, key=u_x.get)
            predicciones.append(pred_clase)
            pertenencias.append(u_x)
    
        return predicciones, pertenencias
    
    
    

        