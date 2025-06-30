import numpy as np
from collections import Counter
from scipy.spatial import distance
from collections import defaultdict


class KNN():
    
    "Método que realiza la clasificación KNN"
    def clasificador_KNN(self, entrenamiento, test, clase_train, clase_test, k):
        predicciones = []
        
        X_train = np.array(entrenamiento)
        X_test = np.array(test)
        y_train = np.array(clase_train)
        y_test = np.array(clase_test)
        i= 0
        for x_test in X_test:
            print(i)
            i = i + 1
            distancias = np.sum(np.abs(X_train - x_test), axis=1)
            
            vecinos_idx = np.argpartition(distancias, k)[:k]
            distancias_vecinos = distancias[vecinos_idx]
            
            pesos = 1.0 / (distancias_vecinos ** 2 + 1e-10)
            
            clases_vecinos = y_train[vecinos_idx]
            conteo = {}
            for cls, peso in zip(clases_vecinos, pesos):
                conteo[cls] = conteo.get(cls, 0) + peso
            
            pred = max(conteo.items(), key=lambda x: x[1])[0]
        
        accuracy = np.mean(np.array(predicciones) == y_test)
        print("KNN: ", accuracy)
        return accuracy
    
    
    "Método que realiza la clasificación Fuzzy KNN"
    def clasificador_fuzzy_knn(self,  entrenamiento, test, clase_train, clase_test,k, m):
        
        X_train = np.array(entrenamiento)
        X_test = np.array(test)
        y_train = np.array(clase_train)
        y_test = np.array(clase_test)
        
        
        clases = np.array(sorted(set(y_train)))
        n_clases = len(clases)
        n_train = len(y_train)
    
        # Pertenencias tipo 1.0 para clase real
        u_train = np.zeros((n_train, n_clases))
        for i, c in enumerate(y_train):
            u_train[i, np.where(clases == c)[0][0]] = 1
    
        predicciones = []
        pertenencias = []
        top1_correct = 0
        top2_correct = 0
        
    
        for i, x in enumerate(X_test):
            print(i)
            distancias = np.array([distance.cityblock(x, xi) for xi in X_train])
            distancias = np.maximum(distancias, 1e-8)
            vecinos_idx = np.argsort(distancias)[:k]
    
            pesos = 1.0 / (distancias[vecinos_idx] ** (2 / (m - 1)))
            denom = np.sum(pesos)
    
            u_x = {
                c: np.sum(u_train[vecinos_idx, j] * pesos) / denom
                for j, c in enumerate(clases)
            }
    
            # Ordenar clases por pertenencia descendente
            ordenadas = sorted(u_x.items(), key=lambda x: x[1], reverse=True)
            top1 = ordenadas[0][0]
            top2 = ordenadas[1][0]
    
            # Guardar predicción principal y pertenencias
            predicciones.append(top1)
            pertenencias.append(u_x)
    
            # Calcular accuracies
            if y_test[i] == top1:
                top1_correct += 1
            if y_test[i] in [top1, top2]:
                top2_correct += 1
    
        accuracy_top1 = top1_correct / len(X_test)
        accuracy_top2 = top2_correct / len(X_test)
        
        print("Fuzzy 1: ",accuracy_top1)
        print("Fuzzy 2: ",accuracy_top2)
    
        return accuracy_top1, accuracy_top2
        
    
    

        