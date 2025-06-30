import h5py
import numpy as np
import glob
import os
from KNN import KNN
from scipy.io import savemat

class ProcesoDatos():
    
    
    "Método que obtiene todos los conjuntos de datos cuyos nombres tengan el mismo patrón"
    def obtener_dataset(self, ruta, patron):
        archivos = glob.glob(os.path.join(ruta, patron))
        
        dataset_completo = []
        for archivo in archivos:
            mat_file = h5py.File(archivo, 'r')
            grupo_dataset = mat_file['dataset']

            n_items = grupo_dataset['clase'].shape[1]

            for i in range(n_items):
                entrada = {}
                
                ref_clase = grupo_dataset['clase'][0, i]
                entrada['clase'] = ''.join(chr(c) for c in mat_file[ref_clase][:].flatten())

                ref_superClase = grupo_dataset['superClase'][0, i]
                entrada['superClase'] = ''.join(chr(c) for c in mat_file[ref_superClase][:].flatten())

                id_ref = grupo_dataset['idSuperClass'][0, i]
                entrada['idSuperClass'] = int(np.array(mat_file[id_ref])[0][0])

                ant_ref = grupo_dataset['antocianinas'][0, i]
                entrada['antocianinas'] = np.array(mat_file[ant_ref]).flatten()

                htrain_ref = grupo_dataset['h2d_lab_train'][0, i]
                entrada['h2d_lab_train'] = np.array(mat_file[htrain_ref])

                htest_ref = grupo_dataset['h2d_lab_test'][0, i]
                entrada['h2d_lab_test'] = np.array(mat_file[htest_ref])

                dataset_completo.append(entrada)

            mat_file.close()
        return dataset_completo
    
    
    
    "Método que aplana histograma de dos dimensiones a una"
    def aplanar_histogramas_2d(self, ruta, patron, nombre_entrenamiento, nombre_test):
        datos = self.obtener_dataset(ruta, patron)
        filas = []
        
        for entrada in datos:
            if nombre_entrenamiento in entrada:
                entrada[nombre_entrenamiento] = entrada[nombre_entrenamiento].reshape(-1)
                
            if nombre_test in entrada:
                entrada[nombre_test] = entrada[nombre_test].reshape(-1)
                
        data_dict = {
            'clase': [d['clase'] for d in datos],
            'superClase': [d['superClase'] for d in datos],
            'idSuperClass': [d['idSuperClass'] for d in datos],
            nombre_entrenamiento: [d[nombre_entrenamiento] for d in datos],
            nombre_test: [d[nombre_test] for d in datos],
        }
    
        # Guardar en archivo .mat
        savemat('datos_lab2.mat', data_dict)
                
        return datos
    
    
    

    def leer_primer_elemento(self, ruta_archivo):
        """Abre un archivo .dat y extrae solo el primer elemento del dataset."""
        mat_file = h5py.File(ruta_archivo, 'r')  # Abre el archivo
        grupo_dataset = mat_file['dataset']       # Accede al grupo 'dataset'
        
        # Obtiene solo el primer elemento (i=0)
        entrada = {
            'clase': ''.join(chr(c) for c in mat_file[grupo_dataset['clase'][0, 0]][:].flatten()),
            'superClase': ''.join(chr(c) for c in mat_file[grupo_dataset['superClase'][0, 0]][:].flatten()),
            'idSuperClass': int(np.array(mat_file[grupo_dataset['idSuperClass'][0, 0]])[0][0]),
            'antocianinas': np.array(mat_file[grupo_dataset['antocianinas'][0, 0]]).flatten(),
            'avg_rgb_test': np.array(mat_file[grupo_dataset['avg_rgb_test'][0, 0]]),
            'avg_rgb_train': np.array(mat_file[grupo_dataset['avg_rgb_train'][0, 0]])
        }
        
        mat_file.close()  # Cierra el archivo
        print (entrada)
        return entrada
    
    
    def proceso_KNN(self, ruta, patron, nombre_entrenamiento, nombre_test, nombre_clase, subclase):
        knn = KNN()
        datos_aplanados = self.aplanar_histogramas_2d(ruta, patron, nombre_entrenamiento, nombre_test)
        
        X_train = [d[nombre_entrenamiento] for d in datos_aplanados]
        X_test = [d[nombre_test] for d in datos_aplanados]
        y_clase = [d[nombre_clase] for d in datos_aplanados]
        y_super = [d[subclase] for d in datos_aplanados]
        
        for i in range(10):
            print(i)
            accuracy_clase = knn.clasificador_KNN(X_train, X_test, y_clase,y_clase, 1)  
            accuracy, accurac2 = knn.clasificador_fuzzy_knn(X_train, X_test, y_clase, y_clase, 1, 2)
            print("-----------------------------------------")
                      
       # _, accuracy_super_clase = knn.clasificador_KNN(X_train, X_test, y_super, k)
        #print("Accuracy super clase: ", accuracy_super_clase)


procesoDatos = ProcesoDatos()
ruta = "C:/Users/Saarayim/Desktop/SAARA"
patron = "DB2DLab*.mat"
ruta_archivo = "C:/Users/saara/Desktop/MIA 2/Vision por computadora/DataBase/AverageRGB1_01-Oct-2020.mat"

#dataset = procesoDatos.obtener_dataset(ruta, patron)
#dataset = procesoDatos.aplanar_histogramas_2d(ruta, patron, 'h2d_lab_train', 'h2d_lab_test')
#primer_elemento = procesoDatos.leer_primer_elemento(ruta_archivo)
#with h5py.File(ruta_archivo, 'r') as mat_file:
#    print("Grupos disponibles en el archivo:", list(mat_file.keys()))
#    grupo_dataset = mat_file['dataset']
#    print("Campos en 'dataset':", list(grupo_dataset.keys()))

procesoDatos.proceso_KNN(ruta, patron, 'h2d_lab_train', 'h2d_lab_test', 'clase', 'superClase')



        