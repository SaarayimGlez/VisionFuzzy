import h5py
import numpy as np
import glob
import os

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
        
        for entrada in datos:
            if nombre_entrenamiento in entrada:
                entrada[nombre_entrenamiento] = entrada[nombre_entrenamiento].reshape(-1)
                
            if nombre_test in entrada:
                entrada[nombre_test] = entrada[nombre_test].reshape(-1)
                
        return datos
            
        
        