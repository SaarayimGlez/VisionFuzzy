function [output] = Impl_ClasificacionKNN(pathImg, extension, dimensionType)
 directorio = pathImg;
 dbPopulations = dir(strcat(directorio,extension)); % Cargar todas las muestras de entrenamiento 
 N=length(dbPopulations);
 resAccuClase = [];
 resAccuSuperClase = [];
 resAccuClaseFuzzy = [];
 resAccuSuperClaseFuzzy = [];
 resAccuClaseFuzzy_2 = [];
 resAccuSuperClaseFuzzy_2 = [];
 for i = 1 : N
    dbfile = dbPopulations(i).name;
    %disp([int2str(i), ' ', dbfile]);
    db = load(strcat(directorio, dbfile), '-mat');
    [train, test, clase, superClase] = structDB2TrainingandTest(db,1,size(db.dataset,1), dimensionType);

    %%
    accuracyClase = [];
    accuracySuperClase = [];
    accuracyClaseFuzzy = [];
    accuracySuperClaseFuzzy = [];
    accuracyClaseFuzzy_2 = [];
    accuracySuperClaseFuzzy_2 = [];
    kvector = [1,3,5,7,9,11,13,15];
    distance = 'cityblock';
    ponderar = 'squaredinverse';
   for K=1:length(kvector) % para k1, k3
         [Accu1, confusionMatrix1, predictedClass1, fails1] = knn_Matlab(train, test, clase, distance,kvector(K), ponderar);
         [Accu2, confusionMatrix2, predictedClass2, fails2] = knn_Matlab(train, test, superClase, distance,kvector(K), ponderar);
         [AccuF1_1, AccuF1_2, ~, ~, ~, ~, ~, ~, ~] = fuzzy_knn_Matlab(train, test, clase, kvector(K), 2, distance);
         [AccuF2_1, AccuF2_2, ~, ~, ~, ~, ~, ~, ~] = fuzzy_knn_Matlab(train, test, superClase, kvector(K), 2, distance);
         accuracyClase = [accuracyClase,(Accu1 * 100)]       
         accuracySuperClase = [accuracySuperClase,(Accu2 * 100)] 
         accuracyClaseFuzzy = [accuracyClaseFuzzy; (AccuF1_1 * 100)]       
         accuracySuperClaseFuzzy = [accuracySuperClaseFuzzy; (AccuF2_1 * 100)]
         accuracyClaseFuzzy_2 = [accuracyClaseFuzzy_2, (AccuF1_2 * 100)];
         accuracySuperClaseFuzzy_2 = [accuracySuperClaseFuzzy_2, (AccuF2_2 * 100)];
         
   end
   
   resAccuClase = [resAccuClase;accuracyClase];
   resAccuSuperClase = [resAccuSuperClase;accuracySuperClase];


   resAccuClaseFuzzy = [resAccuClaseFuzzy; accuracyClaseFuzzy'];
   resAccuSuperClaseFuzzy = [resAccuSuperClaseFuzzy; accuracySuperClaseFuzzy'];

   resAccuClaseFuzzy_2  = [resAccuClaseFuzzy_2; accuracyClaseFuzzy_2];
   resAccuSuperClaseFuzzy_2 = [resAccuSuperClaseFuzzy_2; accuracySuperClaseFuzzy_2];

   clear db train test

end   
  
  contador = (1:N)';
  resAccuClase=[resAccuClase;mean(resAccuClase);std(resAccuClase)]
  resAccuSuperClase=[resAccuSuperClase;mean(resAccuSuperClase);std(resAccuSuperClase)]


  resAccuClaseFuzzy = [resAccuClaseFuzzy; mean(resAccuClaseFuzzy); std(resAccuClaseFuzzy)];
  resAccuSuperClaseFuzzy = [resAccuSuperClaseFuzzy; mean(resAccuSuperClaseFuzzy); std(resAccuSuperClaseFuzzy)];
   
  resAccuClaseFuzzy_2 = [resAccuClaseFuzzy_2; mean(resAccuClaseFuzzy_2); std(resAccuClaseFuzzy_2)];
  resAccuSuperClaseFuzzy_2 = [resAccuSuperClaseFuzzy_2; mean(resAccuSuperClaseFuzzy_2); std(resAccuSuperClaseFuzzy_2)];

  
  etiquetas = [strcat("Muestra_", string(1:N))'; "Promedio"; "Desviacion"];

  etiquetas = cellstr(etiquetas);
    
  Concentrado_hs = table(etiquetas, resAccuClase, resAccuClaseFuzzy, resAccuClaseFuzzy_2, ...
    'VariableNames', {'Muestra', 'KNN_Clasico', 'Fuzzy_KNN_1', 'Fuzzy_KNN_2'});

  Concentrado_ab = table(etiquetas, resAccuSuperClase, resAccuSuperClaseFuzzy, resAccuSuperClaseFuzzy_2, ...
    'VariableNames', {'Muestra', 'KNN_Clasico', 'Fuzzy_KNN_1', 'Fuzzy_KNN_2'});


    finalDir = strcat(pathImg, 'Report');
    if ~exist(finalDir, 'dir')
        mkdir(finalDir);
    end
    
    filename = strcat(finalDir, '/D_', dimensionType, '_concentrado_', distance, '_', ponderar, '.xlsx');
    writetable(Concentrado_hs, filename, 'Sheet', 'Clases');
    writetable(Concentrado_ab, filename, 'Sheet', 'SuperClases');
    
    output = 1;
end

