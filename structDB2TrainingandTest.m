function [ train,  test, clase, superClase ] = structDB2TrainingandTest( db, ini, fin, dimensionType )
 % structDB2TrainingandTest Lee base de datos y obtiene la conjunta de entrenamiento y
 % validaci�n para formar los vectores de entrenamiento y validaci�n.
 % Entrada:
 %         matfile: nombre de archivo de base de datos
 %         Kernel : matriz para el kernel de suavizado
 %         filter : bandera para indicar aplicaci�n de filtro de suavizado
 % Salida:
 % training_hs :vector de entrenamiento de los componentes de color HS.
 % training_ab :vector de entrenamiento de los componentes de color AB.
 % training_hs_ab:vector de entrenamiento de los componentes de color HS y
 % AB.
 % test_hs:vector de prueba de los componenentes de color HS.
 % test_ab: vector de prueba de los componentes de color AB.
 % test_hs_ab:vector de prueba de los componentes de color HS y AB.
 % trainingLabel : vector de las clases de entrenamiento.
 % testLabel: vector de las clases de prueba.
 
if(strcmp(dimensionType, 'AvgLab')) % for mean of two channels of color
    train = [];
    test = [];    
elseif(strcmp(dimensionType, 'AvgRGB')) % for mean of two channels of color
    train = [];
    test = [];    
elseif (strcmp(dimensionType, '2DLab')) % dos dimensiones    
    train = [];
    test = [];    
elseif(strcmp(dimensionType, '3DLab')) % en tres dimensiones
    train = zeros(54, 16777216);
    test = zeros(54, 16777216);
elseif(strcmp(dimensionType, '3DRGB')) % en tres dimensiones
    train = zeros(54, 16777216);
    test = zeros(54, 16777216);    
end
    
clase = [];
superClase = [];
     for k = ini:fin            
        data = db.dataset(k);
        %disp([num2str(k), ' - ',data.trainingLabel]);
        if(strcmp(dimensionType, 'AvgLab')) % for mean of two channels of color
            train = [train;data.avg_lab_train]; 
            test = [test;data.avg_lab_test];
        elseif(strcmp(dimensionType, 'AvgRGB')) % for mean of two channels of color
            train = [train;data.avg_rgb_train]; 
            test = [test;data.avg_rgb_test];            
        elseif (strcmp(dimensionType, '2DLab')) % dos dimensiones
            %disp(['    Reshape 2D Lab', data.clase, ' - ', data.superClase]);
            h2d_lab_train = data.h2d_lab_train;
            h2d_lab_test = data.h2d_lab_test;            
            res_train = reshape(h2d_lab_train, size(h2d_lab_train,1)*size(h2d_lab_train,2),1);
            res_test = reshape(h2d_lab_test, size(h2d_lab_test,1)*size(h2d_lab_test,2),1);
            train = [train;res_train'];
            test = [test;res_test'];                    
        elseif(strcmp(dimensionType, '3DLab')) % en tres dimensiones
            %disp(['    Reshape 3D Lab', data.clase, ' - ', data.superClase]);
            h3d_lab_train = data.h3d_lab_train;
            h3d_lab_test = data.h3d_lab_test;                        
            res_train = reshape(h3d_lab_train, size(h3d_lab_train,1)*size(h3d_lab_train,2)*size(h3d_lab_train,3),1);
            res_test = reshape(h3d_lab_test, size(h3d_lab_test,1)*size(h3d_lab_test,2)*size(h3d_lab_test,3),1);
            train(k, :) = res_train';
            test(k, :) = res_test';           
            %train = [train;res_train'];
            %test = [test;res_test'];
        elseif(strcmp(dimensionType, '3DRGB')) % en tres dimensiones
            %disp(['    Reshape 3D RGB', data.clase, ' - ', data.superClase]);
            h3d_rgb_train = data.h3d_rgb_train;
            h3d_rgb_test = data.h3d_rgb_test;                        
            res_train = reshape(h3d_rgb_train, size(h3d_rgb_train,1)*size(h3d_rgb_train,2)*size(h3d_rgb_train,3),1);
            res_test = reshape(h3d_rgb_test, size(h3d_rgb_test,1)*size(h3d_rgb_test,2)*size(h3d_rgb_test,3),1);
            %train = [train;res_train'];
            %test = [test;res_test'];
            train(k, :) = res_train';
            test(k, :) = res_test';                       
        end
        clase = [clase;{data.clase}];
        superClase = [superClase;{data.superClase}];  
     end
end

