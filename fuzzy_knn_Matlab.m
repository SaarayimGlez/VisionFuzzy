function [Accuracy_1, Accuracy_2, ...
          predicted_labels_1, predicted_labels_2, ...
          membership_values, ...
          confusionMatrix_1, confusionMatrix_2, ...
          fails_1, fails_2] = fuzzy_knn_Matlab(train, test, labels_train, K, m, distance_type)

    if ~iscell(labels_train)
        labels_train = cellstr(labels_train);
    end
    
    N_test = size(test, 1);
    unique_classes = unique(labels_train);
    num_classes = length(unique_classes);
    
    predicted_labels_1 = cell(N_test, 1);
    predicted_labels_2 = cell(N_test, 1);
    membership_values = zeros(N_test, num_classes);
    
    dist_matrix = pdist2(test, train, distance_type);
    
    for i = 1:N_test
        [sorted_dists, idx] = sort(dist_matrix(i,:));
        vecinos_idx = idx(1:K);
        vecinos_dists = sorted_dists(1:K);
        vecinos_labels = labels_train(vecinos_idx);
        

        vecinos_dists(vecinos_dists == 0) = 1e-10;
        weights = (1./vecinos_dists).^(2/(m-1));
        
        class_membership = zeros(1, num_classes);
        for c = 1:num_classes
            mask = strcmp(vecinos_labels, unique_classes{c});
            class_membership(c) = sum(weights(mask));
        end
        
        class_membership = class_membership / sum(class_membership);
        membership_values(i,:) = class_membership;

        [sorted_membership, sorted_idx] = sort(class_membership, 'descend');
        
        predicted_labels_1{i} = unique_classes{sorted_idx(1)};
        predicted_labels_2{i} = unique_classes{sorted_idx(2)};
        

        fprintf('%s %.1f%%, %s %.1f%%\n', ...
            unique_classes{sorted_idx(1)}, sorted_membership(1)*100, ...
            unique_classes{sorted_idx(2)}, sorted_membership(2)*100);
    end
    
    [Accuracy_1, confusionMatrix_1, fails_1] = accuracy(labels_train, predicted_labels_1);
    [Accuracy_2, confusionMatrix_2, fails_2] = accuracy(labels_train, predicted_labels_2);
end



function [ accuracy, confusionMatrix, fails ] = accuracy( training_c_label, prediction_c_label )

 
 if ~iscell(training_c_label) || ~iscell(prediction_c_label)  
    error('MyComponent:incorrectType',...
          'Error. \nEntrada debe ser tipo cell');
 end   
 
 confusionMatrix = zeros(length(unique(training_c_label)), length(unique(training_c_label)));
 positions = unique(training_c_label);
 success = 0;
 fails = {};
 for i = 1: length(training_c_label) 
     if(strcmpi(training_c_label{i},prediction_c_label{i}))
        success = success + 1;
     else
         fails = [fails; {training_c_label{i}, prediction_c_label{i}}]; 
     end
     col = find(strcmp(positions, training_c_label{i}));
     row = find(strcmp(positions, prediction_c_label{i}));
     confusionMatrix(col,row) = confusionMatrix(col,row) + 1;     
 end 
accuracy = success / length(training_c_label);
end