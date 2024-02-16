% Load the MNIST training and test sets
[training_set, training_set_classes] = loadMNIST(0);
[test_set, test_set_classes] = loadMNIST(1);
% Define the number of classes and experiments
num_classes = 10;
num_of_experiment = 5;
% Define the k values for k-NN
k_values = [1,2,3,4,5,10,15,20,30,40,50];
% Initialize row and column names for the results table
rowNames = cell(num_classes, 1);
for i = 1:num_classes
    rowNames{i} = ['Class = ', num2str(i)];
end
colNames = cell(size(k_values,2), 1);
for i = 1:size(k_values,2)
    colNames{i} = ['K = ', num2str(k_values(i))];
end
% Initialize the results matrix and quality indexes cell array
results = zeros(num_classes, size(k_values, 2));
quality_indexes = cell(num_of_experiment, num_classes);  % Now we use only one cell array for the tables
var = 0;
% Loop over the number of experiments
for k = 1:num_of_experiment
    % Select a random subset of the training set
    [subset_training_set,subset_training_classes] = select_random_subset(training_set,training_set_classes,0.05);
    
    % Loop over the number of classes
    for i = 1:num_classes
        % Create binary class labels for the current class
        training_target_new = subset_training_classes;
        training_target_new(training_target_new ~= i) = 0;
        
        test_target_new = test_set_classes;
        test_target_new(test_target_new ~= i) = 0;
        
        % Initialize an array of empty tables
        quality_indexes{k,i} = table();
        
        % Create a new figure for each class if used in orther to plot only
        % 10 * 11 confusion matrices
        if (var <= 0)
            figure
        end
        % Loop over the k values
        for j = 1:size(k_values, 2)
            % Compute the k-NN classifier
            [pred_class, acc] = compute_knn(subset_training_set, training_target_new, test_set, k_values(j), test_target_new);
            
            % Store the accuracy in the results matrix
            results(i, j) = acc;
            
            % Compute the confusion matrix
            c = confusionmat(test_target_new, pred_class);
            subplot(2,6,j)
            confusionchart(test_target_new, pred_class);
            
            % Calculate sensitivity and specificity
            sensitivity = c(1, 1) / (c(1, 1) + c(1, 2));
            specificity = c(2, 2) / (c(2, 1) + c(2, 2));
            
            % Add new rows to the table
            new_row = table(sensitivity, specificity, 'VariableNames', {'Sensitivity', 'Specificity'});
            quality_indexes{k,i} = [quality_indexes{k,i}; new_row];
        end
    end
    var = var + 1;
end
% Convert the results matrix to a table and display it
T = array2table(results, 'RowNames', rowNames, 'VariableNames', colNames);
disp(T);
% Covert each cell into a matrix 
quality_indexes_cell = cellfun(@table2array, quality_indexes, 'UniformOutput', false);
cell_array1= quality_indexes_cell(1,:);
cell_array2= quality_indexes_cell(2,:);
cell_array3= quality_indexes_cell(3,:);
cell_array4= quality_indexes_cell(4,:);
cell_array5= quality_indexes_cell(5,:);
% Sum the quality indexes across all experiments
sum_quality_indexes =  cellfun(@(a, b, c, d, e) a + b + c + d + e, cell_array1, cell_array2, cell_array3, cell_array4, cell_array5, 'UniformOutput', false);
% Compute the mean quality index
mean_cell_array = cellfun(@(x) x / num_of_experiment, sum_quality_indexes, 'UniformOutput', false);
% Initialize the cell array for standard deviations
std_cell_array = cell(size(cell_array1));
% Loop over each cell to compute the standard deviation
for i = 1:num_classes
    for j=1:size(k_values,2)
        % Extract the corresponding elements from each cell array
        values1 = [cell_array1{i}(j,1), cell_array2{i}(j,1), cell_array3{i}(j,1), cell_array4{i}(j,1), cell_array5{i}(j,1)];
        values2 = [cell_array1{i}(j,2), cell_array2{i}(j,2), cell_array3{i}(j,2), cell_array4{i}(j,2), cell_array5{i}(j,2)];
        
        % Compute the standard deviation and store it in the cell array
        std_cell_array{i}(j,1) = [std(values1)];
        std_cell_array{i}(j,2) = [std(values2)];
    end
end
% Convert the mean and standard deviation cell arrays to tables and display them
std_table_array = cell(1,num_classes);
mean_table_array = cell(1,num_classes);
for i = 1:num_classes
    mean_table_array{1, i} = array2table(mean_cell_array{1, i}, 'RowNames', colNames, 'VariableNames', {'Sensitivity', 'Specificity'});
    fprintf("Mean of 5 observation for class %d and not %d\n", i,i);
    disp(mean_table_array{1, i});
end
for i = 1:num_classes
    std_table_array{1, i} = array2table(std_cell_array{1, i}, 'RowNames', colNames, 'VariableNames', {'Sensitivity', 'Specificity'});
    fprintf("Std between 5 observation for class %d and not %d\n", i,i);
    disp(std_table_array{1, i});
end




function [predicted_class, accuracy] = compute_knn(training_mat, target_classes, test_mat, k, test_classes)
    % Check the number of input arguments
    numb_arg = nargin;
    if (numb_arg < 4 || numb_arg >5)
        errore = "Wrong numbers of input";
        error(errore);
    end
    % Check the dimensions of the input matrices
    check_dimension(training_mat, target_classes, test_mat, k);
    
    % Compute the distances between the points in the test and training sets
    distanze = pdist2(test_mat, training_mat);
    
    % Find the indices of the k shortest distances for each point in the test set
    [~, indici_righe] = mink(distanze, k, 2);
    
    % Extract the target classes of the k nearest neighbors
    nearest_k = target_classes(indici_righe);
    
    % Compute the predicted class for each point in the test set
    if (numb_arg == 4) 
        predicted_class = mode(nearest_k,2);
        disp("Vector containing predicted classes for test set");
        disp(predicted_class);
    elseif (numb_arg == 5)
        % If the true classes of the test set are provided, compute the accuracy
        predicted_class = mode(nearest_k,2);
        error_vet = (test_classes == predicted_class);
        error_percent = sum(error_vet == 0);
        error_percent = error_percent / size(error_vet,1);
        error_percent = error_percent * 100;
        accuracy = 100-error_percent;
        fprintf("Error percent: %f\n",error_percent);
        fprintf("Accuracy percent: %f\n", accuracy);
    end
end
function [subset_matrix, subset_classes] = select_random_subset(data_set, data_classes, subset_percentage)
    [rows,col] = size(data_set);
    % Calculate the number of rows to include in the subset
    m = subset_percentage * rows;
    m = round(m);
    % Generate a vector of random indices
    allIndices = randperm(rows);
    % Select the m rows to extract 
    randomSubset = allIndices(1:m);
    % Select the remaining rows
    %allIndices_new = allIndices(m+1:rows);
    % Extract the subset and the remaining data
    subset_matrix = data_set(randomSubset,:);
    subset_classes = data_classes(randomSubset,:);
    %remaining_mat = data_set(allIndices_new,:);
end
function check_dimension(training_mat, target_classes, test_mat, k)
    [n,d] = size(training_mat);
    n_new = size(target_classes,1);
    [~, d_new] = size(test_mat);
    if (n ~= n_new)
        error("Number of rows in the training set is different from the number of rows of the correspondent classes");
    elseif (d ~= d_new)
        error("Number of columns in the training set is different from the number of columns of the test set");
    end
    if (k < 0)
        error("K used < 0, provide a k > 0");
    elseif (k > n)
        error("K used is > than number of rows of the training set: %d, provide a k < than %d",n,n);
    end
end