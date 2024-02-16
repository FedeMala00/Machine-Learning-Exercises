data_matrix = readmatrix('datas.csv');
%disp(data_matrix);

 %vet is the vector containing all the size of the vectors that will be stored
 %inside the cells, I've declared it global because it always remains the same
 %unless data_matrix is changed

 global vet;
 vet=zeros(1, size(data_matrix, 2) - 1); %-1 because data_matrix has 4 attributes column + 1 class column

 %calculate the number of unique values for each attribute and store in 'vet'                                        
 for i=1:size(data_matrix, 2) - 1
       vet(i) = [numel(unique(data_matrix(:,i)))];
 end

% generate n and m indexes that correspond to n and m rows which will be
% stored in a new cell array
indexes = randperm(size(data_matrix,1));

%because it's required to have 10 training patterns and 4 (the remaining)
%test patterns
training_set_pat = 10; 

%split data into training and test sets
training_set_matrix = data_matrix(indexes(1:training_set_pat), :);
test_set_matrix = data_matrix(indexes(training_set_pat+1:size(data_matrix,1)), :);

disp("Test set matrix");
disp(test_set_matrix);
disp("Training set matrix");
disp(training_set_matrix);

[cellarr_training, n_y, n_n] = fillCell(training_set_matrix);
disp("Cellarray containing the likelihood probabilities, first row=P(xi|yes) second row=P(xi|no)")
disp(cellarr_training);

% disp(n_y);
% disp(n_n);

%perform classification on the test set and display the results
readResult(test_set_matrix, cellarr_training, n_y, n_n, training_set_matrix);

function test_mat_new = check_attributes(training_mat, test_mat)
%function for checking if there are test set attribute values that are
%not in the training set, and if so it it discards the corresponding pattern
v = training_mat(:);
max_val = max(unique(v));

above_max = test_mat>max_val;
[row_above, col] = find(above_max);

row_above = transpose(row_above);

test_mat(row_above, :) = [];
test_mat_new = test_mat;

end

function num_min_column = check_dimension(training_mat, test_mat)
%function to check the dimensions of training and test matrices
d = size(training_mat,2);

c = size(test_mat,2);

if (c == d || c == d - 1)

    disp("the number of columns of the training matrix and the test one is correct");
    num_min_column = d;

else

    error("error in the size of the two matrices");

end

end

function readResult(matrix, cellArray, num_yes, num_no, training_mat)

    column_training_mat = check_dimension(training_mat,matrix);
    matrix_new = check_attributes(training_mat,matrix);
    
    if  all(matrix_new(:) > 0) && all(training_mat(:) > 0)

        num_c = size(matrix_new, 2) - 1;
        num_r = size(matrix_new, 1);
        vettore = zeros(num_r,1);
            
        for j=1:num_r
                
            g1 = 1;
            g2 = 1;
            

            for i=1:num_c
               
                   g1 = g1 * cellArray{1,i}(1,matrix_new(j,i));
                   vec_g1(j) = g1;
        
                   g2 = g2 * cellArray{2,i}(1,matrix_new(j,i));
                   vec_g2(j) = g2;
                    
            end
            g1 = 1;
            g2 = 1;
        end
    
         %disp(size(vec_g1,2));
         vec_g1 = vec_g1 * (num_yes/(num_yes + num_no));
         vec_g2 = vec_g2 * (num_no/(num_yes + num_no));
         
         disp("Vector containing gyes");
         disp(vec_g1);
         disp("Vector containing gno");
         disp(vec_g2);
    
        greater_g1 = vec_g1 > vec_g2;
    
        for i=1:size(vec_g1,2)
    
            if(greater_g1(i) == true)
                
                vettore(i) = 1;
    
            else
                   
                vettore(i) = 2;       
               
            end
            
        end
        if (column_training_mat == size(matrix_new, 2))
            disp("Classification obtained with 1=Yes and 2=No");
            disp(vettore);
            disp("Real classification based on the last column 1=Yes 2=No");
            disp(matrix_new(:,num_c+1));
            differences = vettore ~= matrix_new(:,num_c+1);
            numDifferences = sum(differences);
            errorRate = numDifferences/num_r;
            fprintf("error rate: %f\n",errorRate);
            fprintf("\n");
        else
            disp("Classification obtained with 1=Yes and 2=No");
            disp(vettore);

        end

    else
        error("wrong values (<1) in one of the two the matrices");
       
    end
end


function [cell_return,num_yes,num_no] = fillCell(initial_matrix)
    global vet;
    a = 1;
    
    num_yes = sum(initial_matrix(:, size(initial_matrix,2)) == 1);
    num_no = sum(initial_matrix(:, size(initial_matrix,2)) == 2);

    num_col = size(initial_matrix,2);
    num_row = size(initial_matrix,1);

    num_class = numel(unique(initial_matrix(:,num_col))); % calculate the number of different element in the last column (class column)
    num_attr = num_col - 1; % because we know that the last column contains classes
    
    cell_return = cell(num_class, num_attr);
   
    %disp(vec_num_attr)
    
    for i=1:num_attr
        %for j=1:num_attr
            cell_return{1,i} = zeros(1, vet(i));
            cell_return{2,i} = zeros(1, vet(i));
        %end
    end        
    
    count1 = 0;
    count2 = 0;
   
  for k=1:num_attr
    for j=1:vet(k)
        for i=1:num_row
            if (initial_matrix(i,k) == j && initial_matrix(i,num_col) == 2)

                count1 = count1 + 1;

            elseif (initial_matrix(i,k) == j && initial_matrix(i,num_col) == 1)

                count2 = count2 + 1;

            end

            cont1 = sum(initial_matrix(:,k) == j) - count1;
            cont2 = sum(initial_matrix(:,k) == j) - count2;
            %used the Laplace smoothing with a=1 and vet=[3,3,2,2]
            %calculated using the number of different occurencies in the
            %various columns in data matrix 
            cell_return{1,k}(1,j) = ((cont1 + a)/(num_yes + vet(k)));
            cell_return{2,k}(1,j) = ((cont2 + a)/(num_no + vet(k)));

        end
        count1 = 0;
        count2 = 0;

    end
  end
end