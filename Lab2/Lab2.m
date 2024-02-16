data_turkish = csvread('turkish-se-SP500vsMSCI.csv');
data_cars = readtable("mtcarsdata-4features.csv");
% Removing first column in order to make the table convertible into a
% matrix
data_cars(:, 1) = [];
% Converting the table into a matrix
data_cars = table2array(data_cars);

% disp(data_turkish);
% disp(data_cars);

%%% Task 2.1
rows_turkish = size(data_turkish,1);

sp = data_turkish(:,1);
msci = data_turkish(:,2);

% Perform linear regression using the with S&P and MSCI columns
[x_turk,y_turk,w_turk] = lin_regr_subset(data_turkish);

% Commands to plot both every datas and regression line 
% plot(sp, msci,'o');
% xlabel('S&P 500 return index'); 
% ylabel('MSCI Europe index'); 
% grid on;
% hold on;
% plot(sp, y_turk, 'r'); 
% hold off;

%%% Task 2.2
% Define the number of tests to compare
num_test_tocompare = 3;
% Comparison between 3 different random subsets and plot the correspondent
% regression lines and data points

% for i=1:num_test_tocompare 
%     color = ['r','b','k','y'];
%     [test_mat, ~] = select_random_subset(data_turkish, 0.1);
%     [x, y, w] = lin_regr_subset(test_mat);  
% 
%     hold on;
%     scatter(x, test_mat(:,2), 'o',color(i));
%     xlabel('S&P 500 return index');
%     ylabel('MSCI Europe index');
%     grid on;
%     plot(x, y, color(i),'LineWidth', 2);
%     hold off;
% 
% end

%%% Taks 2.3
% Extract 'mpg' and 'weight' columns from 'data_cars'
mpg = data_cars(:,1);
weight = data_cars(:,4);

% Perform linear regression with intercept using 'lin_regr_intercept' function
y_new = lin_regr_intercept(mpg,weight);

% Commands to plot the regression line with intercept and the correspondent
% data points
% scatter(weight, mpg, 'o');
% xlabel('Mpg'); 
% ylabel('Weight'); 
% grid on;
% hold on;
% plot(weight, y_new, 'r','LineWidth', 2); 
% hold off;

%%% Task 2.4
% Extract 'dis', 'hp', 'mpg' columns from 'data_cars'
dis = data_cars(:,2);
hp = data_cars(:,3);
y_n = multi_regression(data_cars);

% Commands to plot the 3 attributes function of real mpg values and
% predicted mpg values, in order to show the difference between the two 
labels = {'Displacement (disp)', 'Horsepower (hp)','Weight (weight)'};

% for i=1:3
% subplot(1,3,i)
% scatter(data_cars(:,i+1),mpg,'filled','r');
% xlabel(labels{i})
% ylabel('Miles per gallon (mpg)')
% hold on 
% scatter(data_cars(:,i+1),y_n,'filled','b');
% legend('Using real mpg values', 'Using predicted mpg values');
% grid on
% end

disp("Real mpg values");
disp(mpg);
disp("Predicted mpg values");
disp(y_n);
disp("Difference between real and predicted mpg values expressed in absolute value");
disp(abs(mpg-y_n));

num_indexes = 10;
indexes = zeros(1,num_indexes);
for i=1:size(indexes,2)
    indexes(i) = i;
end

% Initialize arrays to store MSE results
vet_5_turkish = zeros(1,num_indexes);
vet_95_turkish = zeros(1,num_indexes);

vet_5_lin_cars = zeros(1,num_indexes);
vet_95_lin_cars = zeros(1,num_indexes);

vet_5_multi_cars = zeros(1,num_indexes);
vet_95_multi_cars = zeros(1,num_indexes);

% Task 3 repeating es 2.1
for i=1:num_indexes

[subset_turk, remaining_turk] = select_random_subset(data_turkish, 0.05);
[~, y_turk, ~] = lin_regr_subset(subset_turk);
[~, y_turk_new, ~] = lin_regr_subset(remaining_turk);

mse_5_percent = calc_mse(subset_turk(:,2), y_turk);
mse_95_percent = calc_mse(remaining_turk(:,2), y_turk_new);

% disp(mse_5_percent);
% disp(mse_95_percent);

vet_5_turkish(i) = mse_5_percent;
vet_95_turkish(i) = mse_95_percent;

end

% Task 3 repeating es 2.3
for i=1:num_indexes

[subset_cars, remaining_cars] = select_random_subset(data_cars, 0.05);
mpg_sub = subset_cars(:,1);
weight_sub = subset_cars(:,4);

y_new_05 = lin_regr_intercept(mpg_sub,weight_sub);
y_new_95 = lin_regr_intercept(remaining_cars(:,1),remaining_cars(:,4));

mse_5_cars = calc_mse(mpg_sub, y_new_05);
mse_95_cars = calc_mse(remaining_cars(:,1), y_new_95);
% disp(mse_5_cars);
% disp(mse_95_cars);

vet_5_lin_cars(i) = mse_5_cars;
vet_95_lin_cars(i) = mse_95_cars;

end

%%% Task 3 repeating es 2.4
for i=1:num_indexes

[subset_multi_cars, remaining_multi_cars] = select_random_subset(data_cars, 0.05);
y_n_05 = multi_regression(subset_multi_cars);
y_n_95 = multi_regression(remaining_multi_cars);
% disp(y_n_05);
% disp(y_n_95);

mse_5_multi = calc_mse(subset_multi_cars(:,1), y_n_05);
mse_95_multi = calc_mse(remaining_multi_cars(:,1), y_n_95);
% disp(mse_5_multi);
% disp(mse_95_multi);

vet_5_multi_cars(i) = mse_5_multi;
vet_95_multi_cars(i) = mse_95_multi;

end

% Create tables to display the results
tab_turk = table(indexes', vet_5_turkish', vet_95_turkish', 'VariableNames', {'INDEXES', 'MSE 5%', 'MSE 95%'});
disp("LINEAR REGRESSION BETWEEN S&P AND MSCI")
disp(tab_turk);

tab_lin_regr = table(indexes', vet_5_lin_cars', vet_95_lin_cars', 'VariableNames', {'INDEXES', 'MSE 5%', 'MSE 95%'});
disp("LINEAR REGRESSION WITH INTERCEPT BETWEEN MPG AND WEIGHT")
disp(tab_lin_regr);

tab_mult_regr = table(indexes', vet_5_multi_cars', vet_95_multi_cars', 'VariableNames', {'INDEXES', 'MSE 5%', 'MSE 95%'});
disp("MULTI REGRESSION BETWEEN MPG AND DISP, HP, WEIGHT")
disp(tab_mult_regr);



% Function for linear regression on a given matrix
% that returns predicted values (y) and the columns values
function [x,y,w] = lin_regr_subset(subset_mat)
    % Extract the columns as the independent variable (x) 
    % and dependent variable (y)
    x = subset_mat(:,1);
    y = subset_mat(:,2);
    
    % Using the formula on the slide in order to calculate
    % numerator (num) and denominator (den)
    num = sum(x.*y);
    den = sum(x.*x);

    % Calculate the coefficient (w) for linear regression
    % and the predicted values (y)
    w = num / den;
    y = w * x;        

end

% Function to select a random subset of data
function [subset_matrix, remaining_mat] = select_random_subset(data_set, subset_percentage)

    [rows,col] = size(data_set);
    % Calculate the number of rows to include in the subset
    m = subset_percentage * rows;
    m = round(m);
    % Generate a vector of random indices
    allIndices = randperm(rows);
    % Select the m rows to extract 
    randomSubset = allIndices(1:m);
    % Select the remaining rows
    allIndices_new = allIndices(m+1:rows);
    % Extract the subset and the remaining data
    subset_matrix = data_set(randomSubset,:);
    remaining_mat = data_set(allIndices_new,:);

end

% Function for computing multiple regression on a given matrix
function y_n = multi_regression(data_cars)
mpg = data_cars(:,1);
dis = data_cars(:,2);
hp = data_cars(:,3);
weight = data_cars(:,4);

% Create a matrix of independent variables
x = [dis,hp,weight];

% Add a column of 1s for the intercept term
x = [ones(size(x, 1), 1), x];

% x_n = x'*x;
% w_n = x_n\eye(size(x_n)) * x';
% w_n = w_n * mpg;
% disp(w_n);

% Calculate the coefficient vector (w) using the pseudoinverse as seen in
% class
moor_pen = pinv(x);

w = moor_pen * mpg;
%disp(w);

%%% w computed using pinv function while w_n computed using all the
%%% calculations needed in order to avoid using pinv

% Calculate the estimated values (y_n) based on the coefficients (w)
y_n = w(1) + w(2) * dis + w(3) * hp + w(4) * weight;

end

% Function to calculate the mean squared error (MSE)
function mse = calc_mse(y, y_pred)

    mse = mean((y - y_pred).^2);

end

% Function for linear regression with an intercept on two given vectors 
function y_new = lin_regr_intercept(mpg,weight)
    
    row_mpg = size(mpg,1);
    row_weight = size(weight,1);
    
    % Calculate the mean value of weight
    x_medio = mean(weight);
    % Calculate the mean value of mpg
    t_medio = mean(mpg);
    
    % Calculate the denominator for the coefficient (w1) calculation
    w1_den = 0;
    for i=1:row_weight
        w1_den = w1_den + (weight(i)-x_medio)^2;
    end
    %disp(w1_den);
    
    % Calculate the numerator for the coefficient (w1) calculation
    w1_num = 0;
    for i=1:row_mpg
        w1_num = w1_num + ((mpg(i)-t_medio) * (weight(i)-x_medio));
    end
    %disp(w1_num);
    
    % Calculate the coefficient (w1)
    w1 = w1_num/w1_den;
    
    % Calculate the intercept term (w0)
    w0 = t_medio - (w1 * x_medio);
    
    % Calculate the estimated values (y_new) based on w1 and w0
    y_new = (w1 * weight) + w0;

end
