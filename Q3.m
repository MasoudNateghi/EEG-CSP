%% load data
clear; close all; clc;
load("Ex3.mat");
load("AllElectrodes.mat");
fs = 256;
N = size(TrainData, 2);
t = 0:1/fs:(N-1)/fs;
%% part a
% zero mean train data
TrainData_zero_mean = zeros(size(TrainData));
for i = 1:165
    TrainData_zero_mean(:, :, i) = TrainData(:, :, i) - mean(TrainData(:, :, i), 2);
end
% form covariance matrices of each class
C0 = zeros(30, 30);
C1 = zeros(30, 30);
p_index = find(TrainLabel == 1);
n_index = find(TrainLabel == 0);
M1 = length(p_index);
M2 = length(n_index);

for i = p_index
    C1 = C1 + TrainData_zero_mean(:, :, i) * TrainData_zero_mean(:, :, i)';
end
for i = n_index
    C0 = C0 + TrainData_zero_mean(:, :, i) * TrainData_zero_mean(:, :, i)';
end

[V, D] = eig(C1, C0);
[d, index] = sort(diag(D), "descend");
V = V(:, index);
w1  = V(:, 1); 
w30 = V(:, 30);

y1  = w1'  * TrainData(:, :, 1);
y30 = w30' * TrainData(:, :, 1);
figure();
subplot(2, 1, 1)
plot(t, y1)
title('channel 1')
subplot(2, 1, 2)
plot(t, y30)
title('channel 30')
sgtitle('target = 0 (trial 1)')
str = strcat('for trial 1 (target = 0) variance of channel 1 = ', num2str(var(y1)));
disp(str)
str = strcat('for trial 1 (target = 0) variance of channel 30 = ', num2str(var(y30)));
disp(str)


y1  = w1'  * TrainData(:, :, 138);
y30 = w30' * TrainData(:, :, 138);
figure();
subplot(2, 1, 1)
plot(t, y1)
title('channel 1')
subplot(2, 1, 2)
plot(t, y30)
title('channel 30')
sgtitle('target = 1 (trial 138th)')
str = strcat('for trial 1 (target = 1) variance of channel 1 = ', num2str(var(y1)));
disp(str)
str = strcat('for trial 1 (target = 1) variance of channel 30 = ', num2str(var(y30)));
disp(str)
%% variance feature before and after CSP
X1 = TrainData_zero_mean(:, :, p_index);
X2 = TrainData_zero_mean(:, :, n_index);
X = zeros(165, 2);
for i = 1:size(X1, 3)
    signal = X1(:, :, i);
    X(i, 1) = var(signal(1 , :)); 
    X(i, 2) = var(signal(30, :)); 
end
for i = 1:size(X2, 3)
    signal = X2(:, :, i);
    X(i+size(X1, 3), 1) = var(signal(1 , :)); 
    X(i+size(X1, 3), 2) = var(signal(30, :));
end
figure();
scatter(X(1:size(X1, 3), 1), X(1:size(X1, 3), 2), 's', 'filled')
hold on
scatter(X(size(X1, 3)+1:end, 1), X(size(X1, 3)+1:end, 2), 'o', 'filled')
xlabel('variance of channel 1' , 'Interpreter','latex')
ylabel('variance of channel 30', 'Interpreter','latex')
title('variance feature before CSP', 'Interpreter','latex')

X_new = zeros(165, 2);
for i = 1:size(X1, 3)
    signal = X1(:, :, i);
    y1 = w1'*signal;
    y30 = w30'*signal;
    X_new(i, 1) = var(y1); 
    X_new(i, 2) = var(y30);
end
for i = 1:size(X2, 3)
    signal = X2(:, :, i);
    y1 = w1'*signal;
    y30 = w30'*signal;
    X_new(i+size(X1, 3), 1) = var(signal(1 , :)); 
    X_new(i+size(X1, 3), 2) = var(signal(30, :));
end
figure();
scatter(X_new(1:size(X1, 3), 1), X_new(1:size(X1, 3), 2), 's', 'filled')
hold on
scatter(X_new(size(X1, 3)+1:end, 1), X_new(size(X1, 3)+1:end, 2), 'o', 'filled')
xlabel('variance of channel 1' , 'Interpreter','latex')
ylabel('variance of channel 30', 'Interpreter','latex')
title('variance feature after CSP', 'Interpreter','latex')
%% part b
elabels = cell(64, 1);
elocsX = cell(1, 64);
elocsY = cell(1, 64);
for i = 1:64
    elabels{i} = AllElectrodes(i).labels;
    elocsX{i} = AllElectrodes(i).X;
    elocsY{i} = AllElectrodes(i).Y;
end
% elabels = cell2mat(elabels);
elocsX = cell2mat(elocsX)';
elocsY = cell2mat(elocsY)';
used_Elecs = [37 7 5 40 38 42 10 47 45 15 13 48 50 52 18 32 55 23 22 21 20 31 57 58 59 60 26 63 27 64];
figure();
plottopomap(elocsX(used_Elecs),elocsY(used_Elecs),elabels(used_Elecs),abs(w1))
title('movement of foot imagination', 'Interpreter','latex')
figure();
plottopomap(elocsX(used_Elecs),elocsY(used_Elecs),elabels(used_Elecs),abs(w30))
title('mental subtraction', 'Interpreter','latex')
%% part c
acc = zeros(1, 15);
y = TrainLabel';
for F = 1:15
    W_CSP = [V(:, 1:F), V(:, end-F+1:end)];
    X = zeros(165, 2*F);
    for i = 1:165
        Y = W_CSP' * TrainData_zero_mean(:, :, i);
        X(i, :) = var(Y, [], 2)';
    end
    c = cvpartition(165, "kfold", 3);
    acc_avg = 0;
    for i = 1:3
        indexTrain = training(c, i);
        indexValid = test(c, i);
        XTrain = X(indexTrain, :);
        yTrain = y(indexTrain);
        XValid = X(indexValid, :);
        yValid = y(indexValid);
        SVMModel = fitcsvm(XTrain,yTrain);
        label = predict(SVMModel, XValid);
        acc_avg = acc_avg + sum(label == yValid) / length(yValid) / 3;
    end
    acc(F) = acc_avg;
end
[acc_max, F] = max(acc);
str = strcat('optimum number of filters (F) = ', num2str(F), ' and acc = ', num2str(acc_max));
disp(str)
%% part d
TestData_zero_mean = zeros(size(TestData));
for i = 1:45
    TestData_zero_mean(:, :, i) = TestData(:, :, i) - mean(TestData(:, :, i), 2);
end

W_CSP = [V(:, 1:F), V(:, end-F+1:end)];
XTest = zeros(45, 2*F);

X = zeros(165, 2*F);
for i = 1:165
    Y = W_CSP' * TrainData_zero_mean(:, :, i);
    X(i, :) = var(Y, [], 2)';
end

for i = 1:45
    Y = W_CSP' * TestData_zero_mean(:, :, i);
    XTest(i, :) = var(Y, [], 2)';
end

SVMModel = fitcsvm(X, y);
label = predict(SVMModel, X);
acc = sum(label == TrainLabel') / length(X);
disp(strcat('Train acc = ', num2str(acc)))
TestLabel = predict(SVMModel, XTest);
save('TestLabel.mat', 'TestLabel')




























