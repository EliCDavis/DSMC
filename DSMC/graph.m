data = load("output.txt");
X = data(:,1);
Y = data(:,2);
Z = data(:,3);

scatter3(X(1:2:end),Y(1:2:end), Z(1:2:end));