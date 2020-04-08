
% This function is used to convert a categorical column into numerical values
function new_variable = cat2binary(variable,cat_values,binary_values) 

% Get the number of observations
[rows,~] = size(variable);

% Create a new array with the same size and populate it with zeros
new_variable = zeros(rows,1);

% Assign the corresponding number to the categorical value
for i=1:length(cat_values)
    indx = ismember(variable,cat_values{i});
    new_variable(indx) = binary_values(i);
end 
end 

 