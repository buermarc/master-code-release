labels = {};

for i = 1:32
    labels = [labels {["Joint_" i "_x"]}];
    labels = [labels {["Joint_" i "_y"]}];
    labels = [labels {["Joint_" i "_z"]}];
end
labels