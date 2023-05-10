function [var] = getMeasNoise

% Estimate noise error from standing still data for subject t1
fileName = 'stand_b2_t1_NFOV_UNBINNED_720P_30fps.json'; % filename in JSON extension - TO ESTIMATE VARIABILITY (after t = 10s)

fprintf(['Using data from ' fileName ' to estimate measurement noise. \n']);

fid = fopen(['..\Experiment_data\' fileName]); % Opening the file
raw = fread(fid,inf); % Reading the contents
str = char(raw'); % Transformation
fclose(fid); % Closing the file
data = jsondecode(str); % Using the jsondecode function to parse JSON from string

% Extract data
nframes = length(data.frames);
fps = data.frames_per_second;
joints = NaN(nframes,32,3);
var = NaN(32,3);
avg = NaN(32,3);
for i = 1:nframes
    if isstruct(data.frames(i).bodies)
        joints(i,:,:) = data.frames(i).bodies.joint_positions(1:32,1:3)/1000; % converting from mm to m
        tstamp(i) = 1E-6*data.frames(i).timestamp_usec; % convert from microsec to sec
    end
end
i_i = 210; % beginning of the period to be considered
i_f = 340; % end of the period to considered
fprintf(['Using time span between tstamp = ' num2str(tstamp(i_i)) ' : ' num2str(tstamp(i_f)) ' s \n']);
span = i_i:i_f;
% plot(i_i:i_f, tstamp(span));
npoints = length(span);
for j = 1:32
    for k = 1:3
        avg(j,k) = mean(joints(span,j,k));
        var(j,k) = sum((joints(span,j,k) - avg(j,k)).^2)/(npoints-1);
        
%         plot(tstamp(span),joints(span,j,k));
%         title([data.joint_names(j) ': mean = ' num2str(avg(j,k)) ' m; var = ' num2str(var(j,k)) ' m^2'])
%         pause
    end
end
fprintf(['The estimated measurement noise variances (in m^2) are: \n']);
var




