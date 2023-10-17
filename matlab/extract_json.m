clear
close all

% Pilot May 2021 - Matthew Millard
% fileName = 'calibration_WFOV_UNBINNED_720P_15fps'; % filename in JSON extension
% fileName = 'calibration_WFOV_2X2BINNED_720P_15fps'; % filename in JSON extension
% fileName = 'calibration_NFOV_UNBINNED_720P_30fps'; % filename in JSON extension
% fileName = 'calibration_NFOV_2X2BINNED_720P_15fps'; % filename in JSON extension
% fileName = 'rollator_WFOV_2X2BINNED_720P_15fps_0'; % filename in JSON extension
% fileName = 'sts_NFOV_UNBINNED_720P_30fps'; % filename in JSON extension
% fileName = 'sts_WFOV_2X2BINNED_720P_15fps'; % filename in JSON extension

% Pilot Sept 2022
% fileName = 'dynamic1_NFOV_UNBINNED_720P_30fps'; % filename in JSON extension
fileName = 'sts_NFOV_UNBINNED_720P_30fps'; % filename in JSON extension
% fileName = 'staticstand1_NFOV_UNBINNED_720P_30fps'; % filename in JSON extension
% fileName = 'rollator1_WFOV_2X2BINNED_720P_30fps'; % filename in JSON extension
% fileName = 'walkpert1_NFOV_2X2BINNED_720P_30fps_0'; % filename in JSON extension

% Pilot Oct 12th 2022
% fileName = 'stand_b2_t1_NFOV_UNBINNED_720P_30fps'; % filename in JSON extension - TO ESTIMATE VARIABILITY (after t = 10s)
% fileName = 'dynamicseq_b2_t1_NFOV_UNBINNED_720P_30fps'; % filename in JSON extension
% fileName = 'dynamicseq_b2_t1_NFOV_UNBINNED_720P_30fps_0'; % filename in JSON extension
% fileName = 'staticseq_b2_t1_NFOV_UNBINNED_720P_30fps'; % filename in JSON extension - INICIAR POR ESTE
% fileName = 'staticseq_b2_t1_NFOV_UNBINNED_720P_30fps_0'; % filename in JSON extension - INICIAR POR ESTE
% fileName = 'rollatorsts_b2_t1_WFOV_2X2BINNED_720P_30fps'; % filename in JSON extension

% Pilot Oct 13th 2022 
% fileName = 'stand_b2_s03_NFOV_UNBINNED_720P_30fps';
% fileName = 'staticseq_b2_s03_NFOV_UNBINNED_720P_30fps'; % filename in JSON extension
% fileName = 'dynamicseq_b2_s03_NFOV_UNBINNED_720P_30fps'; % filename in JSON extension
% fileName = 'rollator_b2_s03_WFOV_2X2BINNED_720P_30fps_0'; % filename in JSON extension
% fileName = 'rollatorextra_b2_s03_WFOV_2X2BINNED_720P_30fps'; % filename in JSON extension
% fileName = 'rollatorsts_b2_s03_WFOV_2X2BINNED_720P_30fps'; % filename in JSON extension

% Test data Gabriel
% fileName = 'oultput_NFOV_UNBINNED_OFF_30fps';

fid = fopen([fileName '.json']); % Opening the file
raw = fread(fid,inf); % Reading the contents
str = char(raw'); % Transformation
fclose(fid); % Closing the file
data = jsondecode(str); % Using the jsondecode function to parse JSON from string

% fileName = 'filename.json'; % filename in JSON extension 
% str = fileread(fileName); % dedicated for reading files as text 
% data = jsondecode(str); % Using the jsondecode function to parse JSON from string

% Extract data
nframes = length(data.frames);
if strcmp(fileName, 'sts_NFOV_UNBINNED_720P_30fps')
    nframes = 870; % 870
    fprintf(['Considering only the first ' num2str(nframes) ' frames of the trial. \n']);
end
if strcmp(fileName, 'sts_WFOV_2X2BINNED_720P_15fps')
    nframes = 385;
    fprintf(['Considering only the first ' num2str(nframes) ' frames of the trial. \n']);
end
fps = data.frames_per_second;
joints = NaN(nframes,32,3);
for i = 1:nframes
    if isstruct(data.frames(i).bodies)
        joints(i,:,:) = data.frames(i).bodies.joint_positions(1:32,1:3)/1000; % converting from mm to m
        orientations(i,:,:) = data.frames(i).bodies.joint_orientations(1:32,1:3);
        confidence(i) = data.frames(i).bodies.confidence_level;
        tstamp(i) = 1E-6*data.frames(i).timestamp_usec; % converting from microsecond to sec
    else
        fprintf(['was null\n'])
    end
end

%{
figure 
plot(tstamp(2:end)-tstamp(1:end-1))
pause
%}

joint_names = data.joint_names;

%% Velocity by numerical derivation
vjoints = zeros(size(joints));

for j = 1:32
    for k = 1:3
        vjoints(2:end,j,k) = (joints(2:nframes,j,k)-joints(1:nframes-1,j,k))./(tstamp(2:end)-tstamp(1:end-1))'; % backward finite differences differences
        vjoints(1,j,k) = (joints(2,j,k)-joints(1,j,k))/(tstamp(2)-tstamp(1)); % forward finite differences
    end
end

% %% reduce fps from 30 to 15
% fs = 30;
% time = tstamp(1):1/fs:tstamp(end) %+1/fs;
% nf = length(time);
% joints_r = NaN(nf,32,3);
% joints_filt = NaN(nf,32,3);
% vjoints_filt = NaN(nf,32,3);
% for j = 1:32
%     for k = 1:3
%         joints_r(:,j,k) = resample(joints(:,j,k),tstamp,fs);
%         joints_filt(:,j,k) = BwZeroFilt(joints_r(:,j,k));
%         vjoints_filt(2:end,j,k) = (joints_filt(2:end,j,k)-joints_filt(1:end-1,j,k))./(time(2:end)-time(1:end-1))'; % backward finite differences differences
%         vjoints_filt(1,j,k) = (joints_filt(2,j,k)-joints_filt(1,j,k))/(time(2)-time(1)); % forward finite differences
%     end
% end

%{
% Plot
for i = 1:32
    figure
    subplot(3,2,1)
    hold on
    plot(tstamp,joints(:,i,1))
%     plot(time,joints_filt(:,i,1),'r')
    hold off
    title(['X coord of' data.joint_names(i)])
    ylabel('x coor (m)')
    xlabel('time (s)')

    subplot(3,2,2)
    hold on
    plot(tstamp,vjoints(:,i,1))
    hold off
    title(['vX of' data.joint_names(i)])
    ylabel('vx (m/s)')
    xlabel('time (s)')

    subplot(3,2,3)
    hold on
    plot(tstamp,joints(:,i,2))
%     plot(time,joints_filt(:,i,2),'r')
    hold off
    title(['Y coord of' data.joint_names(i)])
    ylabel('y coor (m)')
    xlabel('time (s)')

    subplot(3,2,4)
    hold on
    plot(tstamp,vjoints(:,i,2)) 
    hold off
    title(['vY of' data.joint_names(i)])
    ylabel('vy (m/s)')
    xlabel('time (s)')

    subplot(3,2,5)
    hold on
    plot(tstamp,joints(:,i,3))
%     plot(time,joints_filt(:,i,3),'r')
    hold off
    title(['Z coord of' data.joint_names(i)])
    ylabel('z coor (m)')
    xlabel('time (s)')

    subplot(3,2,6)
    hold on
    plot(tstamp,vjoints(:,i,3))
    hold off
    title(['vZ of' data.joint_names(i)])
    ylabel('vz (m/s)')
    xlabel('time (s)')
    pause
end  
%}

save(fileName,'joints','orientations','confidence','tstamp','joint_names','vjoints','fps');


function output  = BwZeroFilt(input)
fs = 30; %  [Hz] samplng rate of the input data
Wn = 6/(fs/2); % corresponding to a cut off frequency of 6 Hz;
[B,A] = butter(4, Wn, 'low');
output = filtfilt(B,A,input);
end


