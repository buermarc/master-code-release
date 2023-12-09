%This script:
% - computes joint positions and velocities from raw Kinect data using a Kalman Filter with a gating approach
% - computes joint positions and velocities from raw Kinect data using a Kalman Filter including distance contraints on velocity level for legs (hip, knee, ankle) and arms (shoulder, elbow and wrist) 
% References: 
% [1] Machthaler und Dingler, "Kalman-Filter: Einfuehrung in die Zustandsschaetzunge und ihre Anwendung fuer eingebettete System", 2017.
% [2] Sabatini, A. M., "Real-time Kalman filter applied to biomechanicaldata for state estimation and numerical differentiation", Medical & Biology Engineering & Computing, Vol. 41, 2003.
% [3] Gupta, N.; Hauser, R., "Kalman filtering with equality and inequality state constraints", Report 07/18, Oxford University Compputing Laboratory, 2007.
% Author: Marko Ackermann
% Last update: March 22nd, 2023

% Chooose approach
% Refer to [1] (section 7.5. Kinematische Modelle) 
% 1 - second-order approach with direct discretization - variation of velocity by Gaussian with zero mean
% 2 - second-order approach  with reale Abtastung (Abtast-Halte-Glied) - variation of acceleration by normal distribution with zero mean
ap = 2; 

% Choose trashhold for the normalized innovation  for rejection of outliers
% Refer to [2] (section 2.3. Gaiting technique for outlier detection and rejection)
trashn = 2; % Trashhold for normalized innovation

%% Load experimental data to filter
% Pilot May 2021 - Matthew Millard
% fileName = 'calibration_WFOV_UNBINNED_720P_15fps'; % filename in JSON extension
% fileName = 'rollator_WFOV_2X2BINNED_720P_15fps_0'; % filename in JSON extension
fileName = 'sts_NFOV_UNBINNED_720P_30fps'; % filename in JSON extension - BEST FOR TESTING
% fileName = 'sts_WFOV_2X2BINNED_720P_15fps'; % filename in JSON extension - GOOD
% fileName = 'calibration_NFOV_2X2BINNED_720P_15fps'; % filename in JSON extension

% Pilot Sept 2022
% fileName = 'dynamic1_NFOV_UNBINNED_720P_30fps'; % filename in JSON extension
% fileName = 'sts1_NFOV_UNBINNED_720P_30fps'; % filename in JSON extension
% fileName = 'staticstand1_NFOV_UNBINNED_720P_30fps'; % filename in JSON extension
% fileName = 'rollator1_WFOV_2X2BINNED_720P_30fps'; % filename in JSON extension
% fileName = 'walkpert1_NFOV_2X2BINNED_720P_30fps_0'; % filename in JSON extension

% Pilot Oct 12th 2022
% fileName = 'stand_b2_t1_NFOV_UNBINNED_720P_30fps'; % filename in JSON extension - TO ESTIMATE VARIABILITY (after t = 10s)
% fileName = 'dynamicseq_b2_t1_NFOV_UNBINNED_720P_30fps'; % filename in JSON extension
% fileName = 'dynamicseq_b2_t1_NFOV_UNBINNED_720P_30fps_0'; % filename in JSON extension
% fileName = 'staticseq_b2_t1_NFOV_UNBINNED_720P_30fps'; % filename in JSON extension
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
% fileName = 'output_NFOV_UNBINNED_OFF_30fps';

load(fileName);
fprintf(['Loading experimental data in ' fileName '... \n']);

%% Choose joint
% 1 'PELVIS'
% 2 'SPINE_NAVEL'
% 3 'SPINE_CHEST'
% 4 'NECK'
% 5 'CLAVICLE_LEFT'
% 6 'SHOULDER_LEFT'
% 7 'ELBOW_LEFT'
% 8 'WRIST_LEFT'
% 9 'HAND_LEFT'
% 10'HANDTIP_LEFT'
% 11'THUMB_LEFT'
% 12'CLAVICLE_RIGHT'
% 13'SHOULDER_RIGHT'
% 14'ELBOW_RIGHT'
% 15'WRIST_RIGHT'
% 16'HAND_RIGHT'
% 17'HANDTIP_RIGHT'
% 18'THUMB_RIGHT'
% 19'HIP_LEFT'
% 20'KNEE_LEFT'
% 21'ANKLE_LEFT'
% 22'FOOT_LEFT'
% 23'HIP_RIGHT'
% 24'KNEE_RIGHT'
% 25'ANKLE_RIGHT'
% 26'FOOT_RIGHT'
% 27'HEAD'
% 28'NOSE'
% 29'EYE_LEFT'
% 30'EAR_LEFT'
% 31'EYE_RIGHT'
% 32'EAR_RIGHT'  

%% Choose coordinate
% 1: x - lateral (positive to the left of a person looking at the camera)
% 2: y - vertical (positive downwards)
% 3: z - depth (positive from camera)

% Indexes considered for each leg and arm
jind = [19 20 21; ... % left leg
        23 24 25; ... % right leg
        6  7  8;  ... % left arm
        13 14 15];    % right arm

% jind_LL = [19 20 21]; % left leg
% jind_LR = [23 24 25]; % right leg
% jind_AL = [6 7 8];    % left arm
% jind_AR = [13 14 15]; % right arm
% jind = [1:5 9:12 16:18 22 26:32]; % remaining joints

% figure
% for j = 1:3
%     % Plot data to be filtered
%     figure
%     subplot(3,2,1)
%     hold on
%     plot(tstamp,joints(:,jind(j),1));
%     hold off
%     legend('unfiltered')
%     title(['Coord x (lateral) of joint ' joint_names{jind(j)}])
%     ylabel('x (m)')
%     xlabel('time (s)')
%     box on
%     
%     subplot(3,2,2)
%     hold on
%     plot(tstamp,vjoints(:,jind(j),1))
%     hold off
%     legend('unfiltered')
%     title(['Velocity x of joint ' joint_names{jind(j)}])
%     ylabel('vx (m/s)')
%     xlabel('time (s)')
%     box on
%     
%     subplot(3,2,3)
%     hold on
%     plot(tstamp,joints(:,jind(j),2));
%     hold off
%     legend('unfiltered')
%     title(['Coord y (vertical) of joint ' joint_names{jind(j)}])
%     ylabel('y (m)')
%     xlabel('time (s)')
%     box on
%     
%     subplot(3,2,4)
%     hold on
%     plot(tstamp,vjoints(:,jind(j),2))
%     hold off
%     legend('unfiltered')
%     title(['Velocity y of joint ' joint_names{jind(j)}])
%     ylabel('vy (m/s)')
%     xlabel('time (s)')
%     box on
%     
%     subplot(3,2,5)
%     hold on
%     plot(tstamp,joints(:,jind(j),3))
%     hold off
%     legend('unfiltered')
%     title(['Coord z (depth) of joint ' joint_names{jind(j)}])
%     ylabel('z (m)')
%     xlabel('time (s)')
%     box on
%     
%     subplot(3,2,6)
%     hold on
%     plot(tstamp,vjoints(:,jind(j),3))
%     hold off
%     legend('unfiltered')
%     title(['Velocity z of joint ' joint_names{jind(j)}])
%     ylabel('vz (m/s)')
%     xlabel('time (s)')
%     box on
% end

% figure
% 
%     % Plot data to be filtered
%     figure
%     subplot(3,3,1)
%     hold on
%     plot(tstamp,joints(:,jind(1),1));
%     hold off
%     legend('unfiltered')
%     title(['Coord x (lateral) of joint ' joint_names{jind(1)}])
%     ylabel('x (m)')
%     xlabel('time (s)')
%     box on
%     
%     subplot(3,3,2)
%     hold on
%     plot(tstamp,joints(:,jind(2),1))
%     hold off
%     legend('unfiltered')
%     title(['Coord x (lateral) of joint  ' joint_names{jind(2)}])
%     ylabel('x (m)')
%     xlabel('time (s)')
%     box on
%     
%     subplot(3,3,3)
%     hold on
%     plot(tstamp,joints(:,jind(3),1));
%     hold off
%     legend('unfiltered')
%     title(['Coord x (lateral) of joint ' joint_names{jind(3)}])
%     ylabel('x (m)')
%     xlabel('time (s)')
%     box on
%     
%     subplot(3,3,4)
%     hold on
%     plot(tstamp,joints(:,jind(1),2));
%     hold off
%     legend('unfiltered')
%     title(['Coord y (vertical) of joint ' joint_names{jind(1)}])
%     ylabel('y (m)')
%     xlabel('time (s)')
%     box on
%     
%     subplot(3,3,5)
%     hold on
%     plot(tstamp,joints(:,jind(2),2))
%     hold off
%     legend('unfiltered')
%     title(['Coord y (vertical) of joint  ' joint_names{jind(2)}])
%     ylabel('y (m)')
%     xlabel('time (s)')
%     box on
%     
%     subplot(3,3,6)
%     hold on
%     plot(tstamp,joints(:,jind(3),2));
%     hold off
%     legend('unfiltered')
%     title(['Coord y (vertical) of joint ' joint_names{jind(3)}])
%     ylabel('y (m)')
%     xlabel('time (s)')
%     box on
% 
%     subplot(3,3,7)
%     hold on
%     plot(tstamp,joints(:,jind(1),3));
%     hold off
%     legend('unfiltered')
%     title(['Coord z (depth) of joint ' joint_names{jind(1)}])
%     ylabel('z (m)')
%     xlabel('time (s)')
%     box on
%     
%     subplot(3,3,8)
%     hold on
%     plot(tstamp,joints(:,jind(2),3))
%     hold off
%     legend('unfiltered')
%     title(['Coord z (depth) of joint  ' joint_names{jind(2)}])
%     ylabel('z (m)')
%     xlabel('time (s)')
%     box on
%     
%     subplot(3,3,9)
%     hold on
%     plot(tstamp,joints(:,jind(3),3));
%     hold off
%     legend('unfiltered')
%     title(['Coord z (depth) of joint ' joint_names{jind(3)}])
%     ylabel('z (m)')
%     xlabel('time (s)')
%     box on

%% System dynamics continuous
% For each coordinate
% xdot = A.x + (B.u) + G.z
% y = C.x + (D.u)

A = [0 1; 0 0];


B = [0; 0];

G = [0; 1];

C = [1 0];

d = length(A); % system dimension

%% System dynamics continuous legs and arms
% For the legs and arms (with three joints each)
% xdot = Al.x + (Bl.u) + Gl.z
% y = Cl.x + (Dl.u)
% x = [p_hip; p_knee; p_ankle; v_hip; v_knee; v_ankle]
% with: p_i = [xi; yi; zi]
%       v_i = [vxi; vyi; vzi]

Al = [zeros(9,9) eye(9); zeros(9,9) zeros(9,9)];

Gl = [zeros(9,9); eye(9,9)];

Cl = [eye(9,9) zeros(9,9)];

dl = length(Al); % system dimension


%% Discrete system dynamics
% x(k+1) = Ad.x(k) + Gd.z(k)
% y(k) = C.x(k)
% with: Ts - sample time
syms Ts 'real'
Ad = simplify(expm(A*Ts));
Bd = simplify(int(expm(Ts*A)*B));

if ap == 1
    Gd = Ad*G; % considering that velocity has normal distribution with zero mean
elseif ap == 2
%     Gd = [Ts^2/2; Ts]; % considering that acceleration has normal ditribution with zero mean, See truck example in wikipedia and Machthaler und Dingler (2017)
    Gd = simplify(int(expm(Ts*A)*G)); % fuer reale Abtastung (Abtast-Halte-Glied)
else
    error('ap must be either 1 or 2.')
end

%% Discrete system dynamics leg
% x(k+1) = Ad.x(k) + ( Bb.u(k) ) + Gd.z(k)
% y(k) = C.x(k) + ( D.u(k) )
% with: Ts - sample time
syms Ts 'real'
Ald = simplify(expm(Al*Ts));
% Bld = simplify(int(expm(Ts*Al)*Bl));

if ap == 1
    Gld = Ald*Gl; % considering that velocity has normal distribution with zero mean
elseif ap == 2
    Gld = [(Ts^2/2)*eye(9,9); Ts*eye(9,9)]; % considering that acceleration has normal ditribution with zero mean, See truck example in wikipedia and Machthaler und Dingler (2017)
else
    error('ap must be either 1 or 2.')
end

%% Determine observabilty
% Pg. 11 Machthaler und Dingler (2017)
SB=C;
for n=1:length(A)-1
    SB = [SB; C*Ad^n];
end

ra = rank(SB);

if ra == d
    fprintf('This system is observable. \n');
elseif ra < d
    fprintf('This system is NOT observable. \n');
else
    error('');
end

%% Determine observability legs and arms
% Pg. 11 Machthaler und Dingler (2017)
SBl=Cl;
for n=1:length(Al)-1
    SBl = [SBl; Cl*Ald^n];
end

ral = rank(SBl);

if ral == dl
    fprintf('This system is observable. \n');
elseif ral < dl
    fprintf('This system is NOT observable. \n');
else
    error('');
end

%% Data structure
% body.NF - nonfiltered data
% body.KF - filtered with Kalman filter and gating
% body.KFC - filtered with Kalman filter and constraints on amrs and legs lengths

body.tstamp = tstamp;       % time instants vector
body.trashn = trashn;       % trashhold for normilized innovation for outlier rejection
body.ap = ap;               % approch choice: 1 - first order (null mean velocity), 2 - second order (null mean acceleration)
body.NF.joints = joints;    % original unfiltered joint positions
body.NF.vjoints = vjoints;  % original unfiltered joint velocities (backward finite differences
body.fileName = fileName;   % file name of data file
body.KF.Ad = Ad;            % discretized system matrix in xdot = A.x + B.u + G.z
body.KF.Bd = Bd;            % discretized B matrix in xdot = A.x + B.u + G.z
body.KF.Gd = Gd;            % discretized G matrix in xdot = A.x + B.u + G.z
body.KF.C = C;              % C matrix in y = C.x + D.u
% body.KF.D = D;            % D matrix in y = C.x + D.u
body.KFC.Ald = Ald;         % discretized system matrix in xdot = A.x + B.u + G.z
% body.KFC..Bld = Bld;      % discretized B matrix in xdot = A.x + B.u + G.z
body.KFC.Gld = Gld;         % discretized G matrix in xdot = A.x + B.u + G.z
body.KFC.Cl = Cl;           % C matrix in y = C.x + D.u


%% System Noise
% Filter data and compute max velocity for approach 1 or max acceleration for
% approach 2.
if ap == 1
    factorSN = 1/20; % 1/5 factorSN = 1/20 and factorMN = 100 gave good results
    vmax = 2*factorSN; % [m/s] - for approach 1 - max observed in staticseq
    sigmaSN = vmax/3; % assuming max will be exceeded for 3 standard deviations (0,3%) - approach 1
else % ap == 2
    factorSN = 1/3; % 1/5 factorSN = 1/20 and factorMN = 100 gave good results
    amax = 10*factorSN; % [m/s2] - for approach 2 - base for 1G
    sigmaSN = amax/3; % assuming max will be exceeded for 3 standard deviations (0,3%) - approach 2
end

% System noise covariance matrix
Q_x = sigmaSN^2;
Q_y = sigmaSN^2;
Q_z = sigmaSN^2;

% System noise covariance matrix leg
Ql = eye(9,9)*sigmaSN^2;

%% Measurement Noise
% Extract the measurement noise for the standing quite position
factorMN = 10; % 100
var = getMeasNoise.*[ones(32,1), ones(32,1), ones(32,1)./10];
sigmaMN = factorMN*(var).^0.5; % compute standard deviation assuming Normal distribution

% Legs and Arms
factorMNl = 10; % 100
% sigmaMNl_LL = factorMNl*sqrt([var(jind_LL(1),:)'; var(jind_LL(2),:)'; var(jind_LL(3),:)']); % leg left
% Rl_LL = diag(sigmaMNl_LL.^2);
% sigmaMNl_LR = factorMNl*sqrt([var(jind_LR(1),:)'; var(jind_LR(2),:)'; var(jind_LR(3),:)']); % leg right
% Rl_LR = diag(sigmaMNl_LR.^2);
% sigmaMNl_AL = factorMNl*sqrt([var(jind_AL(1),:)'; var(jind_AL(2),:)'; var(jind_AL(3),:)']); % arm left
% Rl_AL = diag(sigmaMNl_AL.^2);
% sigmaMNl_AR = factorMNl*sqrt([var(jind_AR(1),:)'; var(jind_AR(2),:)'; var(jind_AR(3),:)']); % arm right
% Rl_AR = diag(sigmaMNl_AR.^2);

fprintf('Starting individual KF with gating ... \n')

% Prealocation for individual filtering
xp_x = NaN(2,length(joints(:,1,1)));  % predicted state
xc_x = NaN(2,length(joints(:,1,1)));   % corrected state
Pp_x = NaN(2,2,length(joints(:,1,1))); % predicted error covariance matrix
Pc_x = NaN(2,2,length(joints(:,1,1))); % corrected error covariance matrix
K_x = NaN(2,length(joints(:,1,1)));
v_x = NaN(1,length(joints(:,1,1))); % innovation value
S_x = NaN(1,length(joints(:,1,1))); % innovation covariance matrix (as in Sabatini, 2003)
xp_y = NaN(2,length(joints(:,1,1)));  % predicted state
xc_y = NaN(2,length(joints(:,1,1)));   % corrected state
Pp_y = NaN(2,2,length(joints(:,1,1))); % predicted error covariance matrix
Pc_y = NaN(2,2,length(joints(:,1,1))); % corrected error covariance matrix
K_y = NaN(2,length(joints(:,1,1)));    
v_y = NaN(1,length(joints(:,1,1))); % innovation value
S_y = NaN(1,length(joints(:,1,1))); % innovation covariance matrix (as in Sabatini, 2003)
xp_z = NaN(2,length(joints(:,1,1)));  % predicted state
xc_z = NaN(2,length(joints(:,1,1)));   % corrected state
Pp_z = NaN(2,2,length(joints(:,1,1))); % predicted error covariance matrix
Pc_z = NaN(2,2,length(joints(:,1,1))); % corrected error covariance matrix
K_z = NaN(2,length(joints(:,1,1)));
v_z = NaN(1,length(joints(:,1,1))); % innovation value
S_z = NaN(1,length(joints(:,1,1))); % innovation covariance matrix (as in Sabatini, 2003)

for j = 1:32

    % Measurement covariance matrix
    R_x = sigmaMN(j,1)^2;
    R_y = sigmaMN(j,2)^2;
    R_z = sigmaMN(j,3)^2;
    
    %% Initialize Kalman Filter
    %%% INITIALISIERUNG KALMAN-FILTER %%%
    % x
    y_x = joints(:,j,1);      % measurement vector    
    xc_x(:,1) = [y_x(1); 0];    % initializaton for corrected state
    Pc_x(:,:,1) = [1 0; 0 1]; % initialization for corrected errors covariance matrix
    
    % y
    y_y = joints(:,j,2);      % measurement vector
    xc_y(:,1) = [y_y(1); 0];    % initializaton for corrected state
    Pc_y(:,:,1) = [1 0; 0 1]; % initialization for corrected erros covariance matrix
    
    % z
    y_z = joints(:,j,3);      % measurement vector
    xc_z(:,1) = [y_z(1); 0];    % initializaton for corrected state
    Pc_z(:,:,1) = [1 0; 0 1]; % initialization for corrected erros covariance matrix
    
    
    %% Kalman Filter
    for k = 2:length(joints(:,j,1))
        Ts = tstamp(k) - tstamp(k-1); % Current time step
    
        % Converting symbolic to double
        Adn = double(subs(Ad,Ts));
        Gdn = double(subs(Gd,Ts));
    
        % Prediction step
        xp_x(:,k) = Adn*xc_x(:,k-1);
        Pp_x(:,:,k) = Adn*Pc_x(:,:,k-1)*Adn' + Gdn*Q_x*Gdn';
        xp_y(:,k) = Adn*xc_y(:,k-1);
        Pp_y(:,:,k) = Adn*Pc_y(:,:,k-1)*Adn' + Gdn*Q_y*Gdn';
        xp_z(:,k) = Adn*xc_z(:,k-1);
        Pp_z(:,:,k) = Adn*Pc_z(:,:,k-1)*Adn' + Gdn*Q_z*Gdn';
    
        % Innovation
        vi_x(k) = y_x(k) - C*xp_x(:,k); % innovation
        vi_y(k) = y_y(k) - C*xp_y(:,k); % innovation
        vi_z(k) = y_z(k) - C*xp_z(:,k); % innovation
        Si_x(k) = C*Adn*Pp_x(:,:,k)*Adn'*C' + C*Q_x*C' + R_x; % covariance matrix of the innovation as n Eq. (5) of Sabatini (2003)
        Si_y(k) = C*Adn*Pp_y(:,:,k)*Adn'*C' + C*Q_y*C' + R_y; % covariance matrix of the innovation as n Eq. (5) of Sabatini (2003)
        Si_z(k) = C*Adn*Pp_z(:,:,k)*Adn'*C' + C*Q_z*C' + R_z; % covariance matrix of the innovation as n Eq. (5) of Sabatini (2003)
        sigmaSi_x = sqrt(Si_x(k)); % only valid if covariance matrix Si is a scalar
        sigmaSi_y = sqrt(Si_y(k)); % only valid if covariance matrix Si is a scalar
        sigmaSi_z = sqrt(Si_z(k)); % only valid if covariance matrix Si is a scalar
        vinorm_x = vi_x(k)/sigmaSi_x;
        vinorm_y = vi_y(k)/sigmaSi_y;
        vinorm_z = vi_z(k)/sigmaSi_z;
    
        if all([abs(vinorm_x); abs(vinorm_y); abs(vinorm_z) ] < trashn)
            % Correction step
            K_x(:,k) = Pp_x(:,:,k)*C'*pinv(C*Pp_x(:,:,k)*C' + R_x);
            K_y(:,k) = Pp_y(:,:,k)*C'*pinv(C*Pp_y(:,:,k)*C' + R_y);
            K_z(:,k) = Pp_z(:,:,k)*C'*pinv(C*Pp_z(:,:,k)*C' + R_z);
            xc_x(:,k) = xp_x(:,k) + K_x(:,k)*(y_x(k)' - C*xp_x(:,k));
            xc_y(:,k) = xp_y(:,k) + K_y(:,k)*(y_y(k)' - C*xp_y(:,k));
            xc_z(:,k) = xp_z(:,k) + K_z(:,k)*(y_z(k)' - C*xp_z(:,k));
            Pc_x(:,:,k) = (eye(2) - K_x(:,k)*C)*Pp_x(:,:,k);
            Pc_y(:,:,k) = (eye(2) - K_y(:,k)*C)*Pp_y(:,:,k);
            Pc_z(:,:,k) = (eye(2) - K_z(:,k)*C)*Pp_z(:,:,k);
        else % rejection of outlier as gaiting strategy
            xc_x(:,k) = xp_x(:,k); % Do not correct predicted value
            Pc_x(:,:,k) = Pp_x(:,:,k); % freeze update of error covariance matrix
            xc_y(:,k) = xp_y(:,k); % Do not correct predicted value
            Pc_y(:,:,k) = Pp_y(:,:,k); % freeze update of error covariance matrix
            xc_z(:,k) = xp_z(:,k); % Do not correct predicted value
            Pc_z(:,:,k) = Pp_z(:,:,k); % freeze update of error covariance matrix
        end
    end
    
    body.KF.joint{j}.coord{1}.sigmaSN = sigmaSN;
    body.KF.joint{j}.coord{1}.Q = Q_x;
    body.KF.joint{j}.coord{1}.sigmaMN = sigmaMN(j,1);
    body.KF.joint{j}.coord{1}.R = R_x;
    body.KF.joint{j}.coord{1}.xp = xp_x;
    body.KF.joint{j}.coord{1}.Pp = Pp_x;
    body.KF.joint{j}.coord{1}.vi = vi_x;
    body.KF.joint{j}.coord{1}.Si = Si_x;
    body.KF.joint{j}.coord{1}.K = K_x;
    body.KF.joint{j}.coord{1}.xc = xc_x;
    body.KF.joint{j}.coord{1}.Pc = Pc_x;

    body.KF.joint{j}.coord{2}.sigmaSN = sigmaSN;
    body.KF.joint{j}.coord{2}.Q = Q_y;
    body.KF.joint{j}.coord{2}.sigmaMN = sigmaMN(j,2);
    body.KF.joint{j}.coord{2}.R = R_y;
    body.KF.joint{j}.coord{2}.xp = xp_y;
    body.KF.joint{j}.coord{2}.Pp = Pp_y;
    body.KF.joint{j}.coord{2}.vi = vi_y;
    body.KF.joint{j}.coord{2}.Si = Si_y;
    body.KF.joint{j}.coord{2}.K = K_y;
    body.KF.joint{j}.coord{2}.xc = xc_y;
    body.KF.joint{j}.coord{2}.Pc = Pc_y;

    body.KF.joint{j}.coord{3}.sigmaSN = sigmaSN;
    body.KF.joint{j}.coord{3}.Q = Q_z;
    body.KF.joint{j}.coord{3}.sigmaMN = sigmaMN(j,3);
    body.KF.joint{j}.coord{3}.R = R_z;
    body.KF.joint{j}.coord{3}.xp = xp_z;
    body.KF.joint{j}.coord{3}.Pp = Pp_z;
    body.KF.joint{j}.coord{3}.vi = vi_z;
    body.KF.joint{j}.coord{3}.Si = Si_z;
    body.KF.joint{j}.coord{3}.K = K_z;
    body.KF.joint{j}.coord{3}.xc = xc_z;
    body.KF.joint{j}.coord{3}.Pc = Pc_z;
   
%     % Compare filtered with non-filteres data
%     figure
%     subplot(3,2,1)
%     hold on
%     plot(tstamp,joints(:,j,1))
%     plot(tstamp,xc_x(1,:),'r','LineWidth',1)
%     hold off
%     legend('unfiltered','KF with gating')
%     title(['Coord x (lateral) of joint ' joint_names{j}])
%     ylabel('x (m)')
%     xlabel('time (s)')
%     box on
%     
%     subplot(3,2,2)
%     hold on
%     plot(tstamp,vjoints(:,j,1))
%     plot(tstamp,xc_x(2,:),'r','LineWidth',1)
%     hold off
%     legend('unfiltered','KF with gating')
%     title(['Velocity x of joint ' joint_names{j}])
%     ylabel('vx (m/s)')
%     xlabel('time (s)')
%     box on
%     
%     subplot(3,2,3)
%     hold on
%     plot(tstamp,joints(:,j,2))
%     plot(tstamp,xc_y(1,:),'r','LineWidth',1)
%     hold off
%     legend('unfiltered','KF with gating')
%     title(['Coord y (vertical) of joint ' joint_names{j}])
%     ylabel('y (m)')
%     xlabel('time (s)')
%     box on
%     
%     subplot(3,2,4)
%     hold on
%     plot(tstamp,vjoints(:,j,2))
%     plot(tstamp,xc_y(2,:),'r','LineWidth',1)
%     hold off
%     legend('unfiltered','KF with gating')
%     title(['Velocity y of joint ' joint_names{j}])
%     ylabel('vy (m/s)')
%     xlabel('time (s)')
%     box on
%     
%     subplot(3,2,5)
%     hold on
%     plot(tstamp,joints(:,j,3))
%     plot(tstamp,xc_z(1,:),'r','LineWidth',1)
%     hold off
%     legend('unfiltered','KF with gating')
%     title(['Coord z (depth) of joint ' joint_names{j}])
%     ylabel('z (m)')
%     xlabel('time (s)')
%     box on
%     
%     subplot(3,2,6)
%     hold on
%     plot(tstamp,vjoints(:,j,3))
%     plot(tstamp,xc_z(2,:),'r','LineWidth',1)
%     hold off
%     legend('unfiltered','KF with gating')
%     title(['Velocity z of joint ' joint_names{j}])
%     ylabel('vz (m/s)')
%     xlabel('time (s)')
%     box on
%     j
end

% %% Compare individually KF filtered with non-filtered data
% for j = 1:32
%     figure
%     subplot(3,2,1)
%     hold on
%     plot(tstamp,body.NF.joints(:,j,1))
%     plot(tstamp,body.KF.joint{j}.coord{1}.xc(1,:),'r','LineWidth',1)
%     hold off
%     legend('unfiltered','KF with gating')
%     title(['Coord x (lateral) of joint ' joint_names{j}])
%     ylabel('x (m)')
%     xlabel('time (s)')
%     box on
%     
%     subplot(3,2,2)
%     hold on
%     plot(tstamp,body.NF.vjoints(:,j,1))
%     plot(tstamp,body.KF.joint{j}.coord{1}.xc(2,:),'r','LineWidth',1)
%     hold off
%     legend('unfiltered','KF with gating')
%     title(['Velocity x of joint ' joint_names{j}])
%     ylabel('vx (m/s)')
%     xlabel('time (s)')
%     box on
%     
%     subplot(3,2,3)
%     hold on
%     plot(tstamp,body.NF.joints(:,j,2))
%     plot(tstamp,body.KF.joint{j}.coord{2}.xc(1,:),'r','LineWidth',1)
%     hold off
%     legend('unfiltered','KF with gating')
%     title(['Coord y (vertical) of joint ' joint_names{j}])
%     ylabel('y (m)')
%     xlabel('time (s)')
%     box on
%     
%     subplot(3,2,4)
%     hold on
%     plot(tstamp,body.NF.vjoints(:,j,2))
%     plot(tstamp,body.KF.joint{j}.coord{2}.xc(2,:),'r','LineWidth',1)
%     hold off
%     legend('unfiltered','KF with gating')
%     title(['Velocity y of joint ' joint_names{j}])
%     ylabel('vy (m/s)')
%     xlabel('time (s)')
%     box on
%     
%     subplot(3,2,5)
%     hold on
%     plot(tstamp,body.NF.joints(:,j,3))
%     plot(tstamp,body.KF.joint{j}.coord{3}.xc(1,:),'r','LineWidth',1)
%     hold off
%     legend('unfiltered','KF with gating')
%     title(['Coord z (depth) of joint ' joint_names{j}])
%     ylabel('z (m)')
%     xlabel('time (s)')
%     box on
%     
%     subplot(3,2,6)
%     hold on
%     plot(tstamp,body.NF.vjoints(:,j,3))
%     plot(tstamp,body.KF.joint{j}.coord{3}.xc(2,:),'r','LineWidth',1)
%     hold off
%     legend('unfiltered','KF with gating')
%     title(['Velocity z of joint ' joint_names{j}])
%     ylabel('vz (m/s)')
%     xlabel('time (s)')
%     box on
% end


%% Kalman filter for legs and arms

fprintf('Starting KF with constraints ... \n')

% Prealocation
xp = NaN(18,length(joints(:,1,1)));  % predicted state
xc = NaN(18,length(joints(:,1,1)));   % corrected state
Pp = NaN(18,18,length(joints(:,1,1))); % predicted error covariance matrix
Pc = NaN(18,18,length(joints(:,1,1))); % corrected error covariance matrix
xcp = NaN(18,length(joints(:,1,1)));   % projected state after correction step
Pcp = NaN(18,18,length(joints(:,1,1))); % projected error covariance matrix after correction
K = NaN(18,9,length(joints(:,1,1)));
% v = NaN(1,length(joints(:,1,1))); % innovation value
% S = NaN(1,length(joints(:,1,1))); % innovation covariance matrix (as in Sabatini, 2003)

% Initialize Kalman Filter

% For buiding the constraint matrices later
% Refer to Notes\KF_contraints_01.jpg to KF_contraints_04.jpg
phi1 = [eye(3),    -eye(3),     zeros(3,3); ...
        -eye(3),    eye(3),     zeros(3,3); ...
        zeros(3,3), zeros(3,3), zeros(3,3)];

phi2 = [zeros(3,3), zeros(3,3), zeros(3,3);
        zeros(3,3), eye(3),     -eye(3); ...
        zeros(3,3), -eye(3),    eye(3)]; 

% Copying joint information from KF to KFC before repopulating joints which are part of the chains
body.KFC.joint = body.KF.joint; 

for i = 1:size(jind,1) % for each chain
    % Measurement noise
    sigmaMNl = factorMNl*sqrt([var(jind(i,1),:)'; var(jind(i,2),:)'; var(jind(i,3),:)']); 
    Rl = diag(sigmaMNl.^2);

    % Measurement vector: y = [px1; py1; pz1; px2; py2; pz2; px3; py3; pz3]
    % Leg: 1 - hip;      2 - knee;       3 - ankle
    % Arm: 1 - shoulder  2 - elbow;      3 - wrist
    y = [joints(:,jind(i,1),1)'; joints(:,jind(i,1),2)'; joints(:,jind(i,1),3)'; ...
         joints(:,jind(i,2),1)'; joints(:,jind(i,2),2)'; joints(:,jind(i,2),3)'; ...
         joints(:,jind(i,3),1)'; joints(:,jind(i,3),2)'; joints(:,jind(i,3),3)']; 
    
    % Initialization for first time instant
    xc(:,1) = [y(:,1); zeros(9,1)];     % initializaton for corrected state (measured positions, zero velocities)
    Pc(:,:,1) = eye(18,18);             % initialization for corrected errors covariance matrix
    xcp(:,1) = [y(:,1); zeros(9,1)];    % initializaton for projected corrected state (measured positions, zero velocities)
    Pcp(:,:,1) = eye(18,18);            % initialization for projected corrected errors covariance matrix

    for k = 2:length(joints(:,1,1))
        Ts = tstamp(k) - tstamp(k-1); % Current time step
    
        % Converting symbolic to double
        Aldn = double(subs(Ald,Ts));
        Gldn = double(subs(Gld,Ts));
    
        % Prediction step
    %     xp(:,k) = Aldn*xc(:,k-1);
    %     Pp(:,:,k) = Aldn*Pc(:,:,k-1)*Aldn' + Gldn*Ql*Gldn';
        xp(:,k) = Aldn*xcp(:,k-1);
        Pp(:,:,k) = Aldn*Pcp(:,:,k-1)*Aldn' + Gldn*Ql*Gldn';
    
        % Correction step
        K(:,:,k) = Pp(:,:,k)*Cl'*pinv(Cl*Pp(:,:,k)*Cl' + Rl);
        xc(:,k) = xp(:,k) + K(:,:,k)*(y(:,k) - Cl*xp(:,k));
        Pc(:,:,k) = (eye(18) - K(:,:,k)*Cl)*Pp(:,:,k);
    
        % Projection step
        phi = [xc(1:9,k)'*phi1; xc(1:9,k)'*phi2];
        xcp(:,k) = [eye(9), zeros(9,9); zeros(9,9), (eye(9)-phi'*inv(phi*phi')*phi)]*xc(:,k);
        Pcp(:,:,k) = [eye(9), zeros(9,9); zeros(9,9), (eye(9)-phi'*inv(phi*phi')*phi)]*Pc(:,:,k);
    end

    body.KFC.chain(i).jind = jind(i,:); % indexes of joints in the current chain
    body.KFC.chain(i).y = y;            % measurements
    body.KFC.chain(i).xp = xp;          % predicted states
    body.KFC.chain(i).xc = xc;          % corrected state
    body.KFC.chain(i).Pp = Pp;          % predicted error covariance matrix
    body.KFC.chain(i).Pc = Pc;          % corrected error covariance matrix
    body.KFC.chain(i).xcp = xcp;        % projected state after correction step
    body.KFC.chain(i).Pcp = Pcp;        % projected error covariance matrix after correction
    body.KFC.chain(i).K = K;

    body.KFC.chain(i).sigmaMNl = sigmaMN;       % diagonal matrix of standard deviations 
    body.KFC.chain(i).Rl = diag(sigmaMNl.^2);   % covariance matrix of measurement noise  
    body.KFC.chain(i).Ql = Ql;                  % covariance matrix of process/model noise  
    
    % for each joint of the chain
    for j = 1:size(jind,2) % for each joint of the chain
        body.KFC.joint{jind(i,j)}.coord{1}.sigmaSN = NaN;   % not defined
        body.KFC.joint{jind(i,j)}.coord{1}.Q = NaN;         % not defined  
        body.KFC.joint{jind(i,j)}.coord{1}.sigmaMN = NaN;   % not defined
        body.KFC.joint{jind(i,j)}.coord{1}.R = NaN;         % not defined
        body.KFC.joint{jind(i,j)}.coord{1}.Pp = NaN;        % not defined
        body.KFC.joint{jind(i,j)}.coord{1}.vi = NaN;        % not defined
        body.KFC.joint{jind(i,j)}.coord{1}.Si = NaN;        % not defined
        body.KFC.joint{jind(i,j)}.coord{1}.K = NaN;         % not defined
        body.KFC.joint{jind(i,j)}.coord{1}.Pc = NaN;        % not defined

        kp = 3*(j-1)+1;                 % position index for x
        kv = 3*size(jind,2)+3*(j-1)+1;  % velocity index for x
        body.KFC.joint{jind(i,j)}.coord{1}.xp = [xp(kp,:); xp(kv,:)];       % predicted states
        body.KFC.joint{jind(i,j)}.coord{1}.xc = [xc(kp,:); xc(kv,:)];       % corrected states
        body.KFC.joint{jind(i,j)}.coord{1}.xcp = [xcp(kp,:); xcp(kv,:)];    % projected state after correction step

        kp = 3*(j-1)+2;                 % position index for y
        kv = 3*size(jind,2)+3*(j-1)+2;  % velocity index for y
        body.KFC.joint{jind(i,j)}.coord{2}.xp = [xp(kp,:); xp(kv,:)];       % predicted states
        body.KFC.joint{jind(i,j)}.coord{2}.xc = [xc(kp,:); xc(kv,:)];       % corrected states
        body.KFC.joint{jind(i,j)}.coord{2}.xcp = [xcp(kp,:); xcp(kv,:)];    % projected state after correction step

        kp = 3*(j-1)+3;                 % position index for z
        kv = 3*size(jind,2)+3*(j-1)+3;  % velocity index for z
        body.KFC.joint{jind(i,j)}.coord{3}.xp = [xp(kp,:); xp(kv,:)];       % predicted states
        body.KFC.joint{jind(i,j)}.coord{3}.xc = [xc(kp,:); xc(kv,:)];       % corrected states
        body.KFC.joint{jind(i,j)}.coord{3}.xcp = [xcp(kp,:); xcp(kv,:)];    % projected state after correction step
    end
end

matrix = zeros(32 * 3, length(joints));

for j = 1:32
    joint = body.KFC.joint{j};
    for idx = 1:3
        coord = joint.coord{idx};
        if isfield(coord, "xcp")
            matrix(3 * (j-1) + idx, :) = coord.xcp(1, :);
        else
            matrix(3 * (j-1) + idx, :) = coord.xc(1, :);
        end
    end
end
labels = {};

for i = 1:32
    labels = [labels {['Joint_' int2str(i) '_x']}];
    labels = [labels {['Joint_' int2str(i) '_y']}];
    labels = [labels {['Joint_' int2str(i) '_z']}];
end

T = array2table(matrix');
T.Properties.VariableNames(1:96) = labels;
writetable(T,'file1.csv');

matrix = zeros(32 * 3, length(joints));

for j = 1:32
    joint = body.KFC.joint{j};
    for idx = 1:3
        coord = joint.coord{idx};
        if isfield(coord, "xcp")
            matrix(3 * (j-1) + idx, :) = coord.xcp(2, :);
        else
            matrix(3 * (j-1) + idx, :) = coord.xc(2, :);
        end
    end
end
labels = {};

for i = 1:32
    labels = [labels {['Joint_' int2str(i) '_x']}];
    labels = [labels {['Joint_' int2str(i) '_y']}];
    labels = [labels {['Joint_' int2str(i) '_z']}];
end

T2 = array2table(matrix');
T2.Properties.VariableNames(1:96) = labels;
writetable(T2,'file2.csv');


%%% Compare individually KF filtered, KF with constraints and non-filtered data
%for j = 1:32
%    figure
%    subplot(3,2,1)
%    hold on
%    plot(tstamp,body.NF.joints(:,j,1))
%    plot(tstamp,body.KF.joint{j}.coord{1}.xc(1,:),'r','LineWidth',1)
%    plot(tstamp,body.KFC.joint{j}.coord{1}.xc(1,:),'g','LineWidth',1)
%    hold off
%    legend('unfiltered','KF with gating','KF with constraints')
%    title(['Coord x (lateral) of joint ' joint_names{j}])
%    ylabel('x (m)')
%    xlabel('time (s)')
%    box on
%    
%    subplot(3,2,2)
%    hold on
%    plot(tstamp,body.NF.vjoints(:,j,1))
%    plot(tstamp,body.KF.joint{j}.coord{1}.xc(2,:),'r','LineWidth',1)
%    plot(tstamp,body.KFC.joint{j}.coord{1}.xc(2,:),'g','LineWidth',1)
%    hold off
%    legend('unfiltered','KF with gating','KF with constraints')
%    title(['Velocity x of joint ' joint_names{j}])
%    ylabel('vx (m/s)')
%    xlabel('time (s)')
%    box on
%    
%    subplot(3,2,3)
%    hold on
%    plot(tstamp,body.NF.joints(:,j,2))
%    plot(tstamp,body.KF.joint{j}.coord{2}.xc(1,:),'r','LineWidth',1)
%    plot(tstamp,body.KFC.joint{j}.coord{2}.xc(1,:),'g','LineWidth',1)
%    hold off
%    legend('unfiltered','KF with gating','KF with constraints')
%    title(['Coord y (vertical) of joint ' joint_names{j}])
%    ylabel('y (m)')
%    xlabel('time (s)')
%    box on
%    
%    subplot(3,2,4)
%    hold on
%    plot(tstamp,body.NF.vjoints(:,j,2))
%    plot(tstamp,body.KF.joint{j}.coord{2}.xc(2,:),'r','LineWidth',1)
%    plot(tstamp,body.KFC.joint{j}.coord{2}.xc(2,:),'g','LineWidth',1)
%    hold off
%    legend('unfiltered','KF with gating','KF with constraints')
%    title(['Velocity y of joint ' joint_names{j}])
%    ylabel('vy (m/s)')
%    xlabel('time (s)')
%    box on
%    
%    subplot(3,2,5)
%    hold on
%    plot(tstamp,body.NF.joints(:,j,3))
%    plot(tstamp,body.KF.joint{j}.coord{3}.xc(1,:),'r','LineWidth',1)
%    plot(tstamp,body.KFC.joint{j}.coord{3}.xc(1,:),'g','LineWidth',1)
%    hold off
%    legend('unfiltered','KF with gating','KF with constraints')
%    title(['Coord z (depth) of joint ' joint_names{j}])
%    ylabel('z (m)')
%    xlabel('time (s)')
%    box on
%    
%    subplot(3,2,6)
%    hold on
%    plot(tstamp,body.NF.vjoints(:,j,3))
%    plot(tstamp,body.KF.joint{j}.coord{3}.xc(2,:),'r','LineWidth',1)
%    plot(tstamp,body.KFC.joint{j}.coord{3}.xc(2,:),'g','LineWidth',1)
%    hold off
%    legend('unfiltered','KF with gating','KF with constraints')
%    title(['Velocity z of joint ' joint_names{j}])
%    ylabel('vz (m/s)')
%    xlabel('time (s)')
%    box on
%end
%
%
%%% Compute lengths of segments over time

for i = 1:size(jind,1) % for each chain
    % Segment length before filtering
    L12(i,:) =  sqrt((body.NF.joints(:,jind(i,1),1)-body.NF.joints(:,jind(i,2),1)).^2 + ...
                     (body.NF.joints(:,jind(i,1),2)-body.NF.joints(:,jind(i,2),2)).^2 + ...
                     (body.NF.joints(:,jind(i,1),3)-body.NF.joints(:,jind(i,2),3)).^2);
    
    L23(i,:) =  sqrt((body.NF.joints(:,jind(i,2),1)-body.NF.joints(:,jind(i,3),1)).^2 + ...
                     (body.NF.joints(:,jind(i,2),2)-body.NF.joints(:,jind(i,3),2)).^2 + ...
                     (body.NF.joints(:,jind(i,2),3)-body.NF.joints(:,jind(i,3),3)).^2);
    
    % Segment length after filtering with KF for each coordinate and gating for outlier rejection
    L12_KF(i,:) =  sqrt((body.KF.joint{jind(i,1)}.coord{1}.xc(1,:)-body.KF.joint{jind(i,2)}.coord{1}.xc(1,:)).^2 + ...
                        (body.KF.joint{jind(i,1)}.coord{2}.xc(1,:)-body.KF.joint{jind(i,2)}.coord{2}.xc(1,:)).^2 + ...
                        (body.KF.joint{jind(i,1)}.coord{3}.xc(1,:)-body.KF.joint{jind(i,2)}.coord{3}.xc(1,:)).^2);
    
    L23_KF(i,:) =  sqrt((body.KF.joint{jind(i,2)}.coord{1}.xc(1,:)-body.KF.joint{jind(i,3)}.coord{1}.xc(1,:)).^2 + ...
                        (body.KF.joint{jind(i,2)}.coord{2}.xc(1,:)-body.KF.joint{jind(i,3)}.coord{2}.xc(1,:)).^2 + ...
                        (body.KF.joint{jind(i,2)}.coord{3}.xc(1,:)-body.KF.joint{jind(i,3)}.coord{3}.xc(1,:)).^2);
    
    
    % Segment length after Filtering with leg KF
    L12_KFC(i,:) =  sqrt((body.KFC.joint{jind(i,1)}.coord{1}.xc(1,:)-body.KFC.joint{jind(i,2)}.coord{1}.xc(1,:)).^2 + ...
                         (body.KFC.joint{jind(i,1)}.coord{2}.xc(1,:)-body.KFC.joint{jind(i,2)}.coord{2}.xc(1,:)).^2 + ...
                         (body.KFC.joint{jind(i,1)}.coord{3}.xc(1,:)-body.KFC.joint{jind(i,2)}.coord{3}.xc(1,:)).^2);
    
    L23_KFC(i,:) =  sqrt((body.KFC.joint{jind(i,2)}.coord{1}.xc(1,:)-body.KFC.joint{jind(i,3)}.coord{1}.xc(1,:)).^2 + ...
                         (body.KFC.joint{jind(i,2)}.coord{2}.xc(1,:)-body.KFC.joint{jind(i,3)}.coord{2}.xc(1,:)).^2 + ...
                         (body.KFC.joint{jind(i,2)}.coord{3}.xc(1,:)-body.KFC.joint{jind(i,3)}.coord{3}.xc(1,:)).^2);
    
    % Average values of segment lengths
    L12_avg(i) = mean(L12(i,:));
    L23_avg(i) = mean(L23(i,:));
    L12_KF_avg(i) = mean(L12_KF(i,:));
    L23_KF_avg(i) = mean(L23_KF(i,:));
    L12_KFC_avg(i) = mean(L12_KFC(i,:));
    L23_KFC_avg(i) = mean(L23_KFC(i,:));

    % RMS with respect to average
    L12_rms(i) = rms(L12(i,:)-L12_avg(i));
    L23_rms(i) = rms(L23(i,:)-L23_avg(i));
    L12_KF_rms(i) = rms(L12_KF(i,:)-L12_KF_avg(i));
    L23_KF_rms(i) = rms(L23_KF(i,:)-L23_KF_avg(i));
    L12_KFC_rms(i) = rms(L12_KFC(i,:)-L12_KFC_avg(i));
    L23_KFC_rms(i) = rms(L23_KFC(i,:)-L23_KFC_avg(i));
end

body.NF.L12 = L12;
body.NF.L23 = L23;
body.NF.L12_avg = L12_avg;
body.NF.L23_avg = L23_avg;
body.NF.L12_rms = L12_rms;
body.NF.L23_rms = L23_rms;
body.KF.L12 = L12_KF;
body.KF.L23 = L23_KF;
body.KF.L12_avg = L12_KF_avg;
body.KF.L23_avg = L23_KF_avg;
body.KF.L12_rms = L12_KF_rms;
body.KF.L23_rms = L23_KF_rms;
body.KF.L12 = L12_KFC;
body.KFC.L23 = L23_KFC;
body.KFC.L12_avg = L12_KFC_avg;
body.KFC.L23_avg = L23_KFC_avg;
body.KFC.L12_rms = L12_KFC_rms;
body.KFC.L23_rms = L23_KFC_rms;

save([fileName '_KF'],'body');

%% Plot segment lengths over time
figure
for i=1:4
    subplot(2,4,i)
        hold on
        plot(body.tstamp,L12(i,:));
        plot(body.tstamp,L12_KF(i,:),'r','LineWidth',2);
        plot(body.tstamp,L12_KFC(i,:),'g','LineWidth',2);
        hold off
        legend(['unfiltered: Avg = ' num2str(L12_avg(i)) ' m; RMS = ' num2str(L12_rms(i))], ...
               ['KF with gating: Avg = ' num2str(L12_KF_avg(i)) ' m; RMS = ' num2str(L12_KF_rms(i))], ...
               ['KF with constraints: Avg = ' num2str(L12_KFC_avg(i)) ' m; RMS = ' num2str(L12_KFC_rms(i))]);
        title(['Upper segment chain ' num2str(i)]);
        ylabel('[m]')
        xlabel('[s]')
        box on
    subplot(2,4,4+i)
        hold on
        plot(body.tstamp,L23(i,:));
        plot(body.tstamp,L23_KF(i,:),'r','LineWidth',2);
        plot(body.tstamp,L23_KFC(i,:),'g','LineWidth',2);
        hold off
        legend(['unfiltered: Avg = ' num2str(L23_avg(i)) ' m; RMS = ' num2str(L23_rms(i))], ...
               ['KF with gating: Avg = ' num2str(L23_KF_avg(i)) ' m; RMS = ' num2str(L23_KF_rms(i))], ...
               ['KF with constraints: Avg = ' num2str(L23_KFC_avg(i)) ' m; RMS = ' num2str(L23_KFC_rms(i))]);
        title(['Lower segment chain ' num2str(i)]);
        ylabel('[m]')
        xlabel('[s]')
        box on
end



