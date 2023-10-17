%% 
% Plots experimental data, including CoM position estimation

% Subject's gender
gender = 'f';
MT = 72; % [kg] total body mass of the subject
H = 1.75; % [m] height of the subject

%% Load experimental 
% Pilot May 2021 - Matthew Millard
% fileName = 'calibration_WFOV_UNBINNED_720P_15fps'; 
% fileName = 'rollator_WFOV_2X2BINNED_720P_15fps_0'; 
fileName = 'sts_NFOV_UNBINNED_720P_30fps'; % For "Dynamic Walking"
% fileName = 'sts_WFOV_2X2BINNED_720P_15fps'; % For "Dynamic Walking"
% fileName = 'calibration_NFOV_2X2BINNED_720P_15fps'; 

% Pilot Sept 2022
% fileName = 'dynamic1_NFOV_UNBINNED_720P_30fps'; 
% fileName = 'sts1_NFOV_UNBINNED_720P_30fps'; 
% fileName = 'staticstand1_NFOV_UNBINNED_720P_30fps'; 
% fileName = 'rollator1_WFOV_2X2BINNED_720P_30fps'; 
% fileName = 'walkpert1_NFOV_2X2BINNED_720P_30fps_0'; 

% Pilot Oct 12th 2022
% fileName = 'stand_b2_t1_NFOV_UNBINNED_720P_30fps'; % (after t = 10s)
% fileName = 'dynamicseq_b2_t1_NFOV_UNBINNED_720P_30fps'; 
% fileName = 'dynamicseq_b2_t1_NFOV_UNBINNED_720P_30fps_0'; n
% fileName = 'staticseq_b2_t1_NFOV_UNBINNED_720P_30fps'; 
% fileName = 'staticseq_b2_t1_NFOV_UNBINNED_720P_30fps_0'; INICIAR POR ESTE
% fileName = 'rollatorsts_b2_t1_WFOV_2X2BINNED_720P_30fps'; 

% Pilot Oct 13th 2022 
% fileName = 'stand_b2_s03_NFOV_UNBINNED_720P_30fps';
% fileName = 'staticseq_b2_s03_NFOV_UNBINNED_720P_30fps'; 
% fileName = 'dynamicseq_b2_s03_NFOV_UNBINNED_720P_30fps'; 
% fileName = 'rollator_b2_s03_WFOV_2X2BINNED_720P_30fps_0'; 
% fileName = 'rollatorextra_b2_s03_WFOV_2X2BINNED_720P_30fps'; 
% fileName = 'rollatorsts_b2_s03_WFOV_2X2BINNED_720P_30fps'; 

% Test data Gabriel
% fileName = 'output_NFOV_UNBINNED_OFF_30fps';

fprintf(['Loading original data in ' fileName '.mat ... \n']);
load(fileName);

fprintf(['Loading filtered data in ' fileName '_KF.mat ... \n']);
load([fileName '_KF']);

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

%% Compute CoM and vCoM
% Get mapping of joints to CoM 
% Maps joint positions into center of mass positions according to:
% CoM = MF*pj or
% CoM = MM*pj or
% where:
%   CoM - position of the center of mass
%   pj  - column vector of joint coordinates (either x, y or z)
%   MF,MM   - row mapping matrix for female and male subjects, respectively
[MFs, MMs, MF, MM,  mf, mm] =  getMMaps;

n = length(joints(:,1,1));

CoM = NaN(n,3);
vCoM = NaN(n,3);
CoMs = NaN(n,16,3);
vCoMs = NaN(n,16,3);
Lcom = NaN(n,3);

joints_KF = NaN(n,32,3);
vjoints_KF = NaN(n,32,3);
CoM_KF = NaN(n,3);
vCoM_KF = NaN(n,3);
CoMs_KF = NaN(n,16,3);
vCoMs_KF = NaN(n,16,3);
Lcom_KF = NaN(n,3);

joints_KFC = NaN(n,32,3);
vjoints_KFC = NaN(n,32,3);
CoM_KFC = NaN(n,3);
vCoM_KFC = NaN(n,3);
CoMs_KFC = NaN(n,16,3);
vCoMs_KFC = NaN(n,16,3);
Lcom_KFC = NaN(n,3);

for j = 1:32
    for i = 1:3
        joints_KF(:,j,i) = body.KF.joint{j}.coord{i}.xc(1,:)';
        vjoints_KF(:,j,i) = body.KF.joint{j}.coord{i}.xc(2,:)';
        joints_KFC(:,j,i) = body.KFC.joint{j}.coord{i}.xc(1,:)';
        vjoints_KFC(:,j,i) = body.KFC.joint{j}.coord{i}.xc(2,:)';
    end
end

% Vertical position of the ground in the kinect global frame
vfloor = -0.85; % [m]

if gender == 'f'
    for i=1:3
        CoM(:,i) = joints(:,:,i)*MF';  
        vCoM(:,i) = vjoints(:,:,i)*MF'; 
        CoMs(:,:,i) = joints(:,:,i)*MFs'; 
        vCoMs(:,:,i) = vjoints(:,:,i)*MFs';

        CoM_KF(:,i) = joints_KF(:,:,i)*MF';  
        vCoM_KF(:,i) = vjoints_KF(:,:,i)*MF'; 
        CoMs_KF(:,:,i) = joints_KF(:,:,i)*MFs'; 
        vCoMs_KF(:,:,i) = vjoints_KF(:,:,i)*MFs';

        CoM_KFC(:,i) = joints_KFC(:,:,i)*MF';  
        vCoM_KFC(:,i) = vjoints_KFC(:,:,i)*MF'; 
        CoMs_KFC(:,:,i) = joints_KFC(:,:,i)*MFs'; 
        vCoMs_KFC(:,:,i) = vjoints_KFC(:,:,i)*MFs';
    end
    Ms = mf*MT;
else % gender == 'm'
    for i=1:3
        CoM(:,i) = joints(:,:,i)*MM'; 
        vCoM(:,i) = vjoints(:,:,i)*MM';
        CoMs(:,:,i) = joints(:,:,i)*MMs'; 
        vCoMs(:,:,i) = vjoints(:,:,i)*MMs';

        CoM_KF(:,i) = joints_KF(:,:,i)*MM';  
        vCoM_KF(:,i) = vjoints_KF(:,:,i)*MM'; 
        CoMs_KF(:,:,i) = joints_KF(:,:,i)*MMs'; 
        vCoMs_KF(:,:,i) = vjoints_KF(:,:,i)*MMs';

        CoM_KFC(:,i) = joints_KFC(:,:,i)*MM';  
        vCoM_KFC(:,i) = vjoints_KFC(:,:,i)*MM'; 
        CoMs_KFC(:,:,i) = joints_KFC(:,:,i)*MMs'; 
        vCoMs_KFC(:,:,i) = vjoints_KFC(:,:,i)*MMs';
    end
    Ms = mm*MT;
end

% Compute mass of individual segments
for k = 1:n
    % Get rotation vector and angular momenta
    [Ws(k,:,:),Ls(k,:,:),Lcom(k,:)] = getAngMom(squeeze(joints(k,:,:)),squeeze(vjoints(k,:,:)),squeeze(CoMs(k,:,:)),squeeze(vCoMs(k,:,:)),CoM(k,:),vCoM(k,:),Ms,gender,MT,H);
    [Ws_KF(k,:,:),Ls_KF(k,:,:),Lcom_KF(k,:)] = getAngMom(squeeze(joints_KF(k,:,:)),squeeze(vjoints_KF(k,:,:)),squeeze(CoMs_KF(k,:,:)),squeeze(vCoMs_KF(k,:,:)),CoM_KF(k,:),vCoM_KF(k,:),Ms,gender,MT,H);
    [Ws_KFC(k,:,:),Ls_KFC(k,:,:),Lcom_KFC(k,:)] = getAngMom(squeeze(joints_KFC(k,:,:)),squeeze(vjoints_KFC(k,:,:)),squeeze(CoMs_KFC(k,:,:)),squeeze(vCoMs_KFC(k,:,:)),CoM_KFC(k,:),vCoM_KFC(k,:),Ms,gender,MT,H);
end

figure
subplot(3,3,1)
hold on
plot(tstamp,CoM(:,1));
plot(tstamp,CoM_KF(:,1),'r','Linewidth',1);
plot(tstamp,CoM_KFC(:,1),'g','Linewidth',1);
hold off
title('CoM x');

subplot(3,3,2)
hold on
plot(tstamp,vCoM(:,1));
plot(tstamp,vCoM_KF(:,1),'r','Linewidth',1);
plot(tstamp,vCoM_KFC(:,1),'g','Linewidth',1);
hold off
title('vCoM x');

subplot(3,3,3)
hold on
plot(tstamp,Lcom(:,1));
plot(tstamp,Lcom_KF(:,1),'r','Linewidth',1);
plot(tstamp,Lcom_KFC(:,1),'g','Linewidth',1);
hold off
title('LCoM x');

subplot(3,3,4)
hold on
plot(tstamp,CoM(:,2));
plot(tstamp,CoM_KF(:,2),'r','Linewidth',1);
plot(tstamp,CoM_KFC(:,2),'g','Linewidth',1);
hold off
title('CoM y');

subplot(3,3,5)
hold on
plot(tstamp,vCoM(:,2));
plot(tstamp,vCoM_KF(:,2),'r','Linewidth',1);
plot(tstamp,vCoM_KFC(:,2),'g','Linewidth',1);
hold off
title('vCoM y');

subplot(3,3,6)
hold on
plot(tstamp,Lcom(:,2));
plot(tstamp,Lcom_KF(:,2),'r','Linewidth',1);
plot(tstamp,Lcom_KFC(:,2),'g','Linewidth',1);
hold off
title('LCoM y');

subplot(3,3,7)
hold on
plot(tstamp,CoM(:,3));
plot(tstamp,CoM_KF(:,3),'r','Linewidth',1);
plot(tstamp,CoM_KFC(:,3),'g','Linewidth',1);
hold off
title('CoM z');

subplot(3,3,8)
hold on
plot(tstamp,vCoM(:,3));
plot(tstamp,vCoM_KF(:,3),'r','Linewidth',1);
plot(tstamp,vCoM_KFC(:,3),'g','Linewidth',1);
hold off
title('vCoM z');

subplot(3,3,9)
hold on
plot(tstamp,Lcom(:,3));
plot(tstamp,Lcom_KF(:,3),'r','Linewidth',1);
plot(tstamp,Lcom_KFC(:,3),'g','Linewidth',1);
hold off
title('LCoM z');


%% Compute the Extrapolated CoM (XCoM)
g = 9.81; % [m/s2]

% Average distance between CoM and midpoint of ankle joint
L = mean(sqrt((CoM(:,1) - 0.5*(joints(:,21,1)+joints(:,25,1))).^2 + ...
    (CoM(:,2) - 0.5*(joints(:,21,2)+joints(:,25,2))).^2 + ...
    (CoM(:,3) - 0.5*(joints(:,21,3)+joints(:,25,3))).^2)); % average distance between CoM and midpoint of ankle joint

L_KF = mean(sqrt((CoM_KF(:,1) - 0.5*(joints_KF(:,21,1)+joints_KF(:,25,1))).^2 + ...
    (CoM_KF(:,2) - 0.5*(joints_KF(:,21,2)+joints_KF(:,25,2))).^2 + ...
    (CoM_KF(:,3) - 0.5*(joints_KF(:,21,3)+joints_KF(:,25,3))).^2)); % average distance between CoM and midpoint of ankle joint

L_KFC = mean(sqrt((CoM_KFC(:,1) - 0.5*(joints_KFC(:,21,1)+joints_KFC(:,25,1))).^2 + ...
    (CoM_KFC(:,2) - 0.5*(joints_KFC(:,21,2)+joints_KFC(:,25,2))).^2 + ...
    (CoM_KFC(:,3) - 0.5*(joints_KFC(:,21,3)+joints_KFC(:,25,3))).^2)); % average distance between CoM and midpoint of ankle joint

w0 = sqrt(g/L);
w0_KF = sqrt(g/L_KF);
w0_KFC = sqrt(g/L_KFC);

% XCoM - projected on the ground plane (x,z), with x - lateral, and z - depth
XCoM(:,1) = CoM(:,1) + vCoM(:,1)/w0;
XCoM(:,2) = CoM(:,3) + vCoM(:,3)/w0;

XCoM_KF(:,1) = CoM_KF(:,1) + vCoM_KF(:,1)/w0_KF;
XCoM_KF(:,2) = CoM_KF(:,3) + vCoM_KF(:,3)/w0_KF;

XCoM_KFC(:,1) = CoM_KFC(:,1) + vCoM_KFC(:,1)/w0_KFC;
XCoM_KFC(:,2) = CoM_KFC(:,3) + vCoM_KFC(:,3)/w0_KFC;

figure
hold on
plot(CoM(:,1),CoM(:,3),'--b');
plot(XCoM(:,1),XCoM(:,2),'b');
plot(CoM_KF(:,1),CoM_KF(:,3),'--r','Linewidth',1);
plot(XCoM_KF(:,1),XCoM_KF(:,2),'r','Linewidth',1);
plot(CoM_KFC(:,1),CoM_KFC(:,3),'--g','Linewidth',1);
plot(XCoM_KFC(:,1),XCoM_KFC(:,2),'g','Linewidth',1);
legend('CoM','XCoM','CoM KF','XCoM KF','CoM KFC','XCoM KFC')
xlabel('x [m]');
ylabel('z [m]');
axis equal
hold off

figure
subplot(2,1,1)
hold on
plot(tstamp,CoM(:,1),'--b');
plot(tstamp,XCoM(:,1),'b');
plot(tstamp,CoM_KF(:,1),'--r','Linewidth',1);
plot(tstamp,XCoM_KF(:,1),'r','Linewidth',1);
plot(tstamp,CoM_KFC(:,1),'--g','Linewidth',1);
plot(tstamp,XCoM_KFC(:,1),'g','Linewidth',1);
legend('CoM','XCoM','CoM KF','XCoM KF','CoM KFC','XCoM KFC')
xlabel('t [s]');
ylabel('x [m]');
hold off

subplot(2,1,2)
hold on
plot(tstamp,CoM(:,3),'--b');
plot(tstamp,XCoM(:,2),'b');
plot(tstamp,CoM_KF(:,3),'--r','Linewidth',1);
plot(tstamp,XCoM_KF(:,2),'r','Linewidth',1);
plot(tstamp,CoM_KFC(:,3),'--g','Linewidth',1);
plot(tstamp,XCoM_KFC(:,2),'g','Linewidth',1);
legend('CoM','XCoM','CoM KF','XCoM KF','CoM KFC','XCoM KFC')
xlabel('t [s]');
ylabel('z [m]');
hold off


%% Choose coordinate
% 1: x - lateral
% 2: y - vertical
% 3: z - depth
% i = 2;
% jind = [23 24 25]; % right leg: hip, knee and ankle 
% jind = [13 14 15]; % right arm: shoulder, elbow and wrist
jind = 1:32;
jind_lr = [23 24 25]; % right leg chain
jind_ll = [19 20 21]; % left leg chain
jind_ar = [13 14 16]; % right arm chain
jind_al = [6 7 8]; % left arm chain

% fprintf(['Filtering data for coordinate ' num2str(i) ' of ' joint_names{j} '. \n']);

jump = 1;
f_L = 1/4; % factor for scaling angular momentum vector in animation
f_v = 7; % factor for scaling CoM velocity vector in animation
r = 0.02; % [m] radius of the sphere corresponding to the center of mass
jcolor = [0.5 0.5 0.5]; % joint color
jwidth = 0.5; % line width of the joints 
scolor = [0 0 1]; % segment color
swidth = 2; % line width of the segments 
lframe = 0.3; % (m) length of the unit vector representing the global frame
genanim = 1; % generate animation: 0 - no; 1 - yes
% figure
% for i = 1:jump:length(tstamp)
%     plot3([1 3 3 1 1],[-1 -1 1 1 -1],[-0.85 -0.85 -0.85 -0.85 -0.85],'k'); % floor
%     hold on
%     % Joints
%     for j = 1:length(jind)
%         plot3(joints_KFC(i,jind(j),3),-joints_KFC(i,jind(j),1),-joints_KFC(i,jind(j),2),'o','Color',jcolor,'LineWidth', jwidth);
%     end
%     % Segment chains
%     plot3([ joints_KFC(i,jind_lr(1),3), joints_KFC(i,jind_lr(2),3), joints_KFC(i,jind_lr(3),3)],...
%           [-joints_KFC(i,jind_lr(1),1),-joints_KFC(i,jind_lr(2),1),-joints_KFC(i,jind_lr(3),1)],...
%           [-joints_KFC(i,jind_lr(1),2),-joints_KFC(i,jind_lr(2),2),-joints_KFC(i,jind_lr(3),2)],...
%             '-o','Color',scolor,'LineWidth', swidth,'MarkerSize',4,'MarkerFaceColor',[0.5 0.5 0.5]); % right leg
% 
%     plot3([ joints_KFC(i,jind_ll(1),3), joints_KFC(i,jind_ll(2),3), joints_KFC(i,jind_ll(3),3)],...
%           [-joints_KFC(i,jind_ll(1),1),-joints_KFC(i,jind_ll(2),1),-joints_KFC(i,jind_ll(3),1)],...
%           [-joints_KFC(i,jind_ll(1),2),-joints_KFC(i,jind_ll(2),2),-joints_KFC(i,jind_ll(3),2)],...
%              '-o','Color',scolor,'LineWidth', swidth,'MarkerSize',4,'MarkerFaceColor',[0.5 0.5 0.5]); % left leg
% 
%     plot3([ joints_KFC(i,jind_ar(1),3), joints_KFC(i,jind_ar(2),3), joints_KFC(i,jind_ar(3),3)],...
%           [-joints_KFC(i,jind_ar(1),1),-joints_KFC(i,jind_ar(2),1),-joints_KFC(i,jind_ar(3),1)],...
%           [-joints_KFC(i,jind_ar(1),2),-joints_KFC(i,jind_ar(2),2),-joints_KFC(i,jind_ar(3),2)],...
%             '-o','Color',scolor,'LineWidth', swidth,'MarkerSize',4,'MarkerFaceColor',[0.5 0.5 0.5]); % right arm
% 
%     plot3([ joints_KFC(i,jind_al(1),3), joints_KFC(i,jind_al(2),3), joints_KFC(i,jind_al(3),3)],...
%           [-joints_KFC(i,jind_al(1),1),-joints_KFC(i,jind_al(2),1),-joints_KFC(i,jind_al(3),1)],...
%           [-joints_KFC(i,jind_al(1),2),-joints_KFC(i,jind_al(2),2),-joints_KFC(i,jind_al(3),2)],...
%             '-o','Color',scolor,'LineWidth', swidth,'MarkerSize',4,'MarkerFaceColor',[0.5 0.5 0.5]); % left arm
% 
%     [Xs,Ys,Zs] = sphere;
%     surf(CoM_KFC(i,3)+r*Xs,-CoM_KFC(i,1)+r*Ys,-CoM_KFC(i,2)+r*Zs,'Edgecolor','r','Facecolor','r')
% %     plot3(CoM_KFC(i,3),-CoM_KFC(i,1),-CoM_KFC(i,2),'or', 'LineWidth', 3); % center of mass
%     plot3(CoM_KFC(i,3),-CoM_KFC(i,1),vfloor,'or', 'LineWidth', 2); % center of mass projected on the ground
%     plot3(XCoM_KFC(i,2),-XCoM_KFC(i,1),vfloor,'og', 'LineWidth', 3)
%     %plot3(CoMs_KFC(i,:,3),-CoMs_KFC(i,:,1),-CoMs_KFC(i,:,2),'og', 'LineWidth', 4);
%     quiver3(CoM_KFC(i,3),-CoM_KFC(i,1),-CoM_KFC(i,2),vCoM_KFC(i,3),-vCoM_KFC(i,1),-vCoM_KFC(i,2),'g','LineWidth', 3);
%     quiver3(CoM_KFC(i,3),-CoM_KFC(i,1),-CoM_KFC(i,2),f_L*Lcom_KFC(i,3),-f_L*Lcom_KFC(i,1),-f_L*Lcom_KFC(i,2),'m','LineWidth', 3);
%     axis equal 
%     xlabel('z (m)')
%     ylabel('x (m)')
%     zlabel('y (m)')
%     xlim([1 3]);
%     ylim([-1 1]);
%     zlim([-1 1])
%     view([-1 1 0.5])
%     hold off 
%     axis off
%     pause
% end

%% Dynamic walking
colorun = [0.8 0.8 0.8];
% h = figure('Color','white','Position',[1 41 1920 963]);
h = figure('Color','white');
pause
if genanim == 1
    F(length(1:jump:length(tstamp))) = struct('cdata',[],'colormap',[]);
    v = VideoWriter([fileName '_anim.avi']);
    open(v);
    k = 1; 
end

for i = 1:jump:length(tstamp)
    
    subplot(3,6,4)
    hold on
    plot(tstamp(1:i),CoM(1:i,1),'Color',colorun,'Linewidth',0.5);
    plot(tstamp(1:i),XCoM(1:i,1),'Color',colorun,'Linewidth',0.5);
%     plot(tstamp,CoM_KF(:,1),'r','Linewidth',1);
    plot(tstamp(1:i),CoM_KFC(1:i,1),'r','Linewidth',1.5);
    plot(tstamp(1:i),XCoM_KFC(1:i,1),'Color','#EDB120','Linewidth',1.5);
    legend('','','CoM','XCoM')
    xlim([0 tstamp(end)]);
    ylim([-0.5 -0.1]);
    hold off
    title('CoM and XCoM x');
    box on
    
    subplot(3,6,5)
    hold on
    plot(tstamp(1:i),vCoM(1:i,1),'Color',colorun,'Linewidth',0.5);
%     plot(tstamp,vCoM_KF(:,1),'r','Linewidth',1);
    plot(tstamp(1:i),vCoM_KFC(1:i,1),'g','Linewidth',1.5);
    xlim([0 tstamp(end)]);
    ylim([-0.8 0.8]);
    hold off
    title('Velocity CoM x');
    box on
    
    subplot(3,6,6)
    hold on
    plot(tstamp(1:i),Lcom(1:i,1),'Color',colorun,'Linewidth',0.5);
%     plot(tstamp,Lcom_KF(1:i,1),'r','Linewidth',1);
    plot(tstamp(1:i),Lcom_KFC(1:i,1),'m','Linewidth',1.5);
    xlim([0 tstamp(end)]);
    ylim([-6 6]);
    hold off
    title('Ang. Mom. x');
    box on
    
    subplot(3,6,10)
    hold on
    plot(tstamp(1:i),CoM(1:i,2),'Color',colorun,'Linewidth',0.5);
%     plot(tstamp,CoM_KF(1:i,2),'r','Linewidth',1);
    plot(tstamp(1:i),CoM_KFC(1:i,2),'r','Linewidth',1.5);
    xlim([0 tstamp(end)]);
    ylim([-0.2 0.2]);
    hold off
    title('CoM and XCoM y');
    box on
    
    subplot(3,6,11)
    hold on
    plot(tstamp(1:i),vCoM(1:i,2),'Color',colorun,'Linewidth',0.5);
%     plot(tstamp,vCoM_KF(1:i,2),'r','Linewidth',1);
    plot(tstamp(1:i),vCoM_KFC(1:i,2),'g','Linewidth',1.5);
    xlim([0 tstamp(end)]);
    ylim([-0.8 0.8]);
    hold off
    title('Velocity CoM y');
    box on
    
    subplot(3,6,12)
    hold on
    plot(tstamp(1:i),Lcom(1:i,2),'Color',colorun,'Linewidth',0.5);
%     plot(tstamp,Lcom_KF(1:i,2),'r','Linewidth',1);
    plot(tstamp(1:i),Lcom_KFC(1:i,2),'m','Linewidth',1.5);
    xlim([0 tstamp(end)]);
    ylim([-6 6]);
    hold off
    title('Ang. Mom. y');
    box on
    
    subplot(3,6,16)
    hold on
    plot(tstamp(1:i),CoM(1:i,3),'Color',colorun,'Linewidth',0.5);
    plot(tstamp(1:i),XCoM(1:i,2),'Color',colorun,'Linewidth',0.5);
%     plot(tstamp,CoM_KF(1:i,3),'r','Linewidth',1);
    plot(tstamp(1:i),CoM_KFC(1:i,3),'r','Linewidth',1.5);
    plot(tstamp(1:i),XCoM_KFC(1:i,2),'Color','#EDB120','Linewidth',1.5);
    xlim([0 tstamp(end)]);
    ylim([1.8 2.2]);
    hold off
    title('CoM and XCoM z');
    box on
    
    subplot(3,6,17)
    hold on
    plot(tstamp(1:i),vCoM(1:i,3),'Color',colorun,'Linewidth',0.5);
%     plot(tstamp,vCoM_KF(1:i,3),'r','Linewidth',1);
    plot(tstamp(1:i),vCoM_KFC(1:i,3),'g','Linewidth',1.5);
    xlim([0 tstamp(end)]);
    ylim([-0.8 0.8]);
    hold off
    title('Velocity CoM z');
    box on
    
    subplot(3,6,18)
    hold on
    plot(tstamp(1:i),Lcom(1:i,3),'Color',colorun,'Linewidth',0.5);
%     plot(tstamp,Lcom_KF(1:i,3),'r','Linewidth',1);
    plot(tstamp(1:i),Lcom_KFC(1:i,3),'m','Linewidth',1.5);
    xlim([0 tstamp(end)]);
    ylim([-6 6]);
    hold off
    title('Ang. Mom. z');
    box on

    subplot(3,6,[1 2 3 7 8 9 13 14 15])
    plot3([0 3 3 0 0],[-1 -1 1 1 -1],[-0.85 -0.85 -0.85 -0.85 -0.85],'k'); % floor
    hold on
    % Joints
    for j = 1:length(jind)
        plot3(joints_KFC(i,jind(j),3),-joints_KFC(i,jind(j),1),-joints_KFC(i,jind(j),2),'o','Color',jcolor,'LineWidth', jwidth);
    end
    % Segment chains
    plot3([ joints_KFC(i,jind_lr(1),3), joints_KFC(i,jind_lr(2),3), joints_KFC(i,jind_lr(3),3)],...
          [-joints_KFC(i,jind_lr(1),1),-joints_KFC(i,jind_lr(2),1),-joints_KFC(i,jind_lr(3),1)],...
          [-joints_KFC(i,jind_lr(1),2),-joints_KFC(i,jind_lr(2),2),-joints_KFC(i,jind_lr(3),2)],...
            '-o','Color',scolor,'LineWidth', swidth,'MarkerSize',4,'MarkerFaceColor',[0.5 0.5 0.5]); % right leg

    plot3([ joints_KFC(i,jind_ll(1),3), joints_KFC(i,jind_ll(2),3), joints_KFC(i,jind_ll(3),3)],...
          [-joints_KFC(i,jind_ll(1),1),-joints_KFC(i,jind_ll(2),1),-joints_KFC(i,jind_ll(3),1)],...
          [-joints_KFC(i,jind_ll(1),2),-joints_KFC(i,jind_ll(2),2),-joints_KFC(i,jind_ll(3),2)],...
             '-o','Color',scolor,'LineWidth', swidth,'MarkerSize',4,'MarkerFaceColor',[0.5 0.5 0.5]); % left leg

    plot3([ joints_KFC(i,jind_ar(1),3), joints_KFC(i,jind_ar(2),3), joints_KFC(i,jind_ar(3),3)],...
          [-joints_KFC(i,jind_ar(1),1),-joints_KFC(i,jind_ar(2),1),-joints_KFC(i,jind_ar(3),1)],...
          [-joints_KFC(i,jind_ar(1),2),-joints_KFC(i,jind_ar(2),2),-joints_KFC(i,jind_ar(3),2)],...
            '-o','Color',scolor,'LineWidth', swidth,'MarkerSize',4,'MarkerFaceColor',[0.5 0.5 0.5]); % right arm

    plot3([ joints_KFC(i,jind_al(1),3), joints_KFC(i,jind_al(2),3), joints_KFC(i,jind_al(3),3)],...
          [-joints_KFC(i,jind_al(1),1),-joints_KFC(i,jind_al(2),1),-joints_KFC(i,jind_al(3),1)],...
          [-joints_KFC(i,jind_al(1),2),-joints_KFC(i,jind_al(2),2),-joints_KFC(i,jind_al(3),2)],...
            '-o','Color',scolor,'LineWidth', swidth,'MarkerSize',4,'MarkerFaceColor',[0.5 0.5 0.5]); % left arm

    [Xs,Ys,Zs] = sphere;
    surf(CoM_KFC(i,3)+r*Xs,-CoM_KFC(i,1)+r*Ys,-CoM_KFC(i,2)+r*Zs,'Edgecolor','r','Facecolor','r')
%     plot3(CoM_KFC(i,3),-CoM_KFC(i,1),-CoM_KFC(i,2),'or', 'LineWidth', 3); % center of mass
    plot3([CoM_KFC(i,3) CoM_KFC(i,3)],[-CoM_KFC(i,1) -CoM_KFC(i,1)],[-CoM_KFC(i,2) vfloor],'--r')
    plot3(CoM_KFC(i,3),-CoM_KFC(i,1),vfloor,'or', 'LineWidth', 2); % center of mass projected on the ground
    plot3(XCoM_KFC(i,2),-XCoM_KFC(i,1),vfloor,'o', 'LineWidth', 3,'Color','#EDB120')
    text(XCoM_KFC(i,2)-0.05,-XCoM_KFC(i,1)-0.05,vfloor,'XCoM','Color','#EDB120');
    %plot3(CoMs_KFC(i,:,3),-CoMs_KFC(i,:,1),-CoMs_KFC(i,:,2),'og', 'LineWidth', 4);
    quiver3(CoM_KFC(i,3),-CoM_KFC(i,1),-CoM_KFC(i,2),vCoM_KFC(i,3),-vCoM_KFC(i,1),-vCoM_KFC(i,2),'g','LineWidth', 3);
    quiver3(CoM_KFC(i,3),-CoM_KFC(i,1),-CoM_KFC(i,2),f_L*Lcom_KFC(i,3),-f_L*Lcom_KFC(i,1),-f_L*Lcom_KFC(i,2),'m','LineWidth', 3);
    quiver3(0, 0, 0, lframe, 0, 0,'k','LineWidth', 1.5); % unit vector in z
    quiver3(0, 0, 0, 0, -lframe, 0,'k','LineWidth', 1.5); % unit vector in x
    quiver3(0, 0, 0, 0, 0, -lframe,'k','LineWidth', 1.5); % unit vector in y
    text(lframe+0.05,0,0,'z');
    text(0,-lframe,0,'x');
    text(0,0,-lframe,'y');
    text(0,0,0.1,'Kinect','HorizontalAlignment','center');
    axis equal 
    xlabel('z (m)')
    ylabel('-x (m)')
    zlabel('-y (m)')
    xlim([0 3]);
    ylim([-1 1]);
    zlim([-1 1])
    view([-1 1 0.5])
    hold off 
    axis off
    if i == 1
        pause
    end
    if genanim == 1
        F(k) = getframe(h);
        writeVideo(v,F(k));
        k = k + 1;
    end
end

if genanim == 1
    close(v)
    movie(h,F)
end


