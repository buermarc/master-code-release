function [MFs, MMs, MF, MM,  mf, mm] = getMMaps
% Maps joint positions into center of mass positions of each segment of the
% body and of the total body.
%
% Outputs
%   MFs,MMs - matrices that map the joint positions into the positions of the center of mass of the 16 body segments for female, male subjects, respectively
%   MF,MM   - row matrices that map the joint positions into the whole body CoM for female and male subjects, respectively
%   mf,mm   - column vector of normalized masses of the 16 segments for female, male subjects, respectively
%
% where:
% CoMs = MFs*pj or CoMs = MMs*pj
% CoM = MF*pj or CoM = MM*pj
% ms = mf*MT or mf*MT
%   CoMs - position of the center of mass of each segment
%   CoM  - position of the center of mass of the body
%   ms   - normal16ized masses of the 16 segments of the body
%   pj   - column vector of joint coordinates (either x, y or z)
%   MT   - total mass of the body



%% Joint index
% From Kinect
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

%% Initialization

MFj = zeros(16,32);
MMj = zeros(16,32);
mf = zeros(16,1);
mm = zeros(16,1);

%% Segment
% See de Leva (1996) and https://learn.microsoft.com/en-us/azure/kinect-dk/body-joints
%% 1 - LPT / Pelvis
% Description: lower part of the trunk
% Origin joint: Joints: naval/OMPH  (j = 2)
% Other joint: midpoint MIDH of hip left (j = 19) and hip right (j = 23)
% Assumptions: CoM between midpoint of hip joints (MIDH) and naval (OMPH
s = 1;
jorig = 2;  % OMPH (de Leva) / SPINE_NAVEL (Kinect)
joth1 = 19; % hip left
joth2 = 23; % hip right

mf(s) = 0.1247; % normalized mass of the segment from Tab. 4 of de Leva (1996)
pmf(s) = 0.492; % normalized CoM position from origin from Tab. 4 of de Leva (1996)
MFs(s,joth1) = pmf(s)/2;
MFs(s,joth2) = pmf(s)/2;
MFs(s,jorig) = (1-pmf(s));
MFj(s,joth1) = mf(s)*pmf(s)/2;
MFj(s,joth2) = mf(s)*pmf(s)/2;
MFj(s,jorig) = mf(s)*(1-pmf(s));

mm(s) = 0.1117; % normalized mass of the segment from Tab. 4 of de Leva (1996)
pmm(s) = 0.6115; % normalized CoM position from origin from Tab. 4 of de Leva (1996)
MMs(s,joth1) = pmm(s)/2;
MMs(s,joth2) = pmm(s)/2;
MMs(s,jorig) = (1-pmm(s));
MMj(s,joth1) = mm(s)*pmm(s)/2;
MMj(s,joth2) = mm(s)*pmm(s)/2;
MMj(s,jorig) = mm(s)*(1-pmm(s));

%% 2 - MPT
% Description: middle part of the trunk
% Origin joint: Xyphoid or substernale (XYPH)
% Other joint: naval (OMPH)
s = 2;
jorig = 3;  % XYPH (de Leva) / SPINE_CHEST (Kinect)
joth = 2;  % OMPH (naval)

mf(s) = 0.1465; % normalized mass of the segment from Tab. 4 of de Leva (1996)
pmf(s) = 0.4512; % normalized CoM position from origin from Tab. 4 of de Leva (1996)
MFs(s,joth) = pmf(s);
MFs(s,jorig) = (1-pmf(s));
MFj(s,joth) = mf(s)*pmf(s);
MFj(s,jorig) = mf(s)*(1-pmf(s));

mm(s) = 0.1633; % normalized mass of the segment from Tab. 4 of de Leva (1996)
pmm(s) = 0.4502; % normalized CoM position from origin from Tab. 4 of de Leva (1996)
MMs(s,joth) = pmm(s);
MMs(s,jorig) = (1-pmm(s));
MMj(s,joth) = mm(s)*pmm(s);
MMj(s,jorig) = mm(s)*(1-pmm(s));

%% 3 - UPT
% Description: upper part of the trunk
% Origin joint: suprasternale (SUPR) - in Kinect the midpoint between clavicle left and right
% Other joint: Xyphoid or substernale (XYPH)
s = 3;
jorig1 = 5;  % clavicle left
jorig2 = 12;  % clavicle left
joth = 3;  % XYPH (de Leva) / SPINE_CHEST (Kinect)

mf(s) = 0.1545; % normalized mass of the segment from Tab. 4 of de Leva (1996)
pmf(s) = 0.2077; % normalized CoM position from origin from Tab. 4 of de Leva (1996)
MFs(s,joth) = pmf(s);
MFs(s,jorig1) = (1-pmf(s))/2;
MFs(s,jorig2) = (1-pmf(s))/2;
MFj(s,joth) = mf(s)*pmf(s);
MFj(s,jorig1) = mf(s)*(1-pmf(s))/2;
MFj(s,jorig2) = mf(s)*(1-pmf(s))/2;

mm(s) = 0.1596; % normalized mass of the segment from Tab. 4 of de Leva (1996)
pmm(s) = 0.2999; % normalized CoM position from origin from Tab. 4 of de Leva (1996)
MMs(s,joth) = pmm(s);
MMs(s,jorig1) = (1-pmm(s))/2;
MMs(s,jorig2) = (1-pmm(s))/2;
MMj(s,joth) = mm(s)*pmm(s);
MMj(s,jorig1) = mm(s)*(1-pmm(s))/2;
MMj(s,jorig2) = mm(s)*(1-pmm(s))/2;

%% 4 - HEAD
% Description: Head (from mid-gonion ro vertex)
% Origin joint: midpoint between ears
% Assumption: the mass is concentrated in the midpoint of ear left (EAR_LEFT), ear
% right (EAR_RIGHT), and head (HEAD)
s = 4;
jorig = 27;  % HEAD
joth1 = 30;  % EAR_LEFT
joth2 = 32;  % EAR_RIGHT

mf(s) = 0.0668; % normalized mass of the segment from Tab. 4 of de Leva (1996)
pmf(s) = 2/3; % normalized CoM position from assumption above
MFs(s,joth1) = pmf(s)/2;
MFs(s,joth2) = pmf(s)/2;
MFs(s,jorig) = (1-pmf(s));
MFj(s,joth1) = mf(s)*pmf(s)/2;
MFj(s,joth2) = mf(s)*pmf(s)/2;
MFj(s,jorig) = mf(s)*(1-pmf(s));

mm(s) = 0.0694; % normalized mass of the segment from Tab. 4 of de Leva (1996)
pmm(s) = 2/3; % normalized CoM position from origin from Tab. 4 of de Leva (1996)
MMs(s,joth1) = pmm(s)/2;
MMs(s,joth2) = pmm(s)/2;
MMs(s,jorig) = (1-pmm(s));
MMj(s,joth1) = mm(s)*pmm(s)/2;
MMj(s,joth2) = mm(s)*pmm(s)/2;
MMj(s,jorig) = mm(s)*(1-pmm(s));

%% 5 - upper arm left
% Description: left upper arms
% Origin joint: shoulder joint left(SJC) / SHOULDER_LEFT (Kinect)
% Other joint: elbow joint left (EJC) / ELBOW_LEFT (Kinect)
s = 5;
jorig = 6;  % shoulder joint (SJC) / SHOULDER_LEFT (Kinect)
joth = 7;  % elbow joint (EJC) / ELBOW_LEFT (Kinect)

mf(s) = 0.0255; % normalized mass of the segment from Tab. 4 of de Leva (1996)
pmf(s) = 0.5754; % normalized CoM position from origin from Tab. 4 of de Leva (1996)
MFs(s,joth) = pmf(s);
MFs(s,jorig) = (1-pmf(s));
MFj(s,joth) = mf(s)*pmf(s);
MFj(s,jorig) = mf(s)*(1-pmf(s));

mm(s) = 0.0271; % normalized mass of the segment from Tab. 4 of de Leva (1996)
pmm(s) = 0.5772; % normalized CoM position from origin from Tab. 4 of de Leva (1996)
MMs(s,joth) = pmm(s);
MMs(s,jorig) = (1-pmm(s));
MMj(s,joth) = mm(s)*pmm(s);
MMj(s,jorig) = mm(s)*(1-pmm(s));

%% 6 - forearm left
% Description: left forearm
% Origin joint: elbow joint left (EJC) / ELBOW_LEFT (Kinect)
% Other joint: wrist joint left (WJC) / WRIST_LEFT (Kinect)
s = 6;
jorig = 7;  % elbow joint left (EJC) / ELBOW_LEFT (Kinect)
joth = 8;  % wrist joint left (WJC) / WRIST_LEFT (Kinect)

mf(s) = 0.0138; % normalized mass of the segment from Tab. 4 of de Leva (1996)
pmf(s) = 0.4559; % normalized CoM position from origin from Tab. 4 of de Leva (1996)
MFs(s,joth) = pmf(s);
MFs(s,jorig) = (1-pmf(s));
MFj(s,joth) = mf(s)*pmf(s);
MFj(s,jorig) = mf(s)*(1-pmf(s));

mm(s) = 0.0162; % normalized mass of the segment from Tab. 4 of de Leva (1996)
pmm(s) = 0.4574; % normalized CoM position from origin from Tab. 4 of de Leva (1996)
MMs(s,joth) = pmm(s);
MMs(s,jorig) = (1-pmm(s));
MMj(s,joth) = mm(s)*pmm(s);
MMj(s,jorig) = mm(s)*(1-pmm(s));

%% 7 - hand left
% Description: left forearm
% Origin joint: wrist joint left (WJC) / WRIST_LEFT (Kinect)
% Other joint: 3rd metacarpale (MET3) / HAND_LEFT (Kinect)
s = 7;
jorig = 8;  % wrist joint left (WJC) / WRIST_LEFT (Kinect)
joth = 9;  % 3rd metacarpale (MET3) / HAND_LEFT (Kinect)

mf(s) = 0.0056; % normalized mass of the segment from Tab. 4 of de Leva (1996)
pmf(s) = 0.7474; % normalized CoM position from origin from Tab. 4 of de Leva (1996)
MFs(s,joth) = pmf(s);
MFs(s,jorig) = (1-pmf(s));
MFj(s,joth) = mf(s)*pmf(s);
MFj(s,jorig) = mf(s)*(1-pmf(s));

mm(s) = 0.0061; % normalized mass of the segment from Tab. 4 of de Leva (1996)
pmm(s) = 0.7900; % normalized CoM position from origin from Tab. 4 of de Leva (1996)
MMs(s,joth) = pmm(s);
MMs(s,jorig) = (1-pmm(s));
MMj(s,joth) = mm(s)*pmm(s);
MMj(s,jorig) = mm(s)*(1-pmm(s));

%% 8 - upper arm right
% Description: left upper arms
% Origin joint: shoulder joint right (SJC) / SHOULDER_RIGHT (Kinect)
% Other joint: elbow joint right (EJC) / ELBOW_RIGHT (Kinect)
s = 8;
jorig = 13;  % shoulder joint right (SJC) / SHOULDER_RIGHT (Kinect)
joth = 14;  % elbow joint right (EJC) / ELBOW_RIGHT (Kinect)

mf(s) = 0.0255; % normalized mass of the segment from Tab. 4 of de Leva (1996)
pmf(s) = 0.5754; % normalized CoM position from origin from Tab. 4 of de Leva (1996)
MFs(s,joth) = pmf(s);
MFs(s,jorig) = (1-pmf(s));
MFj(s,joth) = mf(s)*pmf(s);
MFj(s,jorig) = mf(s)*(1-pmf(s));

mm(s) = 0.0271; % normalized mass of the segment from Tab. 4 of de Leva (1996)
pmm(s) = 0.5772; % normalized CoM position from origin from Tab. 4 of de Leva (1996)
MMs(s,joth) = pmm(s);
MMs(s,jorig) = (1-pmm(s));
MMj(s,joth) = mm(s)*pmm(s);
MMj(s,jorig) = mm(s)*(1-pmm(s));

%% 9 - forearm right
% Description: right forearm
% Origin joint: elbow joint right (EJC) / ELBOW_RIGHT (Kinect)
% Other joint: wrist joint right (WJC) / WRIST_RIGHT (Kinect)
s = 9;
jorig = 14;  % elbow joint right (EJC) / ELBOW_RIGHT (Kinect)
joth = 15;  % wrist joint right (WJC) / WRIST_RIGHT (Kinect)

mf(s) = 0.0138; % normalized mass of the segment from Tab. 4 of de Leva (1996)
pmf(s) = 0.4559; % normalized CoM position from origin from Tab. 4 of de Leva (1996)
MFs(s,joth) = pmf(s);
MFs(s,jorig) = (1-pmf(s));
MFj(s,joth) = mf(s)*pmf(s);
MFj(s,jorig) = mf(s)*(1-pmf(s));

mm(s) = 0.0162; % normalized mass of the segment from Tab. 4 of de Leva (1996)
pmm(s) = 0.4574; % normalized CoM position from origin from Tab. 4 of de Leva (1996)
MMs(s,joth) = pmm(s);
MMs(s,jorig) = (1-pmm(s));
MMj(s,joth) = mm(s)*pmm(s);
MMj(s,jorig) = mm(s)*(1-pmm(s));

%% 10 - hand right
% Description: righ forearm
% Origin joint: wrist joint right (WJC) / WRIST_RIGHT (Kinect)
% Other joint: 3rd metacarpale right (MET3) / HAND_RIGHT (Kinect)
s = 10;
jorig = 15;  % wrist joint right (WJC) / WRIST_RIGHT (Kinect)
joth = 16;  % 3rd metacarpale right (MET3) / HAND_RIGHT (Kinect)

mf(s) = 0.0056; % normalized mass of the segment from Tab. 4 of de Leva (1996)
pmf(s) = 0.7474; % normalized CoM position from origin from Tab. 4 of de Leva (1996)
MFs(s,joth) = pmf(s);
MFs(s,jorig) = (1-pmf(s));
MFj(s,joth) = mf(s)*pmf(s);
MFj(s,jorig) = mf(s)*(1-pmf(s));

mm(s) = 0.0061; % normalized mass of the segment from Tab. 4 of de Leva (1996)
pmm(s) = 0.7900; % normalized CoM position from origin from Tab. 4 of de Leva (1996)
MMs(s,joth) = pmm(s);
MMs(s,jorig) = (1-pmm(s));
MMj(s,joth) = mm(s)*pmm(s);
MMj(s,jorig) = mm(s)*(1-pmm(s));

%% 11 - thigh left
% Description: left thigh
% Origin joint: hip joint left (HJC) / HIP_LEFT (Kinect)
% Other joint: knee joint left (KJC) / KNEE_LEFT (Kinect)
s = 11;
jorig = 19;  % shoulder joint (SJC) / SHOULDER_LEFT (Kinect)
joth = 20;  % elbow joint (EJC) / ELBOW_LEFT (Kinect)

mf(s) = 0.1478; % normalized mass of the segment from Tab. 4 of de Leva (1996)
pmf(s) = 0.3612; % normalized CoM position from origin from Tab. 4 of de Leva (1996)
MFs(s,joth) = pmf(s);
MFs(s,jorig) = (1-pmf(s));
MFj(s,joth) = mf(s)*pmf(s);
MFj(s,jorig) = mf(s)*(1-pmf(s));

mm(s) = 0.1416; % normalized mass of the segment from Tab. 4 of de Leva (1996)
pmm(s) = 0.4095; % normalized CoM position from origin from Tab. 4 of de Leva (1996)
MMs(s,joth) = pmm(s);
MMs(s,jorig) = (1-pmm(s));
MMj(s,joth) = mm(s)*pmm(s);
MMj(s,jorig) = mm(s)*(1-pmm(s));

%% 12 - shank left
% Description: left shank (between knee joint and ankle joint (alternative endpoints in Tab. 4 of de Leva (1996))
% Origin joint: knee joint left (KJC) / KNEE_LEFT (Kinect)
% Other joint: ankle joint left / ANKLE_LEFT (Kinect)
s = 12;
jorig = 20;  % knee joint left (KJC) / KNEE_LEFT (Kinect)
joth = 21;  % ankle joint left / ANKLE_LEFT (Kinect)

mf(s) = 0.0481; % normalized mass of the segment from Tab. 4 of de Leva (1996)
pmf(s) = 0.4352; % normalized CoM position from origin from Tab. 4 of de Leva (1996)
MFs(s,joth) = pmf(s);
MFs(s,jorig) = (1-pmf(s));
MFj(s,joth) = mf(s)*pmf(s);
MFj(s,jorig) = mf(s)*(1-pmf(s));

mm(s) = 0.0433; % normalized mass of the segment from Tab. 4 of de Leva (1996)
pmm(s) = 0.4395; % normalized CoM position from origin from Tab. 4 of de Leva (1996)
MMs(s,joth) = pmm(s);
MMs(s,jorig) = (1-pmm(s));
MMj(s,joth) = mm(s)*pmm(s);
MMj(s,jorig) = mm(s)*(1-pmm(s));

%% 13 - foot left
% Description: left foot (between heel (HEEL) and toe tip (TTIP))
% Origin joint: ankle joint left / ANKLE_LEFT (Kinect)
% Other joint: FOOT_LEFT
% Assumption: Considering the reference length as the distance between
% ankle and tip of the toe, and that this length is 3/4 of the length
% between heel and tip of the toe.
s = 13;
jorig = 21;  % ANKLE_LEFT (Kinect)
joth = 22;  % FOOT_LEFT (Kinect)

mf(s) = 0.0129; % normalized mass of the segment from Tab. 4 of de Leva (1996)
pmf(s) = (4/3)*(0.4014 - 1/4); 
MFs(s,joth) = pmf(s);
MFs(s,jorig) = (1-pmf(s));
MFj(s,joth) = mf(s)*pmf(s);
MFj(s,jorig) = mf(s)*(1-pmf(s));

mm(s) = 0.0137; % normalized mass of the segment from Tab. 4 of de Leva (1996)
pmm(s) = (4/3)*(0.4415 - 1/4); 
MMs(s,joth) = pmm(s);
MMs(s,jorig) = (1-pmm(s));
MMj(s,joth) = mm(s)*pmm(s);
MMj(s,jorig) = mm(s)*(1-pmm(s));

%% 14 - thigh right
% Description: right thigh
% Origin joint: hip joint right (HJC) / HIP_RIGHT (Kinect)
% Other joint: knee joint right (KJC) / KNEE_RIGHT (Kinect)
s = 14;
jorig = 23;  % hip joint right (HJC) / HIP_RIGHT (Kinect)
joth = 24;  % knee joint right (KJC) / KNEE_RIGHT (Kinect)

mf(s) = 0.1478; % normalized mass of the segment from Tab. 4 of de Leva (1996)
pmf(s) = 0.3612; % normalized CoM position from origin from Tab. 4 of de Leva (1996)
MFs(s,joth) = pmf(s);
MFs(s,jorig) = (1-pmf(s));
MFj(s,joth) = mf(s)*pmf(s);
MFj(s,jorig) = mf(s)*(1-pmf(s));

mm(s) = 0.1416; % normalized mass of the segment from Tab. 4 of de Leva (1996)
pmm(s) = 0.4095; % normalized CoM position from origin from Tab. 4 of de Leva (1996)
MMs(s,joth) = pmm(s);
MMs(s,jorig) = (1-pmm(s));
MMj(s,joth) = mm(s)*pmm(s);
MMj(s,jorig) = mm(s)*(1-pmm(s));

%% 15 - shank right
% Description: right shank (between knee joint and ankle joint (alternative endpoints in Tab. 4 of de Leva (1996))
% Origin joint: knee joint right (KJC) / KNEE_RIGHT (Kinect)
% Other joint: ankle joint right / ANKLE_RIGHT (Kinect)
s = 15;
jorig = 24;  % knee joint right (KJC) / KNEE_RIGHT (Kinect)
joth = 25;  % ankle joint right / ANKLE_RIGHT (Kinect)

mf(s) = 0.0481; % normalized mass of the segment from Tab. 4 of de Leva (1996)
pmf(s) = 0.4352; % normalized CoM position from origin from Tab. 4 of de Leva (1996)
MFs(s,joth) = pmf(s);
MFs(s,jorig) = (1-pmf(s));
MFj(s,joth) = mf(s)*pmf(s);
MFj(s,jorig) = mf(s)*(1-pmf(s));

mm(s) = 0.0433; % normalized mass of the segment from Tab. 4 of de Leva (1996)
pmm(s) = 0.4395; % normalized CoM position from origin from Tab. 4 of de Leva (1996)
MMs(s,joth) = pmm(s);
MMs(s,jorig) = (1-pmm(s));
MMj(s,joth) = mm(s)*pmm(s);
MMj(s,jorig) = mm(s)*(1-pmm(s));

%% 16 - foot right
% Description: right foot (between heel (HEEL) and toe tip (TTIP))
% Origin joint: ankle joint right / ANKLE_RIGHT (Kinect)
% Other joint: FOOT_RIGHT
% Assumption: Considering the reference length as the distance between
% ankle and tip of the toe, and that this length is 3/4 of the length
% between heel and tip of the toe.
s = 16;
jorig = 25;  % ANKLE_RIGHT (Kinect)
joth = 26;  % FOOT_RIGHT (Kinect)

mf(s) = 0.0129; % normalized mass of the segment from Tab. 4 of de Leva (1996)
pmf(s) = (4/3)*(0.4014 - 1/4); 
MFs(s,joth) = pmf(s);
MFs(s,jorig) = (1-pmf(s));
MFj(s,joth) = mf(s)*pmf(s);
MFj(s,jorig) = mf(s)*(1-pmf(s));

mm(s) = 0.0137; % normalized mass of the segment from Tab. 4 of de Leva (1996)
pmm(s) = (4/3)*(0.4415 - 1/4); 
MMs(s,joth) = pmm(s);
MMs(s,jorig) = (1-pmm(s));
MMj(s,joth) = mm(s)*pmm(s);
MMj(s,jorig) = mm(s)*(1-pmm(s));

% Sum must be 1!
% sum(mf)
% sum(mm)
% sum(MFj,'all')
% sum(MMj,'all')

%% Compute the matrices that map positions of joints to center of mass of the mody
MF = sum(MFj,1);
MM = sum(MMj,1);

end






