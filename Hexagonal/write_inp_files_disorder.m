clc; clear; close all;
rng("shuffle")

% Geomtery which can be 'Demi-A','Demi-B','Demi-C','Kag', 'Hex', 'Tri'
lattice = 'Hex';

L = 10;       % Length of each bar
p = 0.05;     % Relative density
el_size = 10*L;% Average mesh size

domain_size = 10*L;

% Create the vertex coordinates of a perfect lattice
Pvert = generate_vertices(lattice,L,domain_size);

% Create matrix where each rows is a bar connecting vertex i to j
Bars = generate_bars(lattice,Pvert,L);

% Find corresponding vertices on left/right and top/bottom edges
[left,right,bottom,top] = find_edges(lattice,Pvert,L);

frun = fopen('run.bat','w');

% Maximum vertex displacement (between 0 and 0.5)
for RL = 0.01:0.01:0.50
    for i=1:20
        % Move each vertex
        Vert = move_vertices(Pvert,RL*L,left,right,bottom,top);

        % Precise width W and heigth H of domain
        W = Vert(right(1),1) - Vert(left(1),1);
        H = Vert(top(1),2) - Vert(bottom(1),2);

        % Calculate the length of each bar
        bar_L = calc_bar_length(Vert,Bars);

        % Calculate bar thickness
        total_L = sum(bar_L);    % total length of all bars
        t = p*W*H/total_L;       % bar thickness

        % Plot geometry
        %figure;
        %plot_bars(Pvert,Bars,'c')
        %plot_bars(Vert,Bars,'b')
        %hold on; plot(Vert([left;right;bottom;top],1), Vert([left;right;bottom;top],2), 'ro')

        % Create mesh
        [Nodes,Elements] = create_mesh(Vert,Bars,el_size);

        % Write input file
        job_name = strcat(lattice,num2str(domain_size/L,'%d'),'L_R',num2str(round(RL*100),'%02d'),'_n',num2str(i,'%02d'));
        write_inp_file(strcat(job_name,'.inp'),Nodes,Elements,left,right,bottom,top,t);

        fprintf(frun,strcat('call abaqus job=',job_name,' interactive\n'));
    end
end
fclose(frun);

%%
function vert = generate_vertices(lattice,L,domain_size)
if strcmp(lattice,'Demi-A')
    u_vert = L*[ 1.,   5.19615221;
        2.,   5.19615221;
        2.5,   4.33012724;
        3.5,   4.33012724;
        0.5,   4.33012724;
        1.,   3.46410155;
        0.5,   2.59807611;
        1.,   1.73205078;
        0.,   1.73205078;
        0.5,  0.866025388;
        1.,           0.;
        1.5,  0.866025388;
        2.5,  0.866025388;
        2.,           0.;
        2.,   1.73205078;
        2.5,   2.59807611;
        3.,   3.46410155;
        2.,   3.46410155;
        1.5,   4.33012724;
        0.,   3.46410155;
        3.5,  0.866025388;
        3.,   1.73205078;
        3.5,   2.59807611];

    % Width and heigth of unit cell
    u_W = 3*L;
    u_H = 5.19615221*L;
elseif strcmp(lattice,'Demi-B')
    u_vert = L*[ 4.2320509,   1.73205078;
        4.7320509,  0.866025388;
        3.7320509,  0.866025388;
        4.2320509,           0.;
        3.2320509,   1.73205078;
        2.7320509,  0.866025388;
        1.86602545,  0.366025418;
        1.86602545,   1.36602545;
        0.866025388,   1.36602545;
        1.36602545,    2.2320509;
        2.36602545,    2.2320509;
        1.86602545,   3.09807611;
        3.2320509,           0.;
        3.2320509,    2.7320509;
        4.2320509,    2.7320509;
        5.09807634,    2.2320509;
        0.366025418,    2.2320509;
        0.866025388,   3.09807611;
        0.,  0.866025388;
        0.866025388,  0.366025418];

    % Width and heigth of unit cell
    u_W = 4.7320509*L;
    u_H = 2.7320509*L;

elseif strcmp(lattice,'Demi-C')
    u_vert = L*[ 5.96410179,   1.86602545;
        6.96410179,   1.86602545;
        6.46410179,    2.7320509;
        6.46410179,           1.;
        5.59807634,          0.5;
        4.59807634,          0.5;
        3.7320509,           0.;
        3.2320509,  0.866025388;
        2.7320509,           0.;
        1.86602545,          0.5;
        0.866025388,          0.5;
        0.,           1.;
        0.5,   1.86602545;
        0.,    2.7320509;
        0.866025388,    3.2320509;
        1.86602545,    3.2320509;
        1.36602545,   2.36602545;
        2.36602545,   2.36602545;
        3.2320509,   2.86602545;
        2.7320509,    3.7320509;
        3.7320509,    3.7320509;
        4.59807634,    3.2320509;
        5.59807634,    3.2320509;
        5.09807634,   2.36602545;
        4.09807634,   2.36602545;
        4.59807634,    4.2320509;
        5.09807634,   1.36602545;
        4.09807634,   1.36602545;
        2.36602545,   1.36602545;
        1.36602545,   1.36602545;
        0.866025388,    4.2320509;
        5.59807634,    4.2320509;
        1.86602545,    4.2320509];

    % Width and heigth of unit cell
    u_W = 6.46410179*L;
    u_H = 3.7320509*L;
elseif strcmp(lattice,'Kag')
    u_vert = L*[ 1., 0.;
                1/2, sin(pi/3);
                3/2, sin(pi/3);
                5/2, sin(pi/3);
                 0., sqrt(3);
                 2., sqrt(3);
                1/2, 3*sin(pi/3);
                3/2, 3*sin(pi/3);
                5/2, 3*sin(pi/3);
                 1., 2*sqrt(3)];
    
    % Width and heigth of unit cell
    u_W = 2.*L;
    u_H = 2*sqrt(3)*L;

elseif strcmp(lattice,'Hex')
    u_vert = L*[cos(pi/6), 0.0;
                      0.0, 0.5;
                  sqrt(3), 0.5;
                      0.0, 1.5;
                  sqrt(3), 1.5;
                cos(pi/6), 2.0;
                cos(pi/6), 3.0];

    % Width and heigth of unit cell
    u_W = sqrt(3)*L;
    u_H = 3*L;

elseif strcmp(lattice,'Tri')
    u_vert = L*[ 0.0, 0.0;
                 1.0, 0.0;
                 0.5, sin(pi/3);
                 1.5, sin(pi/3);
                 0.0, sqrt(3);
                 1.0, sqrt(3)];

    % Width and heigth of unit cell
    u_W = L;
    u_H = sqrt(3)*L;
end

% Calculate the number of unit cells in x and y
nx = round(domain_size/u_W);
ny = round(domain_size/u_H);
if nx==0
    nx = 1;
end
if ny==0
    ny = 1;
end

% initialise list of vertices
n_unit = size(u_vert,1);  % number of vertices in unit cell
vert = zeros(nx*ny*n_unit , 2);
index = 1;    % index to assign values in vert
for iy=1:ny
    for ix=1:nx
        % copy unit cell and offset by dx and dy
        dx = (ix-1)*u_W;
        dy = (iy-1)*u_H;
        vert(index:index+n_unit-1,:) = u_vert + ones(n_unit,1)*[dx,dy];
        index = index + n_unit;
    end
end

% Remove duplicate vertices
vert = uniquetol(vert,1e-6,'ByRows',true);

end

%%
function B = generate_bars(lattice,Pvert,L)
% Number of vertices
n_vert = size(Pvert,1);

% Initialise list of bars B.  This is overestimating size,
% It will be corrected at the end.
B = zeros(8*n_vert,2);
ibar = 1;    % index to fill matrix B

% Going through each vertex
for i=1:n_vert
    % Compute distance between each vertex and vertex i
    dx = Pvert(:,1)-Pvert(i,1);
    dy = Pvert(:,2)-Pvert(i,2);
    dist = sqrt(dx.^2 + dy.^2);

    % Find nodes that are within a distance L of vertex i
    I = find(dist<1.03*L & dist>0.03*L);
    for k=1:length(I)
        % Create bar if index I(k)>i (to avoid each bar of being
        % duplicated).
        if I(k)>i
            B(ibar,:) = [i,I(k)];
            ibar = ibar + 1;
        end
    end
end
% Delete extra rows in B
B(ibar:end,:) = [];

% Remove bars that are duplicated along edges
% Find midpoint of each bar
V1 = Pvert(B(:,1),:);  % coordinates of vertex i
V2 = Pvert(B(:,2),:);  % coordinates of vertex j
mid = V1 + 0.5*(V2-V1);
if strcmp(lattice,'Demi-A')
    % Remove bars at bottom and diagonal bars on the right side
    x_right = max(mid(:,1));
    I = (mid(:,2)<0.03*L | mid(:,1)>x_right-0.03*L);
    B(I,:) = [];
elseif strcmp(lattice,'Demi-B')
    % angle of each bar
    ang = atan2(V2(:,2)-V1(:,2),V2(:,1)-V1(:,1));

    % Remove horizontal bars along bottom edge
    I = (mid(:,2)<0.53*L & abs(ang) < 0.05);
    B(I,:) = [];
elseif strcmp(lattice,'Demi-C')
    % Remove horizontal bars along bottom edge, some at the top, and right
    % side
    x_right = max(mid(:,1));
    y_top = max(mid(:,2));
    I = (mid(:,2)<0.03*L | mid(:,2)>y_top-0.3*L | mid(:,1)>x_right-0.03*L);
    B(I,:) = [];
elseif strcmp(lattice,'Kag')
    % Remove diagonal bars on the right side
    x_right = max(mid(:,1));
    I = (mid(:,1)>x_right-0.03*L);
    B(I,:) = [];
elseif strcmp(lattice,'Hex')
    % Remove bars on the right side
    x_right = max(mid(:,1));
    I = (mid(:,1)>x_right-0.03*L);
    B(I,:) = [];
elseif strcmp(lattice,'Tri')
    % Remove bars at bottom and diagonal bars on the right side
    x_right = max(mid(:,1));
    I = (mid(:,2)<0.03*L | mid(:,1)>x_right-0.03*L);
    B(I,:) = [];
end
end

%%
function plot_bars(Vert,Bars, colour_code)

for i=1:size(Bars,1)
    i1 = Bars(i,1);
    i2 = Bars(i,2);
    plot([Vert(i1,1),Vert(i2,1)] , [Vert(i1,2),Vert(i2,2)] , colour_code);
    hold on;
end
axis equal
end

%%
function [left,right,bot,top] = find_edges(lattice,Vert,L)

% wb*L is the width used to find vertices along the edges of the unit cell
if strcmp(lattice,'Hex')
    wb = 0.03;
else
    wb = 0.52;
end

% Find indices of vertices on the bottom (y<L/2)
bot = find(Vert(:,2) < wb*L);
% Sort bot from left to right
[~,I] = sort(Vert(bot,1));
bot = bot(I);

% Indices of vertices on the top
ymax= max(Vert(:,2));
top = find(Vert(:,2) > ymax-wb*L);
% Sort top from left to right
[~,I] = sort(Vert(top,1));
top = top(I);

% Indices of vertices on the left (x<L/2)
left = find(Vert(:,1) < wb*L);
% Sort top from bottom to top
[~,I] = sort(Vert(left,2));
left = left(I);

% Indices of vertices on the right
xmax= max(Vert(:,1));
right = find(Vert(:,1) > xmax-wb*L);
% Sort top from bottom to top
[~,I] = sort(Vert(right,2));
right = right(I);

if length(bot) ~= length(top)
    disp('Size issue in top/bottom vertices')
elseif length(left) ~= length(right)
    disp('Size issue in left/right vertices')
end
end
%%
function Vert = move_vertices(Pvert,R,left,right,bottom,top)
% number of vertices
n_vert = size(Pvert,1);

% angle and magnitude of displacement to apply
angle = rand(n_vert,1)*2*pi;
R = sqrt(rand(n_vert,1))*R;

% Ensure periodicity: same displacement on left/right and top/bottom
angle(right) = angle(left);
R(right) = R(left);
angle(top) = angle(bottom);
R(top) = R(bottom);

% Move vertices
Vert = Pvert + [R.*cos(angle),R.*sin(angle)];
end
%%
function bar_L = calc_bar_length(Vert,Bars)

x1 = Vert([Bars(:,1)],1);
y1 = Vert([Bars(:,1)],2);
x2 = Vert([Bars(:,2)],1);
y2 = Vert([Bars(:,2)],2);

bar_L = sqrt((x2-x1).^2 + (y2-y1).^2);  % Length of each bar
end

%%
function [Nodes,Elements] = create_mesh(Vert,Bars,el_size)

% Calculate length of each bar
bar_L = calc_bar_length(Vert,Bars);

% Number of elements for each bar
bar_el = round(bar_L/el_size);
% correction in case it is rounded to 0
I = (bar_el==0);
bar_el(I) = 1;

% Initialise list of elements
Elements = zeros(sum(bar_el),2);
i_el = 0;  % index going through Elements

% Nodes begins with all vertices and then additional nodes
n_newNodes = sum(bar_el-1);    % number of new nodes to create
Nodes = [Vert ; zeros(n_newNodes,2)];
i_nodes = size(Vert,1); % index going through newNodes

for i=1:size(Bars,1)
    % Simply copy element if no refinements are needed
    if bar_el(i)==1
        i_el = i_el + 1;
        Elements(i_el,:) = Bars(i,:);
    else
        % Coordinates of the two vertices at the extremities of the bar
        N1 = Vert(Bars(i,1),:);
        N2 = Vert(Bars(i,2),:);

        % For each new node needed
        for k=1:bar_el(i)-1
            % Append new node to the list
            i_nodes = i_nodes + 1;
            Nodes(i_nodes,:) = N1 + k/bar_el(i)*(N2-N1);

            % Add new element to the list
            if k==1
                i_el = i_el + 1;
                Elements(i_el,:) = [Bars(i,1),i_nodes];
            else
                i_el = i_el + 1;
                Elements(i_el,:) = [i_nodes-1 , i_nodes];
            end
        end
        % Add last element to the list
        i_el = i_el + 1;
        Elements(i_el,:) = [i_nodes , Bars(i,2)];
    end
end

end
%%
function write_inp_file(file_name,Nodes,Elements,left,right,bottom,top,t)

% Materials properties
Es = 200000;
vs = 1/3;

% Create new input file
INPfile = fopen(file_name,'w');

% Write heading and part
fprintf(INPfile,'*Heading\n');
fprintf(INPfile,'** Generated with Matlab script\n');
fprintf(INPfile,'*Preprint, echo=NO, model=NO, history=NO, contact=NO\n');
fprintf(INPfile,'**\n');
fprintf(INPfile,'** PARTS\n');
fprintf(INPfile,'**\n');
fprintf(INPfile,'*Part, name=Lattice\n');

% Write nodes
fprintf(INPfile,'*Node\n');
for i = 1:size(Nodes,1)
    fprintf(INPfile,'%7d, %13f, %13f\n',i,Nodes(i,1),Nodes(i,2));
end

% Write elements
fprintf(INPfile,'*Element, type=B23\n');
for i = 1:size(Elements,1)
    fprintf(INPfile,'%7d, %7d, %7d\n',i,Elements(i,1),Elements(i,2));
end

% Write part sets
fprintf(INPfile,'*Nset, nset=EntireLatticeSet, generate\n');
fprintf(INPfile,'   1, %d,   1\n',size(Nodes,1));
fprintf(INPfile,'*Elset, elset=EntireLatticeSet, generate\n');
fprintf(INPfile,'   1, %d,   1\n',size(Elements,1));
fprintf(INPfile,'** Section: BeamSection  Profile: Rect\n');
fprintf(INPfile,'*Beam Section, elset=EntireLatticeSet, material=Elastic, temperature=GRADIENTS, section=RECT\n');
fprintf(INPfile,'1., %f\n',t);
fprintf(INPfile,'0.,0.,1.\n');
fprintf(INPfile,'*End Part\n');

% Write assembly
fprintf(INPfile,'**\n');
fprintf(INPfile,'**\n');
fprintf(INPfile,'** ASSEMBLY\n');
fprintf(INPfile,'**\n');
fprintf(INPfile,'*Assembly, name=Assembly\n');
fprintf(INPfile,'**\n');
fprintf(INPfile,'*Instance, name=Lattice, part=Lattice\n');
fprintf(INPfile,'*End Instance\n');
fprintf(INPfile,'**\n');

% Write assembly sets
for i=1:length(bottom)
    fprintf(INPfile,'*Nset, nset=B%d, instance=Lattice\n',i);
    fprintf(INPfile,' %d,\n',bottom(i));
end
for i=1:length(left)
    fprintf(INPfile,'*Nset, nset=L%d, instance=Lattice\n',i);
    fprintf(INPfile,' %d,\n',left(i));
end
for i=1:length(right)
    fprintf(INPfile,'*Nset, nset=R%d, instance=Lattice\n',i);
    fprintf(INPfile,' %d,\n',right(i));
end
for i=1:length(top)
    fprintf(INPfile,'*Nset, nset=T%d, instance=Lattice\n',i);
    fprintf(INPfile,' %d,\n',top(i));
end

% Write constraint equations
% Nodes on the left/right
for i=1:length(left)
    if i>1
        % Constraint in x
        fprintf(INPfile,'*Equation\n');
        fprintf(INPfile,'4\n');
        fprintf(INPfile,'R%d, 1, 1.\n',i);
        fprintf(INPfile,'L%d, 1, -1.\n',i);
        fprintf(INPfile,'L1, 1, 1.\n');
        fprintf(INPfile,'R1, 1, -1.\n');
    end
    % Constraints in y and rotation
    for dof=[2,6]
        fprintf(INPfile,'*Equation\n');
        fprintf(INPfile,'2\n');
        fprintf(INPfile,'L%d, %d, 1.\n',i,dof);
        fprintf(INPfile,'R%d, %d, -1.\n',i,dof);
    end
end
% contraints for bottom/top
for i=1:length(bottom)
    if i>1
        % Constraint in y
        fprintf(INPfile,'*Equation\n');
        fprintf(INPfile,'4\n');
        fprintf(INPfile,'T%d, 2, 1.\n',i);
        fprintf(INPfile,'B%d, 2, -1.\n',i);
        fprintf(INPfile,'B1, 2, 1.\n');
        fprintf(INPfile,'T1, 2, -1.\n');
    end
    % Constraints in x and rotation
    for dof=[1,6]
        fprintf(INPfile,'*Equation\n');
        fprintf(INPfile,'2\n');
        fprintf(INPfile,'B%d, %d, 1.\n',i,dof);
        fprintf(INPfile,'T%d, %d, -1.\n',i,dof);
    end
end
fprintf(INPfile,'*End Assembly\n');

% Write materials
fprintf(INPfile,'**\n** MATERIALS\n**\n** A linear elastic model\n');
fprintf(INPfile,'*Material, name=Elastic\n');
fprintf(INPfile,'*Elastic\n');
fprintf(INPfile,'%f, %f\n',Es,vs);

% Write Step 1
fprintf(INPfile,'** ----------------------------------------------------------------\n');
fprintf(INPfile,'**\n** STEP: Step-1\n**\n');
fprintf(INPfile,'*Step, name=Step-1, nlgeom=NO, inc=30000\n');
fprintf(INPfile,'*Static\n');
fprintf(INPfile,'1., 1., 0.0001, 1.\n**\n');

% Write boundary conditions
fprintf(INPfile,'** BOUNDARY CONDITIONS\n**\n');
fprintf(INPfile,'** Name: Displ Type: Displacement/Rotation\n');
fprintf(INPfile,'*Boundary\n');
% pinned node
fprintf(INPfile,'B1, 1, 2\n');
% applied displacement
du = 0.01*(Nodes(top(1),2)-Nodes(bottom(1),2));
fprintf(INPfile,'T1, 2, 2, %f\n',du);

% Write field and history outputs
fprintf(INPfile,'**\n** OUTPUT REQUESTS\n**');
fprintf(INPfile,'*Restart, write, frequency=0\n**\n');

fprintf(INPfile,'** FIELD OUTPUT: F-Output-1\n**\n');
fprintf(INPfile,'*Output, field, variable=PRESELECT, number interval=1\n');
fprintf(INPfile,'**\n');

fprintf(INPfile,'** HISTORY OUTPUT: H-Output-1\n**\n');
fprintf(INPfile,'*Output, history\n');
fprintf(INPfile,'*Node Output, nset=T1\n');
fprintf(INPfile,'RF2, U2\n');
fprintf(INPfile,'*End Step\n');

% Close file
fclose(INPfile);
end