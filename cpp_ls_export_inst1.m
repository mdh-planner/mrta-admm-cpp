% Auto-generated from C++
% Instance 1

instId = 1;
z_hard = [1 1 1 0 1 0; 1 0 1 1 0 1];
tau = [210 99 28 nan 259 nan; 210 nan 28 149 nan 190];
isVirtual = [0 0 0 0 1 1];
isMR = [1 0 1 0 0 0];
theta = [210 9 28 18.800000000000001 49 49];
durTask = [49 49 33 41 15 16];
svcPhysSJ = [49 49 33 41 0 0; 49 49 33 41 0 0];
svcVirtSJ = [0 0 0 0 15 16; 0 0 0 0 15 16];
orders_phys = {[3 2 1], [3 4 1]};
orders_virt = {[5], [6]};
W = [0 0 107 45 28 94 0 0 61; 0 0 107 45 28 94 0 0 61; 107 107 0 62 96 17 0 0 73; 45 45 62 0 38 50 0 0 42; 28 28 96 38 0 88 0 0 73; 94 94 17 50 88 0 0 0 55; 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 0; 61 61 73 42 73 55 0 0 0];
endDepot = [9 9];
Tstart = [21.400000000000002 9 5.6000000000000005 18.800000000000001 70.400000000000006 58];

plot_gantt_mt_per_robot( ...
    z_hard, tau, isVirtual, isMR, theta, durTask, ...
    svcPhysSJ, svcVirtSJ, orders_phys, orders_virt, ...
    W, endDepot, Tstart, instId);
