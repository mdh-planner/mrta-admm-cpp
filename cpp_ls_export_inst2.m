% Auto-generated from C++
% Instance 2

instId = 2;
z_hard = [1 1 1 0 1 1; 1 1 1 1 0 0];
tau = [157 297 230 nan 71 261; 157 297 230 47 nan nan];
isVirtual = [0 0 0 0 0 1];
isMR = [1 1 1 0 0 0];
theta = [157 297 230 9.4000000000000004 14.200000000000001 31];
durTask = [30 41 31 42 13 46];
svcPhysSJ = [30 41 31 42 13 0; 30 41 31 42 13 0];
svcVirtSJ = [0 0 0 0 0 46; 0 0 0 0 0 46];
orders_phys = {[5 1 3 2], [4 1 3 2]};
orders_virt = {[6], []};
W = [0 0 24 78 58 47 71 0 57; 0 0 24 78 58 47 71 0 57; 24 24 0 72 43 68 57 0 56; 78 78 72 0 36 79 29 0 22; 58 58 43 36 0 81 14 0 34; 47 47 68 79 81 0 89 0 57; 71 71 57 29 14 89 0 0 35; 0 0 0 0 0 0 0 0 0; 57 57 56 22 34 57 35 0 0];
endDepot = [9 9];
Tstart = [4.8000000000000007 15.600000000000001 11.600000000000001 9.4000000000000004 14.200000000000001 42.600000000000001];

plot_gantt_mt_per_robot( ...
    z_hard, tau, isVirtual, isMR, theta, durTask, ...
    svcPhysSJ, svcVirtSJ, orders_phys, orders_virt, ...
    W, endDepot, Tstart, instId);
