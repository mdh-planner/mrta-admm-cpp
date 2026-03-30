% Auto-generated from C++
% Instance 10

instId = 10;
z_hard = [0 1 0 0 0 1 0 1 1 0; 1 0 1 1 0 0 1 0 0 0; 1 0 0 1 1 0 0 0 0 1];
tau = [nan 27 nan nan nan 128 nan 263 200 nan; 40 nan 176 220 nan nan 104 nan nan nan; 40 nan nan 220 156 nan nan nan nan 263];
isVirtual = [0 0 0 0 0 0 0 1 1 1];
isMR = [1 0 0 1 0 0 0 0 0 0];
theta = [40 5.4000000000000012 8 220 7 17.600000000000001 4.4000000000000004 43 24 29];
durTask = [44 40 24 43 29 28 18 36 23 42];
svcPhysSJ = [nan 40 nan nan nan 28 nan 0 0 0; 44 nan 24 43 29 nan 18 0 0 0; 44 nan 24 43 29 nan 18 0 0 0];
svcVirtSJ = [nan 0 nan nan nan 0 nan 36 23 42; 0 nan 0 0 0 nan 0 36 23 42; 0 nan 0 0 0 nan 0 36 23 42];
orders_phys = {[2 6], [1 7 3 4], [1 5 4]};
orders_virt = {[9 8], [], [10]};
W = [0 0 0 40 27 40 57 35 88 22 0 0 0 20; 0 0 0 40 27 40 57 35 88 22 0 0 0 20; 0 0 0 40 27 40 57 35 88 22 0 0 0 20; 40 40 40 0 47 75 89 72 97 20 0 0 0 46; 27 27 27 47 0 32 41 31 61 27 0 0 0 7; 40 40 40 75 32 0 20 4 71 54 0 0 0 29; 57 57 57 89 41 20 0 25 58 68 0 0 0 42; 35 35 35 72 31 4 25 0 74 52 0 0 0 27; 88 88 88 97 61 71 58 74 0 82 0 0 0 68; 22 22 22 20 27 54 68 52 82 0 0 0 0 25; 0 0 0 0 0 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 0 0 0 0 0 0; 20 20 20 46 7 29 42 27 68 25 0 0 0 0];
endDepot = [14 14 14];
Tstart = [8 5.4000000000000004 8 11.4 7 17.600000000000001 4.4000000000000004 54.399999999999999 32 36];

plot_gantt_mt_per_robot( ...
    z_hard, tau, isVirtual, isMR, theta, durTask, ...
    svcPhysSJ, svcVirtSJ, orders_phys, orders_virt, ...
    W, endDepot, Tstart, instId);
