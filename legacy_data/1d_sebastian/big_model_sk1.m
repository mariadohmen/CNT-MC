

% This function calls the exciton_fate function and explores the effect of
% different external parameters on fluorescence spectra 
n_photons=100;
l=300;
r=1;
n_defects=10;
t=1;
table1=exciton_dynamics(n_photons,l,r,n_defects,t);
table2=exciton_dynamics(n_photons,l*2,r,n_defects,t);
x_l=800;
x_r=1400;
FWHM1=30;
x0=980;
x1=1130;
FWHM2=30;
h=1;

%Before 
f11=lorentzian_function(FWHM1,h,x_l,x_r,x0);
f12=lorentzian_function(FWHM2,h,x_l,x_r,x1);
f1=f11+f12;




figure(1);
subplot(1,1,1); 


plot(f1(:,1),f1(:,2),'k');
axis([x_l x_r 0 3]);
hold on; 
h=table1(1,1)/table2(1,1);
f11=lorentzian_function(FWHM1,h,x_l,x_r,x0);

h=table1(2,1)/table2(2,1);
f12=lorentzian_function(FWHM2,h,x_l,x_r,x1);
f1=f11+f12;


plot(f1(:,1),f1(:,2),'r');
