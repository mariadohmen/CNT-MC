function lorentzian_function= lorentzian_function(FWHM,h,x_l,x_r,x0)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here


for i=x_l:x_r;
x(i)=i;   
y(i)=h/(1+((x(i)-x0)/(FWHM/2))^2);
lorentzian_function(i,1)=x(i); 
lorentzian_function(i,2)=y(i); 
end

end

