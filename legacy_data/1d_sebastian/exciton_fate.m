function exciton_dynamics=exciton_dynamics()
clear all

l=300; %length of the nanotube [nm]
s= 1 %radius of exciton [nm]
n=10; %number of defects
t_step=1 %time steps in ps

diff=10*10^14 ;           %Diffusion coefficient for diffion of exciton [nm^2 s^-1] (Hertel et al paper,2010)
       
k_r=10^(10);            %constant for radiativ decay
k_nr=10^(11);         %constant of non radiativ decay
k_d=10^11;              %constant for going into dark state
%k_b=
k_n=10^11; 


%Create a number of n defects at integer positions in the nanotube of length
%l 

    rand1=l*rand(1,10);
    pos_defects=round(rand1)
  
%Create an exciton and perform Monte-Carlo
    rand_exc=rand
    rand_exc_pos=round(l*rand_exc)

%We use always positive direction because the problem
%is symmetric

% Use 
%There is an exciton generated and lives...
%Exciton fate contains (E11 excitons, 
n_photons=1000;
exciton_fate=zeros(4,1); 

for i=1:n_photons 
fate=4; 


while fate>2
 
 
    
    p_nr=k_nr*rand %non radiative decay
    p_r=k_r*rand %radiative decay
    p_d=k_d*rand %exciton goes to dark state
    p_nothing=k_n*rand  %exciton diffuses and nothing happens
    
    list_p=[p_nr,p_r,p_d,p_nothing]
    max_p=max(list_p)
    
  
    %non radiative decay fate=
    if max_p==p_nr 
        fate=1;
        exciton_fate(1,1)= exciton_fate(1,1) + 1 
    end
     %radiative E11 decay fate=2
    if max_p==p_r
        fate=2; 
        exciton_fate(2,1)= exciton_fate(2,1) + 1 
    end
         %exciton falls in trap fate=3; 
    if max_p==p_d
    
        fate=3; 
        exciton_fate(3,1)= exciton_fate(3,1) + 1  
    end
    %exciton continues diffusive walk; 
    
    if max_p==p_nothing
        fate=4;   
        exciton_fate(4,1)= exciton_fate(4,1) + 1   
    end
    
    
end


end

QY=exciton_fate(2,1)/n_photons


exciton_fate=e


%Check Quantum yield


%figure
%imagesc(hm_expo2);
%xlabel('probability for radiative decay when exciton meets defect[%]');
%ylabel('radius of exciton [Â]');     
%title('Constant of exponential Fit of the second Part of the Low-Energy Photon');
%colorbar;

end

