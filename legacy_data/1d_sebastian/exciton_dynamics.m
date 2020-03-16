function exciton_dynamics=exciton_dynamics(n_photons,l,r,n_defects,t_step)


%l length of the nanotube [nm]
% r radius of exciton [nm]
% n_defects number of defects
 %t_step time step in ps for the simulation

diff=10^15;           %Diffusion coefficient for diffion of exciton [nm^2 s^-1] (Hertel et al paper,2010)
tau=100; %FLuorescence E11 lifetime in ps     
tau2=1000;%FLuorescence E11* lifetime

k_r=10^11;            %constant for radiativ decay
k_nr=10^12;         %constant of non radiativ decay
%k_d=10^11;              %constant for going into dark state
k_d_b=10^10; 
k_d_nr=10^12;

k_nothing_e11=(k_r+k_nr)*tau/t_step;
k_nothing_e11star=(k_d_b+k_d_nr)*tau2/t_step;

%Create a number of n defects at integer positions in the nanotube of length
%l 

    rand1=l*rand(1,n_defects);
    pos_defects=round(rand1)
  

%We use always positive direction because the problem
%is symmetric

% Use 
%There is an exciton generated and lives...
%Exciton fate contains (E11 excitons, 

exciton_fate=zeros(7,1); 

for i=1:n_photons 
fate=5; 

%Create an exciton and perform Monte-Carlo
    rand_exc=rand
    rand_exc_pos=round(l*rand_exc)

    %position of the exciton at the beginning of the diffusive walk
    pos_exc_0=rand_exc_pos;

    %exciton is not trapped in the beginning
    trapped=0;

    
while fate>4
 
    %Calculate position of the exciton in nm
    if trapped==0
        
    pos_exc_1=pos_exc_0+(2*diff*t_step*10^(-12))^0.5;
    pos_exc_1=round(pos_exc_1);
   
    end
    
    %if the exciton is generated at the end of the tube it is directly quenched
    
    if or(pos_exc_1==l, pos_exc_1>l)
     %non radiative decay fate index = 3
    exciton_fate(3,1)= exciton_fate(3,1) + 1;       
    
    else
    
   %check if the exciton hits a defect
    for j=1:size(pos_defects)
    
        for w=0:(pos_exc_1-pos_exc_0)
            pos_test=w+pos_exc_0;
            w=w+1;
        if pos_test==pos_defects(j)
        
            trapped=1; 
             fate=5; 
     
            exciton_fate(5,1)= exciton_fate(5,1) + 1;  
        end
        
        end
    end
    end
    
    if trapped==1
    %this means the exciton is in the defect and there are only a few possibilities left    
        
    p_d_b=k_d_b*rand; %E11* emission from defect    
    p_d_nr=k_d_nr*rand;% non radiative dissipation
    p_d_nothing=k_nothing_e11star*rand; %exciton trapped in defect survives. 
    
    list_p=[p_d_b,p_d_nr,p_d_nothing];
    max_p=max(list_p);
    
    
    
     %radiative E11* decay fate=2
    if max_p==p_d_b
        fate=2; 
        exciton_fate(2,1)= exciton_fate(2,1) + 1; 
    end
    
   
    % nr dissipation from defect state fate =4
    if max_p==p_d_nr
        fate=4;   
        exciton_fate(4,1)= exciton_fate(4,1) + 1;   
    end
    
     % exciton stays in defect state =7
    if max_p==p_d_nothing
        fate=7;   
        exciton_fate(7,1)= exciton_fate(7,1) + 1;   
    end
    
    
    
    else
        
    %exciton diffuses around   
        
    
    p_nr=k_nr*rand; %non radiative decay
    p_r=k_r*rand;%radiative decay

    %p_d=k_d*rand; %exciton goes to dark state   %not used because we use a
    %diffusive walk
    p_nothing_e11=k_nothing_e11*rand;  %exciton diffuses and nothing happens
    
    list_p=[p_nr,p_r,p_nothing_e11];
    max_p=max(list_p);
    
    %radiative E11 decay fate=1
    if max_p==p_r
        fate=1; 
        exciton_fate(1,1)= exciton_fate(1,1) + 1; 
    end
    
    
    %non radiative decay fate=3
    if max_p==p_nr 
        fate=3;
        exciton_fate(3,1)= exciton_fate(3,1) + 1; 
    end
   
    
    %exciton falls in trap fate=5;
    %if max_p==p_d
    %
     %   fate=5; 
      %  trapped=1;
       % exciton_fate(5,1)= exciton_fate(5,1) + 1;  
    %end
    %exciton continues diffusive walk; 
    
    if max_p==p_nothing_e11
        fate=6;   
        exciton_fate(6,1)= exciton_fate(6,1) + 1;   
    end
    
    

    end
    
 %set new zero position of the exciton
 pos_exc_0=pos_exc_1;
end


 
end

QY1=exciton_fate(1,1)/n_photons;
QY2=exciton_fate(2,1)/n_photons;


exciton_dynamics=exciton_fate

end
