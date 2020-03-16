clear all
long=260;

r_ex_li=linspace(0,1,1);  %radius of exciton
n_def_li=linspace(0,100,101); %number of defects on nanotube
%r_def=2 ;   %radius of defect-influencearea in angström 
            % muss ganzzahlig sein
%p_que=0.5;   %percentage, that has to be covered with defect influenced area, to create new quenching point
p_ph2_li=linspace(0.02,0.8,40);   %probability for a photon with lower energy to be emmited when a defect is reached

diff=10*10^14 ;           %Diffusion coefficient for diffion of exciton [nm^2 s^-1] (Hertel et al paper,2010)
n_ex=100;                 %number of excitons !will some day be calculated!
tau_bind=0   ;          %percentage occupied binding site influences diffusiontime of exciton
quench= 2 ;               %number of quenching sites (=2 if only the tube ends quench the exciton)
dq=long/(quench-1);       %average distance between two quenching sites

int=0;         
k_r=10^(10);            %constant for radiativ decay
k_nr=[10^(11)];         %constant of non radiativ decay

quyi=zeros(numel(n_def_li),3); %table for plotting the quantum yield for different k_r and k_nr values.
upm=zeros(numel(r_ex_li)*numel(p_ph2_li),8);%ultimate plot matrix, saving for a defect radiusand a number of defects the slope of the first part of the graph
                                             %and the  constant b of the
                                             %fit a*exp(b*x) for each kind
                                             %of photon

n_ph2_li=numel(p_ph2_li);
n_ex_li=numel(r_ex_li);

hmint1=zeros(n_ex_li, n_ph2_li);
hmint2=zeros(n_ex_li, n_ph2_li); %HeatMap for the max. INTensity of photon 2 for different r  and p_ph2 values
count=0;
const=1;
for rrex=1:n_ex_li
    r_ex=r_ex_li(1,rrex);
    %disp(rrex)
for ppph2=1:1%n_ph2_li
    p_ph2=p_ph2_li(1,ppph2);
    %disp(ppph2)
    count=count+1;
    disp(count)
    disp('von 840')
for defect=1:numel(n_def_li)
    disp(defect)
    n_def=n_def_li(1,defect);

    
    fileID2= dlmread('/Users/Seb1Work/Dropbox/python/Exciton diffusion model/carbonatom_coordinates_260nm.txt');
    fileID=fileID2+1;

    mid_def=zeros(n_def,1);
    tramid=zeros(n_def+2,3);
%    tramid(n_def,1)=1000;


   
int1=0; %intensity of more enegetic photon
int2=0; %intensity of less energetic photon

%range of each binding site in nm

for ave=1:10
for ex=1:n_ex
%point of exciton-appearance
spw=rand*long;      %(place)

ph1=0;
%calculates position of defects
for pic=1:n_def
    cpic=round(rand*numel(fileID(:,1)));
    if cpic==0
        cpic=1;
    end
  
    y=round(fileID(cpic,3));
    if y==0
        y=1;
    end
    
    mid_def(pic,1)=y;
end
     

%brings defekts in order and from angström to nanometers
tramid(1:n_def, 1)=sort(mid_def)/10;

%calculates distance between defects
for bt=1:n_def-1
    tramid(bt,2)=tramid(bt+1,1)-tramid(bt,1); %for direc =1
    tramid(bt+1,3)=tramid(bt+1,1)- tramid(bt,1); % for direc=-1
end
%exiton in 1D model can move in two directions
direction=rand;  

    if direction>0.5
        direc=-1    ;                %exciton moves "left"
        leng=spw-r_ex;
    else
        direc=1             ;        %exciton moves "right"
        leng=long-spw-r_ex;
    end
        
%calculate lifetime of exciton.therefor a lifetime for tau_n and tau_nr is calculated. Then the lifetime
%of the exciton is compared to the time quenching at an nanotubeend would
%take place.

tau_r=(1/k_r)*log(1/rand);

tau_nr=(1/k_nr)*log(1/rand);

tau_ges=(leng^2)/(2*diff);

%looks for smallest tau value
if tau_r <tau_ges && tau_r < tau_nr
    tauau=tau_r;
    ph1=1;
    
elseif tau_nr <tau_ges && tau_nr < tau_r
    tauau=tau_nr;
    ph1=0;
elseif tau_ges <tau_r && tau_ges < tau_nr
    tauau=tau_ges;
    ph1=0;
end 
%calculates way the exciton moves in tau
pass=sqrt(2*diff*tauau);
fin=spw+(direc*pass)-direc*r_ex;
if n_def>0
%calculates new quenching points by calculating if the distance between two
%defects is bigger or smaller then the exciton radius
 for zr=1:n_def
    if tramid(zr,2)<=r_ex
        tramid(zr,2)=1;
        tramid(zr+1,3)=1;
    elseif tramid(zr,2)>r_ex
        tramid(zr,2)=0;
        tramid(zr+1,3)=0;
    end
 end


if direc==1 
    for vab=1:n_def
      if all([tramid(vab,1)>=spw, tramid(vab,1)<=fin, tramid(vab,2)==0])
        ph1=0;
        supi=rand;
        if supi <= p_ph2
          int2=int2+1;
          break
        end
       elseif all([tramid(vab,1)>=spw, tramid(vab,1)<=fin, tramid(vab,2)==1])
        ph1=0;
        break
       end
      end
elseif direc==-1
    for vab=n_def:-1:1
      if all([tramid(vab,1)<=spw, tramid(vab,1)>=fin, tramid(vab,3)==0])
        ph1=0;
        supi=rand;
        if supi <= p_ph2
          int2=int2+1;
          break
        end
       elseif all([tramid(vab,1)<=spw, tramid(vab,1)>=fin, tramid(vab,3)==1])
        ph1=0;
        break
       end
    end
    
end
end
if ph1==1
   int1=int1+1;


end
end
end

%disp(int);
qy1=int1/(n_ex);%calculates quantum yield of high energy photon.
qy2=int2/(n_ex);%calculates quantumyiel of lower energy photons
quyi(defect,1)=n_def;
quyi(defect,2)=quyi(defect,2)+qy1;
quyi(defect,3)=quyi(defect,3)+qy2;

end
%end
quyi(:,2)=quyi(:,2)/10;
quyi(:,3)=quyi(:,3)/10;
quyi(:,2)=smooth(quyi(:,2),10);
quyi(:,3)=smooth(quyi(:,3),10);

%nom=perc_li(1,pho2int);
name='quyi_model3%d_%d.txt' ;
str=sprintf(name,p_ph2,r_ex);
dlmwrite(str,quyi);

maxiq1=max(quyi(:,2));
maxiq=max(quyi(:,3));
[mph1r,mph1c]=find(quyi(:,2)==maxiq1);
[mph2r,mph2c]=find(quyi(:,3)==maxiq);
hmint1(rrex,ppph2)=quyi(mph1r,2);
if numel(mph2r)>1
    mph2r=1;
end
if quyi(mph2r,3)==0
    hmint2(rrex,ppph2)=0
else
    hmint2(rrex,ppph2)=quyi(mph2r,3);
end
    
pf=polyfit(quyi(1:mph2r,1),quyi(1:mph2r,3),1);

fq1=fit(quyi(:,1),quyi(:,2),'exp1');
if mph2r==numel(quyi(:,1))
    mph2r=mph2r-1;
%elseif mph2r==0
    
end
fq2=fit(quyi((mph2r):numel(quyi(:,1)),1),quyi((mph2r):numel(quyi(:,1)),3),'exp1');
upm(const,1)=r_ex;
upm(const,2)=p_ph2;
upm(const,3)=quyi(mph1r,2);
upm(const,4)=quyi(mph2r,3);
upm(const,5)=quyi(mph2r,1);
upm(const,6)=pf(2);
upm(const,7)=fq1.b;
upm(const,8)=fq2.b;
const=const+1;

    end
end

figure
imagesc(hmint2);
xlabel('probability for radiative decay when exciton meets defect[%]');
ylabel('radius of exciton[Â]');
title('Maximum Intensity of Low-Energy Photon');
colorbar;

figure
imagesc(hmint1);
xlabel('probability for radiative decay when exciton meets defect[%]');
ylabel('radius of exciton[Â]');
title('Maximum Intensity of High-Energy Photon');
colorbar;

hm_n_def_max=zeros(n_ex_li,n_ph2_li);
hm_linplo=zeros(n_ex_li,n_ph2_li);
hm_expo1=zeros(n_ex_li,n_ph2_li);
hm_expo2=zeros(n_ex_li,n_ph2_li);

for zr=1:n_ex_li
    for zph2=1:n_ph2_li
        hm_n_def_max(zr,zph2)=upm((n_ph2_li*zr-(n_ph2_li-zph2)),5);
        hm_linplo(zr,zph2)=upm((n_ph2_li*zr-(n_ph2_li-zph2)),6);
        hm_expo1(zr,zph2)=upm((n_ph2_li*zr-(n_ph2_li-zph2)),7);
        hm_expo2(zr,zph2)=upm((n_ph2_li*zr-(n_ph2_li-zph2)),8);
    end
end
figure
imagesc(hm_n_def_max);
xlabel('probability for radiative decay when exciton meets defect[%]');
ylabel('radius of exciton [Â]');
title('Number of Defects at Intensity-Maximum');
colorbar;

figure
imagesc(hm_linplo);
xlabel('probability for radiative decay when exciton meets defect[%]');
ylabel('radius of exciton [Â]');
title('Slope Fit for linear part of Function');
colorbar;

figure
imagesc(hm_expo1);
xlabel('probability for radiative decay when exciton meets defect[%]');
ylabel('radius of exciton [Â]');
title('Constant of exponential Fit of High-Energy Photons');
colorbar;
 
figure
imagesc(hm_expo2);
xlabel('probability for radiative decay when exciton meets defect[%]');
ylabel('radius of exciton [Â]');     
title('Constant of exponential Fit of the second Part of the Low-Energy Photon');
colorbar;

dlmwrite('upm_exciton_model_3_kr_10^10.txt',upm)
dlmwrite('upm_exciton_model_3_kr_10^10_n_def_max.txt',hm_n_def_max)
dlmwrite('upm_exciton_model_3_kr_10^10_exp_const_ph2.txt',hm_expo2)
dlmwrite('upm_exciton_model_3_kr_10^10_exp_const_ph1.txt',hm_expo1)
dlmwrite('upm_exciton_model_3_kr_10^10_max_int_ph1.txt',hmint1)
dlmwrite('upm_exciton_model_3__10^10_max_int_ph2.txt',hmint2)