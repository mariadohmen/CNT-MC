textFilename = 'carbonatom_coordinates_260nm.txt'

dimension = 1;
diff=10*10^14 ;           %Diffusion coefficient for diffion of exciton [nm^2 s^-1] (Hertel et al paper,2010)
k_r1= [10^(9), 10^(10), 10^(11)];
k_nr1=[10^(11)];
k_r2= [10^(12), 10^(13), 10^(13)];
k_nr2=[10^(10)];
num_defect = [10, 20, 100, 1000];
num_exciton = 100;
energy_count_mat = zeros(numel(k_r1), numel(k_r2), numel(num_defect), 3);
%Open txt file
fid = fopen(textFilename, 'rt');
Data = textscan(fid,'%f%f%f', 'Headerlines',0);   %d%d%d%','Headerlines',0); 
progress_count = 0;
xData = Data{1}/10;   %in [nm]
yData = Data{2}/10;    %*(3.2802)^(-1);
zData = Data{3}/10;

disp('Initialize exciton generation...')
disp('Progress [%]:')


for d = 1:numel(num_defect)
    defect_pos_vector = zeros(num_defect(d), 3);     
    for i = 1:num_defect(d)
        random_defect_iteration = ceil(numel(xData)*rand);
        defect_pos_vector(i,1) = xData(random_defect_iteration);
        defect_pos_vector(i,2) = yData(random_defect_iteration);
        defect_pos_vector(i,3) = zData(random_defect_iteration);
    end
    
for e1 = 1:numel(k_r1)
for e2 = 1:numel(k_r2)
    
E11_count = 0;
E11s_count = 0;

for l = 1:num_exciton    
    
    tau_r1=(1/k_r1(e1))*log(1/rand);
    tau_nr1=(1/k_nr1)*log(1/rand);


    if dimension > 1
        if tau_r1 > tau_nr1
            diff_length = sqrt(4*diff*tau_nr1);
        else
            
            diff_length = sqrt(4*diff*tau_r1);
        end    
            diff_steps = int16(diff_length/0.1421);
            
        start_val = int16(numel(xData)*rand);
        start_vec = [xData(start_val), yData(start_val), zData(start_val)];
        pos = start_vec;
        move_vector = zeros(diff_steps, 3);

        
    for i = 1:diff_steps
       dist_vec = sqrt((xData-pos(1)).^2+(yData-pos(2)).^2+(zData-pos(3)).^2);
       move_vector(i,1) = pos(1);
       move_vector(i,2) = pos(2);
       move_vector(i,3) = pos(3);
       index_next_step = find(dist_vec < 0.1430 & dist_vec > 0);
       if numel(index_next_step) >= 3
          choose_index = ceil(rand*3);
          pos = [xData(index_next_step(choose_index)), yData(index_next_step(choose_index)), zData(index_next_step(choose_index))];
          defect_dist = sqrt((pos(1)-defect_pos_vector(:,1)).^2+(pos(2)-defect_pos_vector(:,2)).^2+(pos(3)-defect_pos_vector(:,3)).^2);
          if any(defect_dist < 2) > 0
              tau_r2=(1/k_r2(e2))*log(1/rand);
              tau_nr2=(1/k_nr2)*log(1/rand);
              if tau_r2 < tau_nr2 & tau_r2 < diff_length^2/(4*diff)
                 E11s_count = E11s_count + 1;
                 break
              end
              if tau_nr2 < tau_r2 & tau_nr2 < diff_length^2/(4*diff)
                 break 
              end
          end
       end
       if numel(index_next_step) < 3
           break
       end
        if i >= diff_steps
           E11_count = E11_count +1;
        end
    end
    end
    
    
    if dimension <= 1
        if tau_r1 > tau_nr1
            diff_length = sqrt(2*diff*tau_nr1);
        else 
            diff_length = sqrt(2*diff*tau_r1);
        end    
            
        start_val = int16(numel(xData)*rand);
        start_vec = [xData(start_val), yData(start_val), zData(start_val)];
        pos = start_vec;
        endvec = [xData(start_val), yData(start_val), zData(start_val)+ (2*rand-1)*diff_length];
        

        
        
    end
end

energy_count_mat(e1, e2, d, 1) = E11_count/num_exciton;
energy_count_mat(e1, e2, d, 2) = E11s_count/num_exciton;
energy_count_mat(e1, e2, d, 3) = 1 - (E11s_count+E11_count)/num_exciton;
progress_count = progress_count + 1;
% if rem(ceil(progress_count/(numel(num_defect)*numel(k_r1)*numel(k_r2)))*100, 10) <= 0
    way = waitbar(ceil(progress_count/(numel(num_defect)*numel(k_r1)*numel(k_r2))), ['Progress: ' num2str(ceil(progress_count/(numel(num_defect)*numel(k_r1)*numel(k_r2))*100)) '%']);
    close(way);
% end


end
end
end

e11_mat = zeros(numel(k_r1), numel(num_defect));
e11s_mat = zeros(numel(k_r2), numel(num_defect));
e11s_k_mat = zeros(numel(k_r1), numel(k_r2));

for i = 1:numel(energy_count_mat(:,1,1,1))
    for j = 1:numel(energy_count_mat(1,1,:,1))
        e11_mat(i,j) = energy_count_mat(i,1,j,1);
    end
end

for i = 1:numel(energy_count_mat(1,:,1,2))
    for j = 1:numel(energy_count_mat(1,1,:,2))
        e11s_mat(i,j) = energy_count_mat(1,i,j,2);
    end
end

for i = 1:numel(energy_count_mat(1,:,1,2))
    for j = 1:numel(energy_count_mat(:,1,1,2))
        e11s_k_mat(j,i) = energy_count_mat(j,i,4,2);
    end
end

figure, scatter3(move_vector(:,1), move_vector(:,2),move_vector(:,3), '.')
% figure, scatter3(xData, yData, zData, '.');