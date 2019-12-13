%% Declare variables
sample_num = 10^5;
window_size = 5;
    
%% Create convolutional code with specified trellis and Send the data over Binary Symmetric Channel with Crossover Probabilty p
catastrophic_trellis = poly2trellis(3,[3, 5, 6 0]);
msg = randi([0 1], 1, sample_num - 2); %Create sample_num-2 random binary sample since last 2 will be zero
msg(sample_num-1) = 0;
msg(sample_num) = 0;
transmitted_code = convenc(msg,catastrophic_trellis); %Create convolutional code
transmitted_code = bpsk(transmitted_code); 
received_code = awgn(transmitted_code, snr, 'measured'); %Send data over AWGN Channel



%% Viterbi Decoder
received_uncoded = oneBitQuantize(received_uncoded);
received_code = reshape(received_code, [4, sample_num]); %change format such that colunms are output at t=t0
next_state_table = [0 0 0 0 0 0; 0 1 0 1 1 2; 1 1 1 1 0 0; 1 0 1 0 1 2; 2 1 0 0 1 1; 2 0 0 1 0 3; 3 0 1 1 1 1; 3 1 1 0 0 3];
last_state_table = -1*ones(4,window_size);
last_state = 0;
traced = -1*ones(1, window_size+1);
cost_table = -1*ones(4,window_size+1);
decoded = zeros(1, sample_num);


for i = 1:1:sample_num-window_size
	last_state_table = -1*ones(4,window_size);
	cost_table = -1*ones(4,window_size+1);
	cost_table(last_state+1,1) = 0;
	%First trellis
	temp_received = transpose(received_code(:,i));
	if last_state == 0
		cost_table(next_state_table(1,6)+1, 2) = euclidianDistance(temp_received, next_state_table(1,2:5));
		last_state_table(next_state_table(1,6)+1, 1) = 0;
		cost_table(next_state_table(2,6)+1, 2) = euclidianDistance(temp_received, next_state_table(2,2:5));
		last_state_table(next_state_table(2,6)+1, 1) = 0;
	elseif last_state == 1
		cost_table(next_state_table(3,6)+1, 2) = euclidianDistance(temp_received, next_state_table(3,2:5));
		last_state_table(next_state_table(3,6)+1, 1) = 1;
		cost_table(next_state_table(4,6)+1, 2) = euclidianDistance(temp_received, next_state_table(4,2:5));
		last_state_table(next_state_table(4,6)+1, 1) = 1;
	elseif last_state == 2
		cost_table(next_state_table(5,6)+1, 2) = euclidianDistance(temp_received, next_state_table(5,2:5));
		last_state_table(next_state_table(5,6)+1, 1) = 2;
		cost_table(next_state_table(6,6)+1, 2) = euclidianDistance(temp_received, next_state_table(6,2:5));
		last_state_table(next_state_table(6,6)+1, 1) = 2;
	else
		cost_table(next_state_table(7,6)+1, 2) = euclidianDistance(temp_received, next_state_table(7,2:5));
		last_state_table(next_state_table(7,6)+1, 1) = 3;
		cost_table(next_state_table(8,6)+1, 2) = euclidianDistance(temp_received, next_state_table(8,2:5));
		last_state_table(next_state_table(8,6)+1, 1) = 3;
	end;
	
	%Second trellis
	temp_received = transpose(received_code(:,i+1));
	if cost_table(1,2) ~= -1
		cost_table(next_state_table(1,6)+1, 3) = cost_table(1,2) + euclidianDistance(temp_received, next_state_table(1,2:5));
		last_state_table(next_state_table(1,6)+1, 2) = 0;
		cost_table(next_state_table(2,6)+1, 3) = cost_table(1,2) + euclidianDistance(temp_received, next_state_table(2,2:5));
		last_state_table(next_state_table(2,6)+1, 2) = 0;
	end;
	if cost_table(2,2) ~= -1
		cost_table(next_state_table(3,6)+1, 3) = cost_table(2,2) + euclidianDistance(temp_received, next_state_table(3,2:5));
		last_state_table(next_state_table(3,6)+1, 2) = 1;
		cost_table(next_state_table(4,6)+1, 3) = cost_table(2,2) + euclidianDistance(temp_received, next_state_table(4,2:5));
		last_state_table(next_state_table(4,6)+1, 2) = 1;
	end;
	if cost_table(3,2) ~= -1
		cost_table(next_state_table(5,6)+1, 3) = cost_table(3,2) + euclidianDistance(temp_received, next_state_table(5,2:5));
		last_state_table(next_state_table(5,6)+1, 2) = 2;
		cost_table(next_state_table(6,6)+1, 3) = cost_table(3,2) + euclidianDistance(temp_received, next_state_table(6,2:5));
		last_state_table(next_state_table(6,6)+1, 2) = 2;
	end;
	if cost_table(4,2) ~= -1
		cost_table(next_state_table(7,6)+1, 3) = cost_table(4,2) + euclidianDistance(temp_received, next_state_table(7,2:5));
		last_state_table(next_state_table(7,6)+1, 2) = 3;
		cost_table(next_state_table(8,6)+1, 3) = cost_table(4,2) + euclidianDistance(temp_received, next_state_table(8,2:5));
		last_state_table(next_state_table(8,6)+1, 2) = 3;
	end;
	
	%Rest trellisses
	for k = 1:1:window_size-2
		temp_received = transpose(received_code(:,i+k+1));
		%making decisions for 0 state
		dist1 = cost_table(1,k+2) + euclidianDistance(temp_received, next_state_table(1,2:5));
		dist2 = cost_table(2,k+2) + euclidianDistance(temp_received, next_state_table(3,2:5));
		if dist1 < dist2
			last_state_table(1,k+2) = 0;
		else
			last_state_table(1,k+2) = 1;
		end;
		cost_table(1,k+3) = min(dist1,dist2);
		%making decisions for 1 state
		dist1 = cost_table(3,k+2) + euclidianDistance(temp_received, next_state_table(5,2:5));
		dist2 = cost_table(4,k+2) + euclidianDistance(temp_received, next_state_table(7,2:5));
		if dist1 < dist2
			last_state_table(2,k+2) = 2;
		else
			last_state_table(2,k+2) = 3;
		end;
		cost_table(2,k+3) = min(dist1,dist2);
		%making decisions for 2 state
		dist1 = cost_table(1,k+2) + euclidianDistance(temp_received, next_state_table(2,2:5));
		dist2 = cost_table(2,k+2) + euclidianDistance(temp_received, next_state_table(4,2:5));
		if dist1 < dist2
			last_state_table(3,k+2) = 0;
		else
			last_state_table(3,k+2) = 1;
		end;
		cost_table(3,k+3) = min(dist1,dist2);
		%making decisions for 3 state
		dist1 = cost_table(3,k+2) + euclidianDistance(temp_received, next_state_table(6,2:5));
		dist2 = cost_table(4,k+2) + euclidianDistance(temp_received, next_state_table(8,2:5));
		if dist1 < dist2
			last_state_table(4,k+2) = 2;
		else
			last_state_table(4,k+2) = 3;
		end;
		cost_table(4,k+3) = min(dist1,dist2);
	end;
	
	[m, n] = min(cost_table(:,window_size+1));  %now n has the state with minimum cost
	traced(window_size+1) = n-1;
	for k = window_size:-1:1
		traced(k) = last_state_table(traced(k+1)+1,k);
	end;
	
	if traced(1) == 0 && traced(2) == 0
		decoded(i) = 0;
	elseif traced(1) == 3 && traced(2) == 3
		decoded(i) = 1;
	elseif traced(2) > traced(1)
		decoded(i) = 1;
	else
		decoded(i) = 0;
	end;
	%     cost_table
	%     last_state_table
	%     traced
	last_state = traced(2);

	
end;






function y = hammingDistance(m,n)
    y = 0;
    for i = 1:1:length(m)
        if m(i) ~= n(i)
            y = y + 1;
        end
    end
end

function y = euclidianDistance(m,n)
    y = 0;
    for i = 1:1:length(m)
        y = y + (m(i)-n(i))^2;
    end;
    y = sqrt(y);
end

function y = bpsk(x)
    y = zeros(1,length(x));
    for i = 1:1:length(x)
        if x(i) == 0
            y(i) = -1;
        else
            y(i) = 1;
        end
    end
end

function y = oneBitQuantize(x)
    y = zeros(1,length(x));
    for i = 1:1:length(x)
        if x(i) >= 0
            y(i) = 1;
        else
            y(i) = 0;
        end
    end
end

function y = twoBitQuantize(x)
    y = zeros(1,length(x));
    for i = 1:1:length(x)
        if x(i) <= -0.5
            y(i) = -0.75;
        elseif x(i) <= 0
            y(i) = -0.25;
        elseif x(i) <= 0.5
            y(i) = 0.25;
        else
            y(i) = 0.75;
        end       
    end
end
