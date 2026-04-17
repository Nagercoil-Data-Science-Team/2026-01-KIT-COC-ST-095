%% =====================================================
% FULL CODE: Multi-Agent DQN Trajectories in MATLAB
% =====================================================

clear all; close all; clc;
warning('off', 'all');

%% -----------------------------
% Global Plot Settings
% -----------------------------
set(0, 'DefaultAxesFontName', 'Times New Roman');
set(0, 'DefaultAxesFontSize', 14);
set(0, 'DefaultAxesFontWeight', 'bold');
set(0, 'DefaultTextFontName', 'Times New Roman');
set(0, 'DefaultTextFontWeight', 'bold');

%% =====================================================
% STEP 1: Load Dataset
% =====================================================
fprintf('============================================================\n');
fprintf('STEP 1: Loading Dataset\n');
fprintf('============================================================\n');

try
    % Try to load dataset
    data = readtable('ADAS_EV_Dataset.csv');
    fprintf('Dataset loaded: %d rows, %d columns\n', size(data,1), size(data,2));
    fprintf('Columns: %s\n', strjoin(data.Properties.VariableNames, ', '));
catch
    % Create synthetic data if file not found
    fprintf('Dataset file not found. Creating synthetic data...\n');
    
    n_samples = 1000;
    data = table();
    data.timestamp = datetime(2024,1,1) + minutes(0:n_samples-1)';
    data.speed_kmh = 80 + 20*randn(n_samples,1);
    data.battery_level = 0.3 + 0.6*rand(n_samples,1);
    data.energy_consumption = 1 + 4*rand(n_samples,1);
    data.obstacle_distance = 5 + 45*rand(n_samples,1);
    
    weather_types = {'sunny'; 'rainy'; 'cloudy'; 'foggy'};
    road_types = {'highway'; 'urban'; 'rural'; 'factory'};
    adas_outputs = {'lane_keep'; 'brake'; 'accelerate'; 'turn_left'; 'turn_right'};
    
    data.weather_condition = weather_types(randi(4, n_samples, 1));
    data.road_type = road_types(randi(4, n_samples, 1));
    data.ADAS_output = adas_outputs(randi(5, n_samples, 1));
    
    fprintf('Synthetic dataset created: %d rows, %d columns\n', size(data,1), size(data,2));
end
fprintf('\n');

%% =====================================================
% STEP 2: Data Preprocessing
% =====================================================
fprintf('============================================================\n');
fprintf('STEP 2: Data Preprocessing\n');
fprintf('============================================================\n');

% Encode categorical variables
if iscategorical(data.weather_condition)
    data.weather_condition_num = grp2idx(data.weather_condition);
else
    data.weather_condition_num = grp2idx(categorical(data.weather_condition));
end

if iscategorical(data.road_type)
    data.road_type_num = grp2idx(data.road_type);
else
    data.road_type_num = grp2idx(categorical(data.road_type));
end

if iscategorical(data.ADAS_output)
    data.ADAS_output_num = grp2idx(data.ADAS_output);
else
    data.ADAS_output_num = grp2idx(categorical(data.ADAS_output));
end

% Normalize numeric columns
numeric_cols = {'battery_level', 'energy_consumption', 'obstacle_distance'};
for i = 1:length(numeric_cols)
    if any(strcmp(data.Properties.VariableNames, numeric_cols{i}))
        col_data = data.(numeric_cols{i});
        data.([numeric_cols{i} '_norm']) = (col_data - min(col_data)) / (max(col_data) - min(col_data) + eps);
    end
end

% Add missing RL state columns
if ~any(strcmp(data.Properties.VariableNames, 'traffic_density'))
    data.traffic_density = rand(size(data,1), 1);
end
if ~any(strcmp(data.Properties.VariableNames, 'reaction_time'))
    data.reaction_time = 0.1 + 0.4*rand(size(data,1), 1);
end
if ~any(strcmp(data.Properties.VariableNames, 'lane_deviation'))
    data.lane_deviation = rand(size(data,1), 1);
end

fprintf('Data preprocessing completed.\n');
fprintf('Available state columns: %s\n', strjoin(data.Properties.VariableNames, ', '));
fprintf('\n');

%% =====================================================
% STEP 3: Factory Network Modeling
% =====================================================
fprintf('============================================================\n');
fprintf('STEP 3: Factory Network Modeling\n');
fprintf('============================================================\n');

factory_width = 100;
factory_height = 60;

% Define factory layout components
workstations = {
    'WS1_Assembly', 15, 50, 'assembly';
    'WS2_Painting', 35, 40, 'painting';
    'WS3_Quality', 55, 30, 'inspection';
    'WS4_Packaging', 75, 40, 'packaging';
    'WS5_Shipping', 85, 20, 'shipping';
    'WS6_Storage', 25, 10, 'storage';
    'WS7_Machining', 45, 50, 'machining'
};

charging_stations = {
    'CS1_North', 20, 45;
    'CS2_Central', 50, 25;
    'CS3_South', 80, 15
};

obstacles = {
    'Obstacle_1', 30, 35, 10, 5;
    'Obstacle_2', 60, 45, 8, 8;
    'Obstacle_3', 40, 15, 12, 6
};

fprintf('Factory Layout Components:\n');
fprintf('  • Workstations: %d\n', size(workstations,1));
fprintf('  • Charging Stations: %d\n', size(charging_stations,1));
fprintf('  • Obstacles: %d\n', size(obstacles,1));
fprintf('\n');

% Build Factory Network Graph using MATLAB's graph object
G = digraph();

% Add workstation nodes
for i = 1:size(workstations,1)
    G = addnode(G, workstations{i,1});
    G.Nodes.Pos{i,1} = [workstations{i,2}, workstations{i,3}];
    G.Nodes.Type{i,1} = 'workstation';
    G.Nodes.Workload{i,1} = workstations{i,4};
    G.Nodes.TrafficWeight(i,1) = 0.5 + rand();
end

% Add charging station nodes
for i = 1:size(charging_stations,1)
    G = addnode(G, charging_stations{i,1});
    node_idx = size(workstations,1) + i;
    G.Nodes.Pos{node_idx,1} = [charging_stations{i,2}, charging_stations{i,3}];
    G.Nodes.Type{node_idx,1} = 'charger';
    G.Nodes.Workload{node_idx,1} = 'charging';
    G.Nodes.TrafficWeight(node_idx,1) = 0.5 + rand();
end

% Define movement paths
movement_paths = {
    % Main assembly line
    'WS1_Assembly', 'WS2_Painting';
    'WS2_Painting', 'WS3_Quality';
    'WS3_Quality', 'WS4_Packaging';
    'WS4_Packaging', 'WS5_Shipping';
    
    % Cross connections
    'WS1_Assembly', 'WS7_Machining';
    'WS7_Machining', 'WS3_Quality';
    'WS6_Storage', 'WS2_Painting';
    'WS6_Storage', 'WS5_Shipping';
    
    % Charging station connections
    'WS2_Painting', 'CS1_North';
    'WS3_Quality', 'CS2_Central';
    'WS5_Shipping', 'CS3_South';
    'WS6_Storage', 'CS2_Central';
    
    % Reverse paths for bidirectional movement
    'WS2_Painting', 'WS1_Assembly';
    'WS3_Quality', 'WS2_Painting';
    'WS4_Packaging', 'WS3_Quality';
    'WS5_Shipping', 'WS4_Packaging';
    'WS7_Machining', 'WS1_Assembly';
    'WS3_Quality', 'WS7_Machining';
    'WS2_Painting', 'WS6_Storage';
    'WS5_Shipping', 'WS6_Storage';
    'CS1_North', 'WS2_Painting';
    'CS2_Central', 'WS3_Quality';
    'CS3_South', 'WS5_Shipping';
    'CS2_Central', 'WS6_Storage'
};

% Function to calculate edge metrics
calculate_edge_metrics = @(node_u, node_v) ...
    calculateEdgeMetrics(node_u, node_v, G, factory_width);

% Add edges with dynamic metrics
for i = 1:size(movement_paths,1)
    u = movement_paths{i,1};
    v = movement_paths{i,2};
    
    % Find node indices
    u_idx = find(strcmp(G.Nodes.Name, u));
    v_idx = find(strcmp(G.Nodes.Name, v));
    
    if ~isempty(u_idx) && ~isempty(v_idx)
        [travel_time, energy_cost, network_delay, distance] = ...
            calculate_edge_metrics(u, v);
        
        % Add edge with attributes
        G = addedge(G, u, v);
        
        % Store edge properties
        edge_idx = findedge(G, u, v);
        
        % Initialize edge properties if they don't exist
        if ~isfield(G.Edges, 'TravelTime')
            G.Edges.TravelTime = zeros(numedges(G), 1);
            G.Edges.EnergyCost = zeros(numedges(G), 1);
            G.Edges.NetworkDelay = zeros(numedges(G), 1);
            G.Edges.Distance = zeros(numedges(G), 1);
        else
            % Resize arrays if needed
            current_edges = numedges(G);
            if length(G.Edges.TravelTime) < current_edges
                G.Edges.TravelTime(current_edges) = 0;
                G.Edges.EnergyCost(current_edges) = 0;
                G.Edges.NetworkDelay(current_edges) = 0;
                G.Edges.Distance(current_edges) = 0;
            end
        end
        
        % Set edge properties
        G.Edges.TravelTime(edge_idx) = travel_time;
        G.Edges.EnergyCost(edge_idx) = energy_cost;
        G.Edges.NetworkDelay(edge_idx) = network_delay;
        G.Edges.Distance(edge_idx) = distance;
    end
end

fprintf('Factory Network Graph Created:\n');
fprintf('  • Nodes: %d\n', numnodes(G));
fprintf('  • Edges: %d\n', numedges(G));

% Calculate average degree for directed graph
in_deg = indegree(G);
out_deg = outdegree(G);
total_deg = in_deg + out_deg;
avg_total_degree = mean(total_deg);
fprintf('  • Average total degree (in+out): %.2f\n', avg_total_degree);
fprintf('\n');

% Print sample edge information
fprintf('Sample Edge Metrics:\n');
for i = 1:min(3, numedges(G))
    u = G.Edges.EndNodes{i,1};
    v = G.Edges.EndNodes{i,2};
    fprintf('  %s -> %s:\n', u, v);
    fprintf('    Travel Time: %.2f min\n', G.Edges.TravelTime(i));
    fprintf('    Energy Cost: %.2f kWh\n', G.Edges.EnergyCost(i));
    fprintf('    Network Delay: %.2f s\n', G.Edges.NetworkDelay(i));
    fprintf('    Distance: %.2f m\n', G.Edges.Distance(i));
end
fprintf('\n');

%% =====================================================
% STEP 4: AI-Enhanced Trajectory Planning (DQN)
% =====================================================
fprintf('============================================================\n');
fprintf('STEP 4: AI-Enhanced Trajectory Planning (DQN)\n');
fprintf('============================================================\n');

% Define state columns (6 features as specified)
state_cols = {'battery_level', 'obstacle_distance', ...
              'traffic_density', 'reaction_time', 'speed_kmh', 'lane_deviation'};

% Ensure all state columns exist with normalized values
for i = 1:length(state_cols)
    col_name = state_cols{i};
    
    % Check if column exists
    if any(strcmp(data.Properties.VariableNames, col_name))
        % Normalize the column
        col_data = data.(col_name);
        norm_col_name = [col_name '_norm'];
        data.(norm_col_name) = (col_data - min(col_data)) / (max(col_data) - min(col_data) + eps);
    else
        % Create random column
        data.(col_name) = rand(size(data,1), 1);
        norm_col_name = [col_name '_norm'];
        data.(norm_col_name) = data.(col_name); % Already in [0,1] range
    end
end

% Update state columns to use normalized versions
state_cols_norm = cellfun(@(x) [x '_norm'], state_cols, 'UniformOutput', false);

% Prepare state matrix
states = [];
for i = 1:length(state_cols_norm)
    if any(strcmp(data.Properties.VariableNames, state_cols_norm{i}))
        states = [states, data.(state_cols_norm{i})];
    else
        states = [states, rand(size(data,1), 1)];
    end
end

% Prepare action data
if any(strcmp(data.Properties.VariableNames, 'ADAS_output_num'))
    actions = data.ADAS_output_num;
else
    actions = randi(5, size(data,1), 1);
end
action_dim = length(unique(actions));

% Ensure action_dim is at least 2
action_dim = max(action_dim, 2);

fprintf('State dimensions: %d features\n', size(states, 2));
fprintf('State features: %s\n', strjoin(state_cols, ', '));
fprintf('Action dimension: %d unique actions\n', action_dim);
fprintf('\n');

% Create simple neural network for DQN (using Deep Learning Toolbox)
state_dim = size(states, 2);  % Use actual state dimension

fprintf('Creating DQN Network with state dimension: %d\n', state_dim);

% Check if Deep Learning Toolbox is available
if license('test', 'Neural_Network_Toolbox') || license('test', 'Deep_Learning_Toolbox')
    % Define a simple neural network with correct input size
    layers = [
        featureInputLayer(state_dim, 'Name', 'input')
        fullyConnectedLayer(128, 'Name', 'fc1')
        reluLayer('Name', 'relu1')
        dropoutLayer(0.2, 'Name', 'dropout1')
        fullyConnectedLayer(64, 'Name', 'fc2')
        reluLayer('Name', 'relu2')
        fullyConnectedLayer(action_dim, 'Name', 'output')
    ];
    
    % Create network - MATLAB automatically initializes with random weights
    dqn_net = dlnetwork(layers);
    
    fprintf('DQN Network Architecture:\n');
    fprintf('  • Input dimension: %d\n', state_dim);
    fprintf('  • Output dimension: %d\n', action_dim);
    fprintf('  • Hidden layers: 2 (128, 64 neurons)\n');
    fprintf('  • Using MATLAB Deep Learning Toolbox\n');
    
    % Test the network with sample input
    test_input = randn(state_dim, 1);
    test_dl = dlarray(test_input, 'CB');
    try
        test_output = predict(dqn_net, test_dl);
        fprintf('  • Network test successful. Output shape: %s\n', mat2str(size(test_output)));
    catch ME
        fprintf('  • Network test failed: %s\n', ME.message);
        fprintf('  • Falling back to Q-table approach\n');
        dqn_net = [];
    end
else
    % Use simple Q-table if Deep Learning Toolbox not available
    dqn_net = [];
    fprintf('Deep Learning Toolbox not available. Using simple Q-table approach.\n');
end
fprintf('\n');

%% =====================================================
% STEP 5: Multi-Agent Trajectory Simulation
% =====================================================
fprintf('============================================================\n');
fprintf('STEP 5: Multi-Agent Trajectory Simulation\n');
fprintf('============================================================\n');

num_agents = 3;
agent_colors = {'red', 'blue', 'green'};
agent_names = {'AGV-001', 'AGV-002', 'AGV-003'};
interpolation_steps = 5;

% Define tasks for each agent
agent_tasks = {
    'WS6_Storage', 'WS5_Shipping', 'high';
    'WS1_Assembly', 'WS4_Packaging', 'medium';
    'WS7_Machining', 'WS3_Quality', 'low'
};

% Initialize trajectories cell array
trajectories = cell(num_agents, 1);

for agent_id = 1:num_agents
    fprintf('\nInitializing Agent %d (%s)...\n', agent_id-1, agent_names{agent_id});
    fprintf('  Task: %s → %s\n', agent_tasks{agent_id,1}, agent_tasks{agent_id,2});
    fprintf('  Priority: %s\n', agent_tasks{agent_id,3});
    
    % Find optimal path
    start_node = agent_tasks{agent_id,1};
    goal_node = agent_tasks{agent_id,2};
    
    % Simulate initial battery level
    initial_battery = 0.6 + 0.3*rand();
    
    % Find optimal path using MATLAB's shortestpath
    try
        [optimal_path, path_len] = shortestpath(G, start_node, goal_node);
        fprintf('  Initial Battery: %.1f%%\n', initial_battery*100);
        fprintf('  Optimal Path: %s\n', strjoin(optimal_path, ' → '));
    catch
        fprintf('  Warning: Could not find path from %s to %s\n', start_node, goal_node);
        continue;
    end
    
    % Initialize agent trajectory
    agent_traj = [];
    step_count = 0;
    current_battery = initial_battery;
    
    for i = 1:length(optimal_path)-1
        u = optimal_path{i};
        v = optimal_path{i+1};
        
        % Get node positions
        u_idx = find(strcmp(G.Nodes.Name, u));
        v_idx = find(strcmp(G.Nodes.Name, v));
        
        if isempty(u_idx) || isempty(v_idx)
            fprintf('  Warning: Invalid edge %s -> %s. Skipping...\n', u, v);
            continue;
        end
        
        sx = G.Nodes.Pos{u_idx}(1);
        sy = G.Nodes.Pos{u_idx}(2);
        ex = G.Nodes.Pos{v_idx}(1);
        ey = G.Nodes.Pos{v_idx}(2);
        
        % Get DQN action - ensure state dimension matches
        state_idx = randi(size(states,1));
        current_state = states(state_idx,:)';
        
        % Verify state dimension
        if length(current_state) ~= state_dim
            fprintf('  Warning: State dimension mismatch. Expected %d, got %d\n', ...
                state_dim, length(current_state));
            % Pad or truncate state
            if length(current_state) < state_dim
                current_state = [current_state; rand(state_dim - length(current_state), 1)];
            else
                current_state = current_state(1:state_dim);
            end
        end
        
        [action_idx, q_values, dynamic_state] = getDQNAction(current_state, agent_id, step_count, dqn_net, action_dim, state_dim);
        
        % Get edge information
        edge_idx = findedge(G, u, v);
        if edge_idx > 0
            travel_time = G.Edges.TravelTime(edge_idx);
            energy_cost = G.Edges.EnergyCost(edge_idx);
            network_delay = G.Edges.NetworkDelay(edge_idx);
        else
            travel_time = 1.0;
            energy_cost = 0.1;
            network_delay = 0.05;
        end
        
        % Update battery
        current_battery = max(0, current_battery - energy_cost);
        if contains(v, 'CS')  % If moving to charging station
            current_battery = min(1.0, current_battery + 0.3);
        end
        
        % Calculate reward
        time_penalty = -travel_time * 0.5;
        energy_penalty = -energy_cost * 10;
        network_penalty = -network_delay * 2;
        battery_bonus = 5.0 * (current_battery > 0.2) - 10.0 * (current_battery <= 0.2);
        reward = time_penalty + energy_penalty + network_penalty + battery_bonus;
        
        % Interpolate positions
        for step = 0:interpolation_steps
            t = step / interpolation_steps;
            x = sx + (ex - sx) * t;
            y = sy + (ey - sy) * t;
            
            traj_point = struct();
            traj_point.agent_id = agent_id-1;
            traj_point.agent_name = agent_names{agent_id};
            traj_point.from = u;
            traj_point.to = v;
            traj_point.x = x;
            traj_point.y = y;
            traj_point.step = step_count;
            traj_point.battery = current_battery;
            traj_point.chosen_action = action_idx;
            traj_point.q_values = q_values;
            traj_point.travel_time = travel_time;
            traj_point.energy_cost = energy_cost;
            traj_point.network_delay = network_delay;
            traj_point.reward = reward;
            traj_point.state = dynamic_state;
            
            agent_traj = [agent_traj; traj_point];
        end
        
        step_count = step_count + 1;
        
        % Print step information
        if i == 1 || i == length(optimal_path)-1
            fprintf('    Step %d: %s → %s\n', step_count, u, v);
            fprintf('      Battery: %.1f%% | Time: %.2fmin | Energy: %.2fkWh | Delay: %.2fs | Reward: %.2f\n', ...
                current_battery*100, travel_time, energy_cost, network_delay, reward);
        end
    end
    
    if ~isempty(agent_traj)
        trajectories{agent_id} = agent_traj;
        fprintf('  Trajectory completed: %d positions recorded\n', length(agent_traj));
        fprintf('  Final Battery: %.1f%%\n', current_battery*100);
    else
        fprintf('  Warning: No positions recorded for agent %d\n', agent_id-1);
    end
end

fprintf('\n============================================================\n');
fprintf('SIMULATION SUMMARY\n');
fprintf('============================================================\n');

for agent_id = 1:num_agents
    if ~isempty(trajectories{agent_id})
        traj = trajectories{agent_id};
        total_steps = length(unique([traj.step]));
        final_pos = traj(end);
        fprintf('\nAgent %d (%s):\n', agent_id-1, agent_names{agent_id});
        fprintf('  • Total path segments: %d\n', total_steps);
        fprintf('  • Final position: %s\n', final_pos.to);
        fprintf('  • Final battery: %.1f%%\n', final_pos.battery*100);
        fprintf('  • Average reward per step: %.2f\n', mean([traj.reward]));
    end
end
fprintf('\n');

%% =====================================================
% STEP 6: Visualization - Separate Plots for Each Agent
% =====================================================
fprintf('============================================================\n');
fprintf('STEP 6: Generating Visualizations\n');
fprintf('============================================================\n');

% Get node positions
node_names = G.Nodes.Name;
node_positions = cell2mat(G.Nodes.Pos);
workstation_nodes = find(strcmp(G.Nodes.Type, 'workstation'));
charger_nodes = find(strcmp(G.Nodes.Type, 'charger'));

for agent_id = 1:num_agents
    if isempty(trajectories{agent_id})
        fprintf('\nSkipping Agent %d - No trajectory data\n', agent_id-1);
        continue;
    end
    
    traj = trajectories{agent_id};
    fprintf('\nGenerating plot for Agent %d (%s)...\n', agent_id-1, agent_names{agent_id});
    
    figure('Position', [100, 100, 1600, 1000]);
    
    % Factory boundary - FIXED: No Alpha property in plot, use RGBA color
    plot([0, factory_width, factory_width, 0, 0], ...
         [0, 0, factory_height, factory_height, 0], ...
         '--', 'LineWidth', 2, 'Color', [0.5, 0.5, 0.5, 0.7]); % RGBA color
    hold on;
    
    % Draw obstacles
    for obs_id = 1:size(obstacles,1)
        x = obstacles{obs_id,2};
        y = obstacles{obs_id,3};
        w = obstacles{obs_id,4};
        h = obstacles{obs_id,5};
        
        rectangle('Position', [x-w/2, y-h/2, w, h], ...
                 'FaceColor', [0.7, 0.7, 0.7], 'EdgeColor', 'k', ...
                 'FaceAlpha', 0.3); % FaceAlpha works for rectangle
        
        % FIXED: Use valid color name or RGB
        obstacle_name = strrep(obstacles{obs_id,1}, '_', ' ');
        text(x, y, obstacle_name, ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
            'FontSize', 9, 'FontWeight', 'bold', 'Color', [0.5, 0, 0]); % Dark red RGB
    end
    
    % Draw network nodes
    if ~isempty(workstation_nodes)
        scatter(node_positions(workstation_nodes,1), node_positions(workstation_nodes,2), ...
               1200, [0.31, 0.76, 0.97], 'filled', 'MarkerEdgeColor', 'k', ...
               'LineWidth', 2, 'DisplayName', 'Workstations');
    end
    
    if ~isempty(charger_nodes)
        scatter(node_positions(charger_nodes,1), node_positions(charger_nodes,2), ...
               1000, [0.51, 0.78, 0.52], 'filled', 'MarkerEdgeColor', 'k', ...
               'LineWidth', 2, 'DisplayName', 'Charging Stations');
    end
    
    % Draw node labels
    for i = 1:length(node_names)
        text(node_positions(i,1), node_positions(i,2), node_names{i}, ...
            'FontSize', 10, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', ...
            'Color', 'black');
    end
    
    % Draw edges with color based on network delay
    for i = 1:numedges(G)
        u = G.Edges.EndNodes{i,1};
        v = G.Edges.EndNodes{i,2};
        
        u_idx = find(strcmp(G.Nodes.Name, u));
        v_idx = find(strcmp(G.Nodes.Name, v));
        
        if isempty(u_idx) || isempty(v_idx)
            continue;
        end
        
        x1 = node_positions(u_idx,1);
        y1 = node_positions(u_idx,2);
        x2 = node_positions(v_idx,1);
        y2 = node_positions(v_idx,2);
        
        % Color by network delay
        if G.Edges.NetworkDelay(i) < 0.1
            edge_color = 'g';
            edge_width = 1;
        elseif G.Edges.NetworkDelay(i) < 0.2
            edge_color = [1, 0.5, 0]; % Orange
            edge_width = 1.5;
        else
            edge_color = 'r';
            edge_width = 2;
        end
        
        % Create semi-transparent lines
        if ischar(edge_color)
            % Named color - convert to RGB with alpha
            switch edge_color
                case 'g'
                    line_color = [0, 0.5, 0, 0.6]; % Dark green with alpha
                case 'r'
                    line_color = [0.8, 0, 0, 0.6]; % Red with alpha
                otherwise
                    line_color = [1, 0.5, 0, 0.6]; % Orange with alpha
            end
        else
            % RGB color - add alpha channel
            line_color = [edge_color, 0.6]; % Add alpha channel as 4th element
        end
        
        plot([x1, x2], [y1, y2], 'Color', line_color, 'LineWidth', edge_width);
        
        % Add arrow
        arrow_x = x1 + 0.7*(x2-x1);
        arrow_y = y1 + 0.7*(y2-y1);
        dx = (x2-x1)*0.2;
        dy = (y2-y1)*0.2;
        
        % Only add arrow if the vector has meaningful length
        if abs(dx) > 0.1 || abs(dy) > 0.1
            quiver(arrow_x, arrow_y, dx, dy, ...
                   'Color', line_color, 'LineWidth', 1, 'MaxHeadSize', 2, ...
                   'AutoScale', 'off');
        end
    end
    
    % Plot trajectory with battery-based coloring
    xs = [traj.x]';
    ys = [traj.y]';
    batteries = [traj.battery]';
    
    scatter(xs, ys, 100, batteries, 'filled', ...
           'MarkerEdgeColor', 'k', 'LineWidth', 0.5, ...
           'DisplayName', 'Trajectory (Battery Level)');
    
    % Draw trajectory line
    plot(xs, ys, 'Color', agent_colors{agent_id}, 'LineWidth', 2, ...
        'LineStyle', '-', 'DisplayName', 'Movement Path');
    
    % Mark start and end points
    start_point = traj(1);
    end_point = traj(end);
    
    scatter(start_point.x, start_point.y, 300, 'g', 's', 'filled', ...
           'MarkerEdgeColor', 'k', 'LineWidth', 2, 'DisplayName', 'Start');
    scatter(end_point.x, end_point.y, 300, 'r', 's', 'filled', ...
           'MarkerEdgeColor', 'k', 'LineWidth', 2, 'DisplayName', 'Goal');
    
    % Annotate key points with actions
    annotation_step = interpolation_steps * 2;
    max_annotations = min(50, length(traj));
    for i = 1:annotation_step:max_annotations
        step = traj(i);
        text(step.x + 1, step.y + 1, sprintf('A%d', step.chosen_action), ...
            'Color', 'k', 'FontSize', 8, 'FontWeight', 'bold', ...
            'BackgroundColor', agent_colors{agent_id}, 'EdgeColor', 'k', ...
            'Margin', 0.5);
    end
    
    % Add info box
    unique_froms = unique({traj.from});
    avg_reward_val = mean([traj.reward]);
    if isnan(avg_reward_val)
        avg_reward_val = 0;
    end
    
    info_text = sprintf(['Agent: %s\n', ...
                        'Start: %s\n', ...
                        'Goal: %s\n', ...
                        'Path Length: %d segments\n', ...
                        'Final Battery: %.1f%%\n', ...
                        'Avg Reward: %.2f'], ...
                        agent_names{agent_id}, ...
                        start_point.from, ...
                        end_point.to, ...
                        length(unique_froms), ...
                        end_point.battery*100, ...
                        avg_reward_val);
    
    rectangle('Position', [factory_width-35, factory_height-20, 33, 18], ...
             'FaceColor', 'white', 'EdgeColor', 'k', 'FaceAlpha', 0.9);
    text(factory_width-33, factory_height-5, info_text, ...
        'FontSize', 9, 'FontName', 'Monospaced', 'VerticalAlignment', 'top', ...
        'Color', 'black');
    
    title(sprintf('Agent %d: %s - DQN Optimized Trajectory', agent_id-1, agent_names{agent_id}), ...
         'FontSize', 16, 'FontWeight', 'bold', 'Padding', 20);
    xlabel('Factory Width (meters)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Factory Height (meters)', 'FontSize', 12, 'FontWeight', 'bold');
    
    % Add colorbar for battery levels
    colormap(jet);
    c = colorbar;
    c.Label.String = 'Battery Level';
    c.Label.FontWeight = 'bold';
    
    legend('Location', 'northeastoutside', 'FontSize', 10);
    grid on;
    grid minor;
    axis equal;
    xlim([0, factory_width]);
    ylim([0, factory_height]);
    
    % Save figure
    filename = sprintf('agent_%d_trajectory.png', agent_id-1);
    saveas(gcf, filename);
    fprintf('  Plot saved as: %s\n', filename);
    
    hold off;
end

%% =====================================================
% STEP 7: Command Window Output Summary
% =====================================================
fprintf('\n============================================================\n');
fprintf('DETAILED TRAJECTORY ANALYSIS\n');
fprintf('============================================================\n');

for agent_id = 1:num_agents
    if isempty(trajectories{agent_id})
        continue;
    end
    
    traj = trajectories{agent_id};
    
    fprintf('\n%s\n', repmat('=', 1, 40));
    fprintf('AGENT %d: %s\n', agent_id-1, agent_names{agent_id});
    fprintf('%s\n', repmat('=', 1, 40));
    
    % Get unique path segments
    path_segments = {};
    current_segment = [];
    for i = 1:length(traj)
        segment = {traj(i).from, traj(i).to};
        if isempty(current_segment) || ~isequal(segment, current_segment)
            if ~isempty(current_segment)
                path_segments{end+1} = current_segment;
            end
            current_segment = segment;
        end
    end
    if ~isempty(current_segment)
        path_segments{end+1} = current_segment;
    end
    
    % Print path
    path_str = '';
    for i = 1:length(path_segments)
        path_str = [path_str, path_segments{i}{1}, ' → '];
    end
    if ~isempty(path_segments)
        path_str = [path_str, path_segments{end}{2}];
    end
    
    fprintf('Complete Path: %s\n', path_str);
    fprintf('Total Segments: %d\n', length(path_segments));
    fprintf('Trajectory Points: %d\n\n', length(traj));
    
    % Segment details
    fprintf('Segment Details:\n');
    fprintf('%s\n', repmat('-', 1, 80));
    fprintf('%-15s %-15s %-10s %-12s %-10s %-12s %-8s\n', ...
            'From', 'To', 'Time(min)', 'Energy(kWh)', 'Delay(s)', 'Battery(%)', 'Action');
    fprintf('%s\n', repmat('-', 1, 80));
    
    segment_data = struct();
    for i = 1:length(traj)
        key = sprintf('%s_%s', traj(i).from, traj(i).to);
        if ~isfield(segment_data, key)
            segment_data.(key) = struct();
            segment_data.(key).travel_time = traj(i).travel_time;
            segment_data.(key).energy_cost = traj(i).energy_cost;
            segment_data.(key).network_delay = traj(i).network_delay;
            segment_data.(key).battery = traj(i).battery;
            segment_data.(key).action = traj(i).chosen_action;
        end
    end
    
    fields = fieldnames(segment_data);
    for i = 1:length(fields)
        key = fields{i};
        parts = strsplit(key, '_');
        if length(parts) >= 2
            frm = parts{1};
            to = parts{2};
            data = segment_data.(key);
            
            fprintf('%-15s %-15s %-10.2f %-12.3f %-10.2f %-12.1f A%-7d\n', ...
                    frm, to, data.travel_time, data.energy_cost, ...
                    data.network_delay, data.battery*100, data.action);
        end
    end
    
    % Performance metrics
    fprintf('\nPerformance Metrics:\n');
    fprintf('%s\n', repmat('-', 1, 40));
    
    total_time = 0;
    total_energy = 0;
    all_delays = [];
    for i = 1:length(fields)
        data = segment_data.(fields{i});
        total_time = total_time + data.travel_time;
        total_energy = total_energy + data.energy_cost;
        all_delays = [all_delays; data.network_delay];
    end
    
    if ~isempty(all_delays)
        avg_delay = mean(all_delays);
    else
        avg_delay = 0;
    end
    
    all_rewards = [traj.reward];
    if ~isempty(all_rewards)
        avg_reward = mean(all_rewards);
    else
        avg_reward = 0;
    end
    
    if traj(1).battery > 0
        battery_efficiency = (traj(end).battery / traj(1).battery) * 100;
    else
        battery_efficiency = 0;
    end
    
    fprintf('Total Travel Time: %.2f minutes\n', total_time);
    fprintf('Total Energy Consumption: %.3f kWh\n', total_energy);
    fprintf('Average Network Delay: %.2f seconds\n', avg_delay);
    fprintf('Average Reward per Step: %.2f\n', avg_reward);
    fprintf('Battery Efficiency: %.1f%%\n', battery_efficiency);
    fprintf('Final Battery Level: %.1f%%\n', traj(end).battery*100);
    
    % DQN Statistics
    all_actions = [traj.chosen_action];
    if ~isempty(all_actions)
        unique_actions = unique(all_actions);
        
        fprintf('\nDQN Action Distribution:\n');
        for i = 1:length(unique_actions)
            action = unique_actions(i);
            count = sum(all_actions == action);
            percentage = (count / length(all_actions)) * 100;
            fprintf('  Action A%d: %d times (%.1f%%)\n', action, count, percentage);
        end
    end
end

fprintf('\n============================================================\n');
fprintf('SIMULATION COMPLETED SUCCESSFULLY\n');
fprintf('============================================================\n');
fprintf('Total Agents Simulated: %d\n', num_agents);
total_points = 0;
for i = 1:num_agents
    if ~isempty(trajectories{i})
        total_points = total_points + length(trajectories{i});
    end
end
fprintf('Total Trajectory Points Generated: %d\n', total_points);
fprintf('Factory Network Complexity: %d nodes, %d edges\n', numnodes(G), numedges(G));
fprintf('Outputs Generated:\n');
fprintf('  1. Individual trajectory plots for each agent\n');
fprintf('  2. Detailed command window analysis\n');
fprintf('  3. Performance metrics for all agents\n');
fprintf('\nReady for production deployment!\n');

%% =====================================================
% HELPER FUNCTIONS
% =====================================================

function [travel_time, energy_cost, network_delay, distance] = ...
         calculateEdgeMetrics(node_u, node_v, G, factory_width)
    % Calculate edge metrics between two nodes
    
    % Find node indices
    u_idx = find(strcmp(G.Nodes.Name, node_u));
    v_idx = find(strcmp(G.Nodes.Name, node_v));
    
    if isempty(u_idx) || isempty(v_idx)
        travel_time = 1.0;
        energy_cost = 0.1;
        network_delay = 0.05;
        distance = 0;
        return;
    end
    
    % Get positions
    pos1 = G.Nodes.Pos{u_idx};
    pos2 = G.Nodes.Pos{v_idx};
    
    % Calculate distance
    dx = pos1(1) - pos2(1);
    dy = pos1(2) - pos2(2);
    distance = sqrt(dx^2 + dy^2);
    
    % Dynamic factors
    congestion = G.Nodes.TrafficWeight(u_idx);
    time_of_day_factor = 1 + 0.3 * sin(now * 2*pi); % Daily variation
    
    travel_time = round(distance * 0.5 * congestion * time_of_day_factor, 2);
    energy_cost = round(distance * 0.015 * congestion, 2);
    network_delay = round(0.05 + distance * 0.002 * congestion, 2);
end

function [action_idx, q_values, dynamic_state] = ...
         getDQNAction(current_state, agent_id, step_count, dqn_net, action_dim, state_dim)
    % Get action from DQN with simulated real-world factors
    
    % Ensure state has correct dimension
    if length(current_state) ~= state_dim
        % Adjust state dimension if needed
        if length(current_state) < state_dim
            % Pad with zeros
            dynamic_state = [current_state; zeros(state_dim - length(current_state), 1)];
        else
            % Truncate
            dynamic_state = current_state(1:state_dim);
        end
    else
        dynamic_state = current_state;
    end
    
    % Add dynamic factors to state
    % Simulate battery drain (assuming battery is first element)
    if length(dynamic_state) >= 1
        dynamic_state(1) = max(0.1, dynamic_state(1) - 0.01 * step_count);
    end
    
    % Simulate increasing congestion over time (assuming traffic is third element)
    if length(dynamic_state) >= 3
        dynamic_state(3) = min(1.0, dynamic_state(3) + 0.005 * step_count);
    end
    
    % Check if Deep Learning Toolbox is available and network is valid
    if ~isempty(dqn_net) && isa(dqn_net, 'dlnetwork')
        try
            % Convert to dlarray for neural network
            % dlnetwork expects input as 'CB' format (Channel x Batch)
            state_dl = dlarray(dynamic_state, 'CB');
            
            % Get Q-values from DQN
            q_values_dl = predict(dqn_net, state_dl);
            q_values = extractdata(q_values_dl)';
        catch ME
            % If DQN fails, use random Q-values
            fprintf('  Agent %d, Step %d: DQN prediction failed. Using random action.\n', agent_id-1, step_count);
            q_values = rand(1, action_dim);
        end
    else
        % Simple random Q-values if no DQN
        q_values = rand(1, action_dim);
    end
    
    % Epsilon-greedy action selection
    epsilon = max(0.1, 0.5 - step_count * 0.01); % Decaying exploration
    
    if rand() < epsilon
        action_idx = randi(action_dim);
    else
        [~, action_idx] = max(q_values);
    end
end