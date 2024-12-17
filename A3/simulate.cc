#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include "platform_load_time_gen.hpp"
#include <iostream>
#include <algorithm>
#include <mpi.h>

#include <thread>
#include <chrono>

using std::string;
using std::unordered_map;
using std::unordered_set;
using std::vector;
using adjacency_matrix = std::vector<std::vector<size_t>>;


bool debug = false;

// Define LineColor Enum with lower-case letters
enum LineColor 
{
    GREEN = 'g',
    YELLOW = 'y',
    BLUE = 'b',
    // Other colors can be added if needed
};


// Comparator
struct CompareTrain 
{
    bool operator()(const std::pair<int, int>& a, const std::pair<int, int>& b) const 
    {
        if (a.first != b.first) {
            return a.first > b.first;  // Correct: Earlier arrival time has higher priority
        }
        return a.second > b.second;   // Correct: Lower train ID has higher priority
    }
};



struct Platform 
{
    std::string station_name = "";               // Name of the station
    std::string dest_station_name = "";          // Destination station name
    int platform_train_id = -1;                  // Train currently at the platform (-1 if none)
    int transit_train_id = -1;                   // Train in transit (-1 if none)
    int transit_train_time_left = -1;           // Time left for transit (-1 if none)
    size_t popularity = static_cast<size_t>(-1); // Popularity metric for the platform
    PlatformLoadTimeGen pltg{0};                 // Generator for platform load time
    std::priority_queue<std::pair<int, int>, 
                        std::vector<std::pair<int, int>>, 
                        CompareTrain> holding_queue;  // Holding queue for trains
    int loading_or_unloading_time_left = -1;     // Time left for loading/unloading
    
    size_t distance_to_dest = static_cast<size_t>(-1); // Distance to destination
    bool is_start = false;                       // Flag indicating if it's a start platform
    bool is_end = false;                         // Flag indicating if it's an end platform
    vector<char> line_colors = vector<char>();   // Lines this platform belongs to
    bool transit_train_time_passed_this_turn = false;
    bool loading_or_unloading_time_passed_this_turn = false;

    

    // Default constructor
    Platform() 
        : holding_queue(CompareTrain{}) {}       // Initialize holding_queue with comparator

    // Constructor with popularity
    Platform(size_t popularity) 
        : popularity(popularity), pltg(popularity), holding_queue(CompareTrain{}) {}
};


struct TrainStation
{
    std::string station_name;
    size_t station_idx; // with respect all stations
    size_t popularity;
    std::vector<Platform> platforms;
};


struct Train {
    int train_id;
    char line;                // Line the train is on ('B', 'G', 'Y')
    int current_station_idx;  // Index in the line's station sequence
    int dest_station_idx;
    int direction;            // 1 for forward, -1 for backward
};

// Define line mappings using lower-case letters
unordered_map<char, int> line_char_to_int = 
{
    {'g', 1},
    {'b', 2},
    {'y', 3}
};

unordered_map<int, char> line_int_to_char = 
{
    {1, 'g'},
    {2, 'b'},
    {3, 'y'}
};


std::vector<std::string> serialize_station_name_to_rank(const std::unordered_map<std::string, int>& map) 
{
    std::vector<std::string> serialized_data;
    for (const auto& pair : map) 
    {
        serialized_data.push_back(pair.first);  // Station name
        serialized_data.push_back(std::to_string(pair.second)); // Rank
    }
    return serialized_data;
}

std::unordered_map<std::string, int> deserialize_station_name_to_rank(const std::vector<std::string>& serialized_data) 
{
    std::unordered_map<std::string, int> map;
    for (size_t i = 0; i < serialized_data.size(); i += 2) 
    {
        std::string station_name = serialized_data[i];
        int rank = std::stoi(serialized_data[i + 1]);
        map[station_name] = rank;
    }
    return map;
}

void distribute_station_name_to_rank(std::unordered_map<std::string, int>& station_name_to_rank, int mpi_rank) //, int mpi_size) 
{
    // Root process serializes the map
    std::vector<std::string> serialized_data;
    if (mpi_rank == 0) 
    {
        serialized_data = serialize_station_name_to_rank(station_name_to_rank);
    }

    // Determine the size of the serialized data
    int serialized_size = serialized_data.size();
    MPI_Bcast(&serialized_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Prepare buffer for broadcasting
    std::vector<char> buffer;
    if (mpi_rank == 0) 
    {
        std::ostringstream oss;
        for (const auto& item : serialized_data) {
            oss << item << '\n'; // Use newline as a separator
        }
        std::string serialized_string = oss.str();
        buffer.assign(serialized_string.begin(), serialized_string.end());
    }
    // dynamically char buffer to serialize.
}





void create_train_stations_and_platforms(unordered_map<string, TrainStation> &stations, const vector<string> &station_names, const adjacency_matrix &mat, 
                                         const size_t num_stations, const unordered_map<char, vector<string>> &station_lines, const std::vector<size_t> &popularities)
{
    for(size_t i = 0; i < num_stations; i++)
    {
        TrainStation curr;
        curr.station_name = station_names[i];
        curr.station_idx = i;
        curr.popularity = popularities[i];
        stations[station_names[i]] = curr;
    }
    
    for(size_t i = 0; i < num_stations; i++)
    {
        string curr_station_name = station_names[i];
        vector<size_t> neighbor_stations = mat[i];

        for(size_t j = 0; j < neighbor_stations.size(); j++)
        {
            size_t dist = neighbor_stations[j];
            if (dist == 0) continue; // no link

            Platform curr_platform(popularities[i]);

            curr_platform.distance_to_dest = dist;
            curr_platform.station_name = curr_station_name; 
            curr_platform.dest_station_name = station_names[j]; 

            stations[curr_station_name].platforms.push_back(curr_platform);
        }
    }

    for(const auto& pair : station_lines)
    {
        char line = pair.first;
        vector<string> line_stations = pair.second;
        
        // Going forward 
        for (size_t i = 0; i < line_stations.size() - 1; i++)
        {
            string curr_station_name = line_stations[i];
            string next_station_name = line_stations[i+1];
            
            TrainStation& curr_station = stations[curr_station_name];
            vector<Platform> &platforms = curr_station.platforms;

            bool platform_found = false;  // Flag to check if platform is found

            for(size_t j = 0; j < platforms.size(); j++)
            {
                Platform& curr_platform = platforms[j];
                if (curr_platform.station_name == curr_station_name && curr_platform.dest_station_name == next_station_name)
                {
                    curr_platform.line_colors.push_back(line);

                    if (i == 0) curr_platform.is_start = true; // start of the line
                    platform_found = true;
                    break;
                }
            }
            if (!platform_found)
            {
                if (debug)
                {
                    std::cout << "Error in Code, no platform found from " << curr_station_name << " to " << next_station_name << " for line " << line << std::endl;
                }
                
            }
        }
    
        // Going backwards
        for (size_t i = line_stations.size() - 1; i > 0; i--)
        {
            string curr_station_name = line_stations[i];
            string next_station_name = line_stations[i-1];
            
            TrainStation &curr_station = stations[curr_station_name];
            vector<Platform> &platforms = curr_station.platforms;

            bool platform_found = false;  // Flag to check if platform is found

            for(size_t j = 0; j < platforms.size(); j++)
            {
                Platform& curr_platform = platforms[j];
                if (curr_platform.station_name == curr_station_name && curr_platform.dest_station_name == next_station_name)
                {
                    curr_platform.line_colors.push_back(line);

                    if (i == line_stations.size() - 1) curr_platform.is_end = true; // end of the line
                    platform_found = true;
                    break;
                }
            }
            if (!platform_found)
            {
                if (debug)
                {
                    std::cout << "Error in Code, no platform found from " << curr_station_name << " to " << next_station_name << " for line " << line << std::endl;
                }
                
            }
        }
    }
}

void distribute_train_stations_among_processes(unordered_map<string, int> &station_name_to_rank, vector<string> &local_station_names, unordered_map<string, TrainStation> &local_stations,
                                               unordered_map<string, TrainStation> &stations, const vector<string> &station_names, const size_t mpi_rank, size_t total_processes)
{
    vector<string> start_end_station_names;
    vector<string> other_station_names;

    for (const auto& station_pair : stations) 
    {
        const std::string& station_name = station_pair.first;
        const TrainStation& station = station_pair.second;

        bool is_start_or_end = false;

        // Check if any of the platforms are start or end
        for (const Platform& platform : station.platforms) 
        {
            if (platform.is_start || platform.is_end) 
            {
                is_start_or_end = true;
                break;
            }
        }
        if (is_start_or_end) 
        {
            start_end_station_names.push_back(station_name);
        } 
        else 
        {
            other_station_names.push_back(station_name);
        }
    }

    // Edge case where we only have 1 process
    if (total_processes == 1) 
    {
        // If only one process, assign all stations to process 0
        for (const auto& station_name : station_names) 
        {
            station_name_to_rank[station_name] = 0;
        }
    } 
    else 
    {
        // Assign process 0 to start and end stations
        for (const string& station_name : start_end_station_names) 
        {
            station_name_to_rank[station_name] = 0;
        }

        // Now, distribute other stations among other processes
        int num_other_processes = total_processes - 1;

        for (size_t i = 0; i < other_station_names.size(); i++) 
        {
            const string& station_name = other_station_names[i];
            int rank = 1 + (i % num_other_processes);
            station_name_to_rank[station_name] = rank;
        }
    }

    for (const auto& station_pair : stations) 
    {
        const string& station_name = station_pair.first;
        const TrainStation& station = station_pair.second;

        int assigned_rank = station_name_to_rank[station_name];

        if (assigned_rank == mpi_rank) 
        {
            local_stations[station_name] = station;
        }
    }

}


void spawn_trains(size_t mpi_rank, const unordered_map<char, vector<string>> &station_lines, unordered_map<char, size_t> &num_trains_mapping, 
                  unordered_map<int, Train> &trains, unordered_map<string, TrainStation> &local_stations, size_t &current_tick, size_t &next_train_id,
                  unordered_map<char, unordered_map<int, int>> &line_idx_to_global_idx, unordered_map<char, unordered_map<std::string, int>> station_name_to_idx_line) 
{
    if (mpi_rank != 0) return; // only process 0 should run this

    // **Spawn Trains for Start Platforms**
    
    vector<char> line_order = {'g', 'y', 'b'};
    for (char line : line_order) 
    {
        const vector<string>& line_station_names = station_lines.at(line);

        // Attempt to spawn train at the start platform
        if (num_trains_mapping[line] > 0)
        {
            std::string start_station_name = line_station_names.front();

            if (local_stations.find(start_station_name) != local_stations.end())
            {
                TrainStation& start_station = local_stations[start_station_name];

                for (auto& platform : start_station.platforms)
                {
                    if (platform.is_start && std::find(platform.line_colors.begin(), platform.line_colors.end(), line) != platform.line_colors.end())
                    {

                        // Spawn train at the end platform's holding queue
                        platform.holding_queue.push({static_cast<int>(current_tick), static_cast<int>(next_train_id)});
                        


                        // Create and store Train object
                        Train new_train;
                        new_train.train_id = static_cast<int>(next_train_id);
                        new_train.line = line;
                        new_train.current_station_idx = line_idx_to_global_idx[line][0]; // Starting index
                        new_train.dest_station_idx = line_idx_to_global_idx[line][1];
                        new_train.direction = 1;            // Forward direction
                        trains[new_train.train_id] = new_train;

                        num_trains_mapping[line]--;
                        next_train_id++;

                        if (debug)
                        {
                            std::cout << "Tick " << current_tick << ": Spawned train " << new_train.train_id
                                << " at start platform " << platform.station_name
                                << " on line " << line << std::endl;
                        }
                        
                        break;
                    }
                }
            }
        }

        // Attempt to spawn train at the end platform
        if (num_trains_mapping[line] > 0)
        {
            std::string end_station_name = line_station_names.back();

            if (local_stations.find(end_station_name) != local_stations.end())
            {
                TrainStation& end_station = local_stations[end_station_name];

                for (auto& platform : end_station.platforms)
                {
                    if (platform.is_end && std::find(platform.line_colors.begin(), platform.line_colors.end(), line) != platform.line_colors.end())
                    {
                        // Spawn train at the end platform's holding queue
                        platform.holding_queue.push({static_cast<int>(current_tick), static_cast<int>(next_train_id)});


                         

                        // Create and store Train object
                        Train new_train_end;
                        new_train_end.train_id = static_cast<int>(next_train_id);
                        new_train_end.line = line;
                        new_train_end.current_station_idx = line_idx_to_global_idx[line][static_cast<int>(line_station_names.size()) - 1]; // Last index
                        new_train_end.dest_station_idx = line_idx_to_global_idx[line][static_cast<int>(line_station_names.size()) - 2];
                        new_train_end.direction = -1;                                  // Backward direction
                        trains[new_train_end.train_id] = new_train_end;

                        num_trains_mapping[line]--;
                        next_train_id++;

                        if (debug)
                        {
                            std::cout << "Tick " << current_tick << ": Spawned train " << new_train_end.train_id
                                << " at end platform " << platform.station_name
                                << " on line " << line << std::endl;
                        }
                        
                        break;
                    }
                }
            }
        }
    }
    
}


void create_mpi_comm_for_lines(unordered_map<char, MPI_Comm> &line_comms, const unordered_map<char, vector<string>> &station_lines, 
                               unordered_map<string, int> &station_name_to_rank, size_t mpi_rank, MPI_Group &world_group, unordered_map<char, unordered_map<int,int>> &lines_world_rank_to_line_rank)
{
    unordered_map<char, unordered_set<int>> line_processes;

    for (const auto& pair : station_lines) 
    {
        char line = pair.first;
        const vector<string>& line_stations = pair.second;

        unordered_set<int> processes_in_line;

        for (const string& station_name : line_stations) 
        {
            int rank = station_name_to_rank[station_name];
            processes_in_line.insert(rank);
        }

        line_processes[line] = processes_in_line;
    }

    

    for (const auto& pair : line_processes) 
    {
        char line = pair.first;
        const unordered_set<int>& processes_in_line = pair.second;

        vector<int> ranks_in_line(processes_in_line.begin(), processes_in_line.end());

        // Sort ranks for consistency (optional)
        std::sort(ranks_in_line.begin(), ranks_in_line.end());

        // Create MPI group
        MPI_Group line_group;
        MPI_Group_incl(world_group, ranks_in_line.size(), ranks_in_line.data(), &line_group);

        // Create MPI communicator
        MPI_Comm line_comm;
        MPI_Comm_create(MPI_COMM_WORLD, line_group, &line_comm);

        // Store the communicator
        line_comms[line] = line_comm;


        // Debug: Check if the current process is part of this line's communicator
        if (line_comms[line] != MPI_COMM_NULL) 
        {
            int line_comm_rank, line_comm_size;
            MPI_Comm_rank(line_comms[line], &line_comm_rank);
            MPI_Comm_size(line_comms[line], &line_comm_size);

            if (debug)
            {
                std::cout << "Process " << mpi_rank << " is in communicator for line " << line
                    << " with rank " << line_comm_rank << " and size " << line_comm_size << std::endl;
            }
            

            lines_world_rank_to_line_rank[line][mpi_rank] = line_comm_rank;
        } 
        else 
        {
            // Optional: Print if the process is not part of the communicator
            if (debug)
            {
                std::cout << "Process " << mpi_rank << " is NOT in communicator for line " << line << std::endl;
            }
            
        }

        // Free the group
        MPI_Group_free(&line_group);
    }
}



void process_local_updates(unordered_map<string, int> &station_name_to_rank, unordered_map<string, TrainStation> &local_stations,
                            const size_t mpi_rank, const unordered_map<char, vector<int>> line_station_indices, unordered_map<int, string> station_idx_to_name,                   
                            std::vector<std::tuple<int,int,int,int,int,int>> &trains_to_send_to_other_processes, unordered_map<int, Train> &trains, 
                            size_t current_tick, unordered_map<char, unordered_map<int, int>> &line_idx_to_global_idx, unordered_map<char, unordered_map<std::string, int>> station_name_to_idx_line) 
{

    


    for(auto &local_station_pair : local_stations)
    {
        TrainStation &station = local_station_pair.second;

        for(auto &platform : station.platforms)
        {
            // Decrement timers -> might be wrong, I try first
            if (platform.transit_train_id != -1) 
            {
                platform.transit_train_time_left--;
                platform.transit_train_time_passed_this_turn = true;

                if (debug) 
                {
                    std::cout << "current tick: " << current_tick << " platform: " << platform.station_name << " decreasing transit train time to " << platform.transit_train_time_left << " for train " << platform.transit_train_id << " " << std::endl; 
                }
                
            }
            if (platform.platform_train_id != -1) 
            {
                platform.loading_or_unloading_time_left--;
                platform.loading_or_unloading_time_passed_this_turn = true;

                if (debug)
                {
                    std::cout << "current tick: " << current_tick << " platform: " << platform.station_name << " decreasing platform loading and unloadin time to " << platform.loading_or_unloading_time_left << " for train " << platform.platform_train_id << " " << std::endl; 
                }
                
            }


            // 1. Transit Train Reached Destination
            if (platform.transit_train_id != -1 && platform.transit_train_time_left == 0) 
            {


                int train_id = platform.transit_train_id;
                Train& train = trains[train_id];

                const vector<int>& station_indices = line_station_indices.at(train.line);
                        
                int dest_station_global_idx = train.dest_station_idx;
                
                
                std::string dest_station_name = station_idx_to_name[dest_station_global_idx];

                int dest_rank_world = station_name_to_rank[dest_station_name];
                //int dest_rank_line = lines_world_rank_to_line_rank[train.line][dest_rank_world];


                /*
                    current station -> station that train departed from
                    dest station -> station that train just arrived
                    next dest station -> station that train is going
                */
                int dest_station_line_idx =  station_name_to_idx_line[train.line][dest_station_name];

                // I think this is the problem
                // This has to be in relative to the line, but right now it is global which is wrong
                int next_dest_station_line_idx = dest_station_line_idx + train.direction;

              
                // Reached the start/end of the line
                if (next_dest_station_line_idx < 0 || next_dest_station_line_idx >= station_indices.size())
                {
                    // Reverse direction
                    train.direction *= -1;
                    // Recalculate next_dest_station_line_idx based on new direction
                    next_dest_station_line_idx = dest_station_line_idx + train.direction;
                }
                // Update train's current and next destination indices
                train.current_station_idx = line_idx_to_global_idx[train.line][dest_station_line_idx];
                train.dest_station_idx = line_idx_to_global_idx[train.line][next_dest_station_line_idx];


                /*  UPDATE after all the stuff above
                    current station -> station that train just arrived
                    dest station -> station that train is going
                */

                // I think this is wrong bcs station_idx_to_name is global to all lines and not to local lines
                // Get the names of the current and next stations
                // std::string curr_station_name = station_idx_to_name[station_indices[train.current_station_idx]];
                // std::string next_dest_station_name = station_idx_to_name[station_indices[train.dest_station_idx]];

                // Get the names of the current and next stations
                std::string curr_station_name = station_idx_to_name[train.current_station_idx];
                std::string next_dest_station_name = station_idx_to_name[train.dest_station_idx];


                 // **Now, determine if we need to send the train to another process**
                if (dest_rank_world == mpi_rank) 
                {
                    // The train is arriving at a station managed by the same process.
                    // Add it directly to the holding queue of the appropriate platform.

                    TrainStation& curr_station = local_stations[curr_station_name];
                    bool platform_found = false;
                    
                    for (auto& platform : curr_station.platforms) 
                    {
                        if (platform.dest_station_name == next_dest_station_name &&
                            std::find(platform.line_colors.begin(), platform.line_colors.end(), train.line) != platform.line_colors.end()) 
                        {
                            
                            platform.holding_queue.push({static_cast<int>(current_tick), train.train_id});

                            if (debug)
                            {
                                std::cout << "Process " << mpi_rank << " train " << train.train_id
                                    << " arrived at station " << curr_station_name
                                    << " and added to holding queue for platform to "
                                    << next_dest_station_name << std::endl;
                            }
                            
                            
                            platform_found = true;
                            break;
                        }
                    }

                    if (!platform_found)
                    {

                        if (debug)
                        {
                            std::cout << "Process " << mpi_rank << " could not find platform from " << curr_station_name
                                << " to " << next_dest_station_name << " for train " << train.train_id << std::endl;
                        }
                        
                        
                        // TODO: remove when submitting
                        std::cerr << "Process " << mpi_rank << " could not find platform from " << curr_station_name
                                << " to " << next_dest_station_name << " for train " << train.train_id << std::endl;
                    }
                
                } 
                else 
                {
                    // The train needs to be sent to another process
                    int line_int = line_char_to_int[train.line];                                                           // this is next_dest_station
                    trains_to_send_to_other_processes.push_back(std::make_tuple(train.train_id, train.current_station_idx, train.dest_station_idx, train.direction, line_int, dest_rank_world));

                    if (debug)
                    {
                        std::cout << "Process " << mpi_rank << " will send train " << train.train_id
                            << " to station " << curr_station_name << " (process " << dest_rank_world << " within world comm) for " << "(process )" << dest_rank_world
                            << "to send train to new dest station of" << next_dest_station_name << std::endl;
                    }
                    
                }
                // Reset transit train
                platform.transit_train_id = -1;
                platform.transit_train_time_left = -1;
            }



            // 2. Platform's loading/unloading complete
            if (platform.platform_train_id != -1 && platform.loading_or_unloading_time_left <= 0) 
            {
                // If no transit train, move platform train to transit
                if (platform.transit_train_id == -1) 
                {
                    platform.transit_train_id = platform.platform_train_id;
                    platform.transit_train_time_left = static_cast<int>(platform.distance_to_dest);

                    if (debug)
                    {
                        std::cout << "Process " << mpi_rank << " moved train " << platform.platform_train_id
                            << " from platform at station " << platform.station_name
                            << " to transit to station " << platform.dest_station_name << std::endl;
                    }
                    

                    // Clear platform train
                    platform.platform_train_id = -1;
                    platform.loading_or_unloading_time_left = -1;
                } else 
                {
                    // Link is occupied; train cannot depart yet
                    // For simplicity, we do nothing in this case
                }
            }
            
            // 3. Move train from platform's holding queue to platform
            if (platform.platform_train_id == -1 && !platform.holding_queue.empty()) 
            {
                auto [arrival_time, train_id] = platform.holding_queue.top();
                platform.holding_queue.pop();

                platform.platform_train_id = train_id;
                platform.loading_or_unloading_time_left = platform.pltg.next(train_id);

                if (debug)
                {
                    std::cout << "Process " << mpi_rank << " moved train " << train_id
                        << " from holding queue to platform at station " << platform.station_name << std::endl;
                }
                
            }

            
            
        }
    }
}


void send_trains_to_other_processes(std::vector<std::tuple<int,int,int,int,int,int>> &trains_to_send_to_other_processes, unordered_map<int, Train> &trains, const size_t mpi_rank)
{
    // **Send Messages to Other Processes**
    for (const auto& train_info : trains_to_send_to_other_processes) 
    {
        // std::make_tuple(train.train_id, train.current_station_idx, train.dest_station_idx, train.direction, dest_rank_world)
        int train_id, current_station_idx, dest_station_idx, direction, line_int, dest_rank_world;
        std::tie(train_id, current_station_idx, dest_station_idx, direction, line_int, dest_rank_world) = train_info;

        int message[5];
        message[0] = train_id;
        message[1] = current_station_idx;
        message[2] = dest_station_idx;
        message[3] = direction;
        message[4] = line_int;

        

        // Send message
        MPI_Send(message, 5, MPI_INT, dest_rank_world, 0, MPI_COMM_WORLD);
        
        if (debug)
        {
            std::cout << "Process " << mpi_rank << " sent train " << train_id
                << " to process " << dest_rank_world << "within world comm" << std::endl;
        }
        

        trains.erase(train_id);
    }
    trains_to_send_to_other_processes.clear(); // Clear the list after sending
}




void print_output(unordered_map<string, TrainStation> &local_stations, unordered_map<int, Train> &trains, const size_t mpi_rank, const size_t total_processes, size_t current_tick)
{
    // Collect positions of trains
    std::vector<std::string> position_strings;

    // For each TrainStation in local_stations
    for (auto& station_pair : local_stations) 
    {
        const std::string& station_name = station_pair.first;
        TrainStation& station = station_pair.second;

        for (auto& platform : station.platforms) 
        {
            // platform_train_id
            if (platform.platform_train_id != -1) {
                int train_id = platform.platform_train_id;
                Train& train = trains[train_id]; // Get the train object
                char train_line = train.line;    // Now 'g', 'b', or 'y'
                std::stringstream ss;
                ss << train_line << train.train_id << "-" << station_name << "%";
                position_strings.push_back(ss.str());
            }

            // transit_train_id
            if (platform.transit_train_id != -1) 
            {
                int train_id = platform.transit_train_id;
                Train& train = trains[train_id]; // Get the train object
                char train_line = train.line;    // Now 'g', 'b', or 'y'
                std::stringstream ss;
                ss << train_line << train.train_id << "-" << platform.station_name << "->" << platform.dest_station_name;
                position_strings.push_back(ss.str());
            }

            // holding_queue
            if (!platform.holding_queue.empty()) 
            {
                // Copy the holding queue
                auto holding_queue_copy = platform.holding_queue;
                while (!holding_queue_copy.empty()) 
                {
                    auto [arrival_time, train_id] = holding_queue_copy.top();
                    holding_queue_copy.pop();
                    Train& train = trains[train_id]; // Get the train object
                    char train_line = train.line;    // Now 'g', 'b', or 'y'
                    std::stringstream ss;
                    ss << train_line << train.train_id << "-" << station_name << "#";
                    position_strings.push_back(ss.str());
                }
            }
        }
    }

    // Join position_strings into a single string, positions separated by spaces
    std::string positions_str = "";
    for (const auto& pos_str : position_strings) 
    {
        if (!positions_str.empty()) positions_str += " ";
        positions_str += pos_str;
    }

    int positions_len = positions_str.size();

    // Now send positions_len and positions_str to process 0
    if (mpi_rank != 0) 
    {
        MPI_Send(&positions_len, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        if (positions_len > 0) {
            MPI_Send(positions_str.c_str(), positions_len, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        }
    } 
    else 
    {
        // Process 0
        // Collect own positions
        std::vector<std::string> all_positions;
        if (!positions_str.empty()) 
        {
            // Split positions_str into individual positions
            std::istringstream iss(positions_str);
            std::string pos;
            while (iss >> pos) 
            {
                all_positions.push_back(pos);
            }
        }

        // Now receive positions from other processes
        for (int rank = 1; rank < total_processes; ++rank) 
        {
            int recv_positions_len = 0;
            MPI_Recv(&recv_positions_len, 1, MPI_INT, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (recv_positions_len > 0) 
            {
                char* recv_buffer = new char[recv_positions_len + 1];
                MPI_Recv(recv_buffer, recv_positions_len, MPI_CHAR, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                recv_buffer[recv_positions_len] = '\0';
                std::string recv_positions_str(recv_buffer);
                delete[] recv_buffer;

                // Split recv_positions_str into individual positions
                std::istringstream iss(recv_positions_str);
                std::string pos;
                while (iss >> pos) 
                {
                    all_positions.push_back(pos);
                }
            }
        }

        // Now sort all_positions lexicographically
        std::sort(all_positions.begin(), all_positions.end());

        // Output the line
        std::cout << current_tick << ":";
        for (const auto& pos_str : all_positions) 
        {
            std::cout << " " << pos_str;
        }
        std::cout << std::endl;
    }
}

// unordered_map<string, int> &station_name_to_rank, unordered_map<string, TrainStation> &local_stations,
// const size_t mpi_rank, const unordered_map<char, vector<int>> line_station_indices, unordered_map<int, string> station_idx_to_name,                   
// std::vector<std::tuple<int,int,int,int,int,int>> &trains_to_send_to_other_processes, unordered_map<int, Train> &trains, size_t current_tick

void receive_trains_from_other_processes(unordered_map<string, TrainStation> &local_stations, unordered_map<int, string> &station_idx_to_name, 
                                         const unordered_map<char, vector<int>> &line_station_indices, const size_t mpi_rank, 
                                         unordered_map<int, Train> &trains, size_t current_tick)
{
    // // **Process Incoming Trains**
    // for (const auto& line_comm_pair : line_comms) 
    // {
    //     char line = line_comm_pair.first; // Now 'g', 'b', or 'y'
        
        //MPI_Comm comm = line_comm_pair.second;
        // For now we try this


    MPI_Comm comm = MPI_COMM_WORLD;
    int flag = 0;
    MPI_Status status;

    do 
    {
        MPI_Iprobe(MPI_ANY_SOURCE, 0, comm, &flag, &status);
        if (flag) 
        {
            int message[5];
            MPI_Recv(message, 5, MPI_INT, MPI_ANY_SOURCE, 0, comm, &status);

            
            
            
            

            int incoming_train_id = message[0];
            int current_station_idx = message[1];
            int dest_station_idx = message[2];
            int direction = message[3];
            int line_int = message[4];
            char line_received = line_int_to_char[line_int];

            // Update or create the Train object
            Train train;
            train.train_id = incoming_train_id;
            train.line = line_received;
            train.current_station_idx = current_station_idx;
            train.dest_station_idx = dest_station_idx;
            train.direction = direction;

            


            

            
          
            // Find the station name 
            std::string station_name = station_idx_to_name[train.current_station_idx];
            std::string dest_station_name = station_idx_to_name[train.dest_station_idx];

            // Add the train to the appropriate platform
            TrainStation& station = local_stations[station_name];
            bool platform_found = false;

            // **Process Incoming Trains**
            for (auto& platform : station.platforms) 
            {
                if (platform.dest_station_name == dest_station_name &&
                    std::find(platform.line_colors.begin(), platform.line_colors.end(), line_received) != platform.line_colors.end()) 
                {
                    platform.holding_queue.push({static_cast<int>(current_tick), train.train_id});

                    if (debug)
                    {
                        std::cout << "Process " << mpi_rank << " train " << train.train_id
                            << " arrived at station " << station_name
                            << " and added to holding queue for platform to "
                            << dest_station_name << std::endl;
                    }
                    
                    
                    platform_found = true;
                    break;
                }
            }


            if (!platform_found) 
            {    
                std::cerr << "Process " << mpi_rank << " could not find platform from " << station_name
                        << " to " << dest_station_name << " for train " << train.train_id << " in incoming trains code section" << std::endl;

                // Debug: Print received message details
                std::cerr << "Debug: Received message from process " << status.MPI_SOURCE << std::endl;
                std::cerr << "  Station Name:  " << station_name << std::endl;
                std::cerr << "  Incoming Train ID: " << message[0] << std::endl;
                std::cerr << "  Current Station Index: " << message[1] << std::endl;
                std::cerr << "  Destination Station Index: " << message[2] << std::endl;
                std::cerr << "  Direction: " << message[3] << std::endl;
                std::cerr << "  Line (as int): " << message[4] << std::endl;
                std::cerr << "  Line (as char): " << line_int_to_char[message[4]] << std::endl;

            }

            // Update the train in the mapping
            trains[train.train_id] = train;
        }
    } 
    while (flag);


    // Only here, when we receive all the trains, then we decide which one to put into the platform
    // Platform is empty and no trains in holding queue
    for(auto &local_station_pair : local_stations)
    {
        TrainStation &station = local_station_pair.second;
        for (auto& platform : station.platforms) 
        {
            if (platform.platform_train_id == -1 && !platform.holding_queue.empty()) 
            {   
                
                auto [arrival_time, train_id] = platform.holding_queue.top();
                platform.holding_queue.pop();

                platform.platform_train_id = train_id;
                platform.loading_or_unloading_time_left = platform.pltg.next(train_id);

                
                // if (!platform.loading_or_unloading_time_passed_this_turn)
                // {
                //     // TODO: Not sure if this is needed
                //     platform.loading_or_unloading_time_left--;
                //     platform.loading_or_unloading_time_passed_this_turn = true;
                // }

                

                // std::cout << "Process " << mpi_rank << " train " << train.train_id
                // << " arrived at station " << station_name
                // << " and moved directly to platform to "
                // << dest_station_name << std::endl;
            }
            // 2. Platform's loading/unloading complete
            if (platform.platform_train_id != -1 && platform.loading_or_unloading_time_left <= 0) 
            {
                // If no transit train, move platform train to transit
                if (platform.transit_train_id == -1) 
                {
                    platform.transit_train_id = platform.platform_train_id;
                    platform.transit_train_time_left = static_cast<int>(platform.distance_to_dest);
                    if (debug)
                    {
                        std::cout << "Process " << mpi_rank << " moved train " << platform.platform_train_id
                            << " from platform at station " << platform.station_name
                            << " to transit to station " << platform.dest_station_name << std::endl;

                    }
                    
                    // Clear platform train
                    platform.platform_train_id = -1;
                    platform.loading_or_unloading_time_left = -1;
                } else 
                {
                    // Link is occupied; train cannot depart yet
                    // For simplicity, we do nothing in this case
                }
            }
            
                
        }
        
    }
    
    
}

void clear_all_time_passed_flags(unordered_map<string, TrainStation> &local_stations)
{
    for (auto& station_pair : local_stations) 
    {
        const std::string& station_name = station_pair.first;
        TrainStation& station = station_pair.second;

        for (Platform& platform : station.platforms) 
        {
            platform.transit_train_time_passed_this_turn = false;
            platform.loading_or_unloading_time_passed_this_turn = false;
        }
        
    }
}


void simulate(size_t num_stations, const vector<string> &station_names, const std::vector<size_t> &popularities,
              const adjacency_matrix &mat, const unordered_map<char, vector<string>> &station_lines, size_t ticks,
              const unordered_map<char, size_t> num_trains, size_t num_ticks_to_print, size_t mpi_rank,
              size_t total_processes) 
{
    // TODO: Implement this with MPI using total_processes
   

    /*
        PART 1: Creation of TrainStations and Platforms
    */

    unordered_map<string, TrainStation> stations; 
    create_train_stations_and_platforms(stations, station_names, mat, num_stations, station_lines, popularities);


    // Add the code here to print stations and platforms
    // Iterate through the stations and print their platforms

    if (debug)
    {
        for (const auto& station_pair : stations) 
        {
            const std::string& station_name = station_pair.first;
            const TrainStation& station = station_pair.second;

            std::cout << "Station Name: " << station_name << "\n";
            std::cout << "Station Index: " << station.station_idx << "\n";
            std::cout << "Station Popularity: " << station.popularity << "\n";

            std::cout << "Platforms:\n";
            for (const Platform& platform : station.platforms) {
                std::cout << "  From " << platform.station_name << " to " << platform.dest_station_name << "\n";
                std::cout << "    Distance to Destination: " << platform.distance_to_dest << "\n";
                std::cout << "    Platform Popularity: " << platform.popularity << "\n";
                std::cout << "    Line Colors: ";
                for (char color : platform.line_colors) {
                    std::cout << color << " ";
                }
                std::cout << "\n";
                std::cout << "    Is Start Platform: " << (platform.is_start ? "Yes" : "No") << "\n";
                std::cout << "    Is End Platform: " << (platform.is_end ? "Yes" : "No") << "\n";
                std::cout << "    Platform Train ID: " << platform.platform_train_id << "\n";
                std::cout << "    Transit Train ID: " << platform.transit_train_id << "\n";
                std::cout << "    Loading/Unloading Time Left: " << platform.loading_or_unloading_time_left << "\n";
                // Add any other platform info you wish to print
            }
            std::cout << "----------------------------------------\n";
        }
    }
    
    

    /*
        END OF PART 1: Creation of TrainStations and Platforms
    */




   /*
        START OF PART 2: Distribution of TrainStations among MPI Processes
   */

   
    // Mapping from station names to MPI ranks
    unordered_map<string, int> station_name_to_rank;

    // Vectors to hold station names assigned to this process
    vector<string> local_station_names;

    // Now, build local stations for this process
    unordered_map<string, TrainStation> local_stations;

    distribute_train_stations_among_processes(station_name_to_rank, local_station_names, local_stations, stations, station_names, mpi_rank, total_processes);

    int mpi_size;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    distribute_station_name_to_rank(station_name_to_rank,mpi_rank); // , // mpi_size);


    // All processes print the received mapping

    if (debug)
    {
        std::cout << "Process " << mpi_rank << " received the mapping:\n";
        for (const auto& pair : station_name_to_rank) 
        {
            std::cout << "  " << pair.first << " -> " << pair.second << "\n";
        }

    }
    
    // Each process now has its local_stations

    if (debug)
    {
        // // Add a delay to ensure processes print in order
        for (int i = 0; i < total_processes; ++i) 
        {
            if (mpi_rank == i) 
            {
                std::cout << "MPI Rank: " << mpi_rank << "\n";
                std::cout << "Local Stations for this process:\n";

                for (const auto& station_pair : local_stations) {
                    const std::string& station_name = station_pair.first;
                    const TrainStation& station = station_pair.second;

                    std::cout << "Station Name: " << station_name << "\n";
                    std::cout << "Station Index: " << station.station_idx << "\n";
                    std::cout << "Station Popularity: " << station.popularity << "\n";

                    std::cout << "Platforms:\n";
                    for (const Platform& platform : station.platforms) {
                        std::cout << "  From " << platform.station_name << " to " << platform.dest_station_name << "\n";
                        std::cout << "    Distance to Destination: " << platform.distance_to_dest << "\n";
                        std::cout << "    Platform Popularity: " << platform.popularity << "\n";
                        std::cout << "    Line Colors: ";
                        for (char color : platform.line_colors) {
                            std::cout << color << " ";
                        }
                        std::cout << "\n";
                        std::cout << "    Is Start Platform: " << (platform.is_start ? "Yes" : "No") << "\n";
                        std::cout << "    Is End Platform: " << (platform.is_end ? "Yes" : "No") << "\n";
                        std::cout << "    Platform Train ID: " << platform.platform_train_id << "\n";
                        std::cout << "    Transit Train ID: " << platform.transit_train_id << "\n";
                        std::cout << "    Loading/Unloading Time Left: " << platform.loading_or_unloading_time_left << "\n";
                    }
                    std::cout << "----------------------------------------\n";
                }
            }
            // Synchronize with other processes to ensure ordered output
            MPI_Barrier(MPI_COMM_WORLD);
            // Add a small delay for better visual separation (optional)
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
    

    /*
        END OF PART 2: Distribution of TrainStations among MPI Processes
   */


    /*
    START OF PART 3: Creating comm world and groups
    */


    // Map to store communicators for each line
    unordered_map<char, MPI_Comm> line_comms;
    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    // This does not work as it is not distributed across the processes, right now we are just using MPI_WORLD so dun care about this first
    unordered_map<char, unordered_map<int,int>> lines_world_rank_to_line_rank;

    create_mpi_comm_for_lines(line_comms, station_lines, station_name_to_rank, mpi_rank, world_group, lines_world_rank_to_line_rank);

    // Now, each process may have line communicators for the lines it is involved in
    // For lines the process is not involved in, the communicator will be MPI_COMM_NULL




    /*
    END OF PART 3: Creating comm world and groups
    */

    
    /*
        START OF PART 4: Main Simulation Loop
    */

    // Additional Data Structures
    unordered_map<int, Train> trains;  // Mapping from train_id to Train struct

    // Map station names to indices and vice versa
    unordered_map<string, int> station_name_to_idx;
    unordered_map<int, string> station_idx_to_name;

    int idx = 0;
    for (const string& station_name : station_names) 
    {
        station_name_to_idx[station_name] = idx;
        station_idx_to_name[idx] = station_name;
        idx++;
    }

    // Map line colors to sequences of station indices and create line-specific index to global index mapping
    unordered_map<char, vector<int>> line_station_indices;
    unordered_map<char, unordered_map<int, int>> line_idx_to_global_idx;

    for (const auto& pair : station_lines) 
    {
        char line = pair.first; // Now 'g', 'b', or 'y'
        const vector<string>& line_station_names = pair.second;

        vector<int> station_indices;
        unordered_map<int, int> idx_mapping; // Maps line-specific index to global index

        for (size_t i = 0; i < line_station_names.size(); ++i) 
        {
            const string& station_name = line_station_names[i];
            int global_station_idx = station_name_to_idx[station_name];
            station_indices.push_back(global_station_idx);
            idx_mapping[static_cast<int>(i)] = global_station_idx; // Line index i maps to global station index
        }
        line_station_indices[line] = station_indices;
        line_idx_to_global_idx[line] = idx_mapping;
    }

    // Map line identifiers to mappings of station names to line-specific station indices
    unordered_map<char, unordered_map<std::string, int>> station_name_to_idx_line;

    for (const auto& line_pair : station_lines) 
    {
        char line = line_pair.first;
        const std::vector<std::string>& line_station_names = line_pair.second;

        unordered_map<std::string, int> name_to_idx;

        for (size_t idx = 0; idx < line_station_names.size(); idx++) 
        {
            const std::string& station_name = line_station_names[idx];
            name_to_idx[station_name] = static_cast<int>(idx);
        }

        station_name_to_idx_line[line] = name_to_idx;
    }


    // OLD
    // // Map line colors to sequences of station indices
    // unordered_map<char, vector<int>> line_station_indices;

    // for (const auto& pair : station_lines) 
    // {
    //     char line = pair.first; // Now 'g', 'b', or 'y'
    //     const vector<string>& line_station_names = pair.second;

    //     vector<int> station_indices;
    //     for (const string& station_name : line_station_names) 
    //     {
    //         station_indices.push_back(station_name_to_idx[station_name]);
    //     }
    //     line_station_indices[line] = station_indices;
    // }

    // Initialize num_trains_mapping as a local copy to modify
    unordered_map<char, size_t> num_trains_mapping = num_trains;
    size_t next_train_id = 0;
    

    for(size_t current_tick = 0; current_tick < ticks; current_tick++)
    {
        // Step 1: Process 0 spawns new trains
        spawn_trains(mpi_rank, station_lines, num_trains_mapping, trains, local_stations, current_tick, next_train_id,line_idx_to_global_idx, station_name_to_idx_line); 
        //MPI_Barrier(MPI_COMM_WORLD);

        // Step 2: Process Local Updates
        std::vector<std::tuple<int,int,int,int,int,int>> trains_to_send_to_other_processes;
        process_local_updates(station_name_to_rank, local_stations, mpi_rank, line_station_indices, station_idx_to_name, trains_to_send_to_other_processes, trains, current_tick, line_idx_to_global_idx, station_name_to_idx_line);
        //MPI_Barrier(MPI_COMM_WORLD);

        // Step 3: Send Messages to other processes
        send_trains_to_other_processes(trains_to_send_to_other_processes,trains,mpi_rank);
        MPI_Barrier(MPI_COMM_WORLD);

        // Step 4: Process Incoming Trains from other processes

        receive_trains_from_other_processes(local_stations, station_idx_to_name,line_station_indices, mpi_rank,trains, current_tick);
        clear_all_time_passed_flags(local_stations);
        MPI_Barrier(MPI_COMM_WORLD);

        // Step 5: Collect and output train positions 
        if (current_tick >= ticks - num_ticks_to_print)
        {
            print_output(local_stations, trains, mpi_rank, total_processes, current_tick);
        }
        
        //MPI_Barrier(MPI_COMM_WORLD);


    }



   
    /*
        END OF PART 4: Main Simulation Loop
    */


    // **Cleanup**
    // Free communicators
    for (auto& pair : line_comms) 
    {
        if (pair.second != MPI_COMM_NULL) 
        {
            MPI_Comm_free(&pair.second);
        }
    }
    MPI_Group_free(&world_group);




}