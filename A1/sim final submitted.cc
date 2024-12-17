#include <omp.h>

#include <cmath>
#include <fstream>
#include <vector>

#include "collision.h"
#include "io.h"
#include "sim_validator.h"



inline bool isBorder(int i, int L) {
    int row = i / L;
    int col = i % L;
    
    return (row == 0 || row == L - 1 || col == 0 || col == L - 1);
}


inline bool is_particle_collision_better(Vec2 loc1, Vec2 vel1, Vec2 loc2, Vec2 vel2, int radius) {
    return is_particle_moving_closer(loc1, vel1, loc2, vel2) && is_particle_overlap(loc1, loc2, radius);
}

bool is_resolved_between_two_grids(std::vector<int>&v1, std::vector<int>&v2, std::vector<Particle>&particles, int radius)
{
    for(unsigned int i = 0; i < v1.size(); i++)
    {
        int p1_idx = v1[i];
        for (unsigned int j = 0; j < v2.size(); j++)
        {
            int p2_idx = v2[j];
            if(p1_idx == p2_idx) continue; // same particle we dc

            if(is_particle_collision_better(particles[p1_idx].loc, particles[p1_idx].vel, particles[p2_idx].loc, particles[p2_idx].vel, radius))
            {
                return false;   
            }
        }
    }  
    return true;
}

bool is_resolved_grid_with_wall(std::vector<int>&v1, std::vector<Particle>&particles, int square_size, int radius)
{
    for(unsigned int i = 0; i < v1.size(); i++)
    {
        int p1_idx = v1[i];
        if
        (
            is_wall_overlap(particles[p1_idx].loc, square_size, radius) 
            &&
            is_wall_collision(particles[p1_idx].loc, particles[p1_idx].vel, square_size, radius)
        )
        {  
            return false;
        }
    }
    return true;
}

void clear_and_reassign(std::vector<std::vector<int>> &grid, std::vector<Particle>&particles, int grid_len, int square_size)
{
    
    for (auto& v : grid) 
        v.clear();
    
    for(unsigned int j = 0; j < particles.size(); j++)
    {
        int grid_x = floor( (particles[j].loc.x * grid_len) / square_size );
        int grid_y = floor( (particles[j].loc.y * grid_len) / square_size );


        // Clamp grid_x and grid_y to ensure they stay within the valid grid indices
        if (grid_x < 0) grid_x = 0;
        if (grid_x >= grid_len) grid_x = grid_len - 1;
        if (grid_y < 0) grid_y = 0;
        if (grid_y >= grid_len) grid_y = grid_len - 1;

        grid[grid_x + grid_y * grid_len].push_back(particles[j].i);
        
        
        
    }
}

bool resolve_two_grids_particles(std::vector<int>&v1, std::vector<int>&v2, std::vector<Particle>&particles, int radius)
{
    bool collisions_detected = false;
    for(unsigned int i = 0; i < v1.size(); i++)
    {
        int p1_idx = v1[i];
        // Particle to Particle Collision Check
        for (unsigned int j = 0; j < v2.size(); j++)
        {
            int p2_idx = v2[j];

            if(p1_idx == p2_idx) continue; // same particle we dc
            
            if(is_particle_collision_better(particles[p1_idx].loc, particles[p1_idx].vel, particles[p2_idx].loc, particles[p2_idx].vel, radius))
            {   
                #pragma omp critical
                {
                    resolve_particle_collision(particles[p1_idx].loc, particles[p1_idx].vel, particles[p2_idx].loc, particles[p2_idx].vel);   
                }
                collisions_detected = true;
            }
            
            
        }
    }
    return collisions_detected;  
}

bool resolve_grid_with_wall(std::vector<int>&v1, std::vector<Particle>&particles, int square_size, int radius)
{
    bool collisions_detected = false;
    for(unsigned int i = 0; i < v1.size(); i++)
    {
        int p1_idx = v1[i];
        if
        (
            is_wall_overlap(particles[p1_idx].loc, square_size, radius) 
            &&
            is_wall_collision(particles[p1_idx].loc, particles[p1_idx].vel, square_size, radius)
        )
        {  
            #pragma omp critical
            {
                resolve_wall_collision(particles[p1_idx].loc, particles[p1_idx].vel, square_size, radius);                                    
            }
            collisions_detected = true;
        }
    }
    return collisions_detected;
}


int main(int argc, char* argv[]) {
    // Read arguments and input file
    Params params{};
    std::vector<Particle> particles;
    read_args(argc, argv, params, particles);

    // Set number of threads
    omp_set_num_threads(params.param_threads);


#if CHECK == 1
    // Initialize collision checker
    SimulationValidator validator(params.param_particles, params.square_size, params.param_radius);
    // Initialize with starting positions
    validator.initialize(particles);

    // Uncomment the line below to enable visualization (makes program much slower)
   // validator.enable_viz_output("test.out");
#endif


    // TODO: this is the part where you simulate particle behavior.


    int grid_len = ceil(static_cast<double>(params.square_size) / (2.1* params.param_radius));

    
    std::vector<std::vector<int>> grid(grid_len * grid_len);

    

    // initially Assigning particles
    clear_and_reassign(grid,particles,grid_len, params.square_size);
    
    
    

    /* Main Loop 
    */
    for(int a = 0; a < params.param_steps; a++)
    {

        // // Phase 1: Updating of Positions
        #pragma omp parallel for schedule(static) // this actually slows it down but just a bit but i think if N is big it will help
        for(unsigned int c = 0; c < particles.size(); c++)
        {
            particles[c].loc.x += particles[c].vel.x;
            particles[c].loc.y += particles[c].vel.y;
            // might or might not be needed
        }
        // Reassign particles to new grid cells 
        clear_and_reassign(grid,particles,grid_len, params.square_size);


        


        bool collisions_detected;

        do 
        {
            collisions_detected = false;

            #pragma omp parallel for schedule(static) reduction(|| : collisions_detected)
            for(unsigned int grid_idx = 0; grid_idx < (unsigned int)(grid_len*grid_len); grid_idx++)
            {
                std::vector<int> v1 = grid[grid_idx];

                if (isBorder(grid_idx, grid_len)) 
                {
                    if (resolve_grid_with_wall(v1,particles,params.square_size,params.param_radius))
                    {
                        collisions_detected = true;
                    }
                }
                
                
           
                for (int x_offset = -1; x_offset <= 1; x_offset++)  // we want to check all 9 directions to increase chances of zero collisions remaining
                {
                    for (int y_offset = -1; y_offset <= 1; y_offset++) 
                    {   
                        int neighbor_idx = grid_idx + x_offset + y_offset * grid_len;
                        if (neighbor_idx < 0 || neighbor_idx >= grid_len*grid_len) continue;

                        std::vector<int> v2 = grid[neighbor_idx];
                        if (resolve_two_grids_particles(v1,v2,particles,params.param_radius))
                        {
                            collisions_detected = true;
                        }
     
                    }
                }
            }
        } while(collisions_detected);


        
        #if CHECK == 1
        validator.validate_step(particles);
        #endif
        

        


        

        /*
        After simulating each timestep, you must call:

        #if CHECK == 1
        validator.validate_step(particles);
        #endif
        */
    } // End of main logic Loop
}



