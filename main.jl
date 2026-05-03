include("model_yaml.jl")
using .ModelYAML
include("gen_meas.jl")
include("gibbs.jl")

include("plot_utils.jl")
using .Simulator



model = ModelYAML.Model("config.yaml")

trajs = simulate_truth(model)
measurements = simulate_measurements(trajs, model)



Z_GPU = [CuArray(measurements[t]) for t in 1:model.config["temporal"]["T"]]


function extract_estimates(ws::MOTWorkspace)
    T, num_tracks = ws.T, ws.max_tracks
    num_particles = ws.num_particles
    
    
    particles_cpu = Array(ws.particles)
    is_active_cpu = ws.is_active  
    labels_cpu = ws.labels
    
    estimated_X = [Dict{VoLabel, Vector{Float32}}() for _ in 1:T]
    
    for t in 1:T
        active_dix_t = findall(is_active_cpu[t, :]) 
        for k in active_dix_t
            if is_active_cpu[t, k]
                parts_x = view(particles_cpu, 1, :, t, k)
                parts_y = view(particles_cpu, 2, :, t, k)
                
                mean_x = sum(parts_x) / num_particles
                mean_y = sum(parts_y) / num_particles
                
                #
                label = labels_cpu[t, k] 
                if label != VoLabel(-1, -1) 
                    estimated_X[t][label] = [mean_x, mean_y]
                end
            end
        end
    end
    
    return estimated_X
end


results = run_full_gibbs_MOT(model, Z_GPU)

estimated_X = extract_estimates(results)
# print(estimated_X)
display(plot_results_plotly(measurements, trajs ,estimated_X))