# module Simulator

# using LinearAlgebra
# using StaticArrays
# using Random

# export ModelConfig, simulate_truth, simulate_measurements

# # Structure pour remplacer le dictionnaire 'model' de Python
# struct ModelConfig
#     # Dynamics: dt, sigma_w
#     dt::Float32
#     sigma_w::Float32
#     # Temporal: T
#     T::Int
#     # Grid: Nx, Ny, dx, dy
#     Nx::Int
#     Ny::Int
#     dx::Float32
#     dy::Float32
#     # PSF: I0, sigma_s
#     I0::Float32
#     sigma_s::Float32
#     # Objects: Liste de NamedTuples ou Dicts
#     objects::Vector{Any}
# end

# function F_matrix(omega, dt)
#     om = abs(omega) < 1f-6 ? 1f-6 : omega
#     s, c = sin(om * dt), cos(om * dt)
#     return @SMatrix [
#         1.0f0  0.0f0  s/om        -(1f0-c)/om;
#         0.0f0  1.0f0  (1f0-c)/om  s/om;
#         0.0f0  0.0f0  c           -Float32(s);
#         0.0f0  0.0f0  s           Float32(c)
#     ]
# end

# function G_matrix(dt)
#     dt2 = (dt^2) / 2.0f0
#     return @SMatrix [
#         dt2   0.0f0;
#         0.0f0  dt2;
#         dt    0.0f0;
#         0.0f0  dt
#     ]
# end

# function simulate_truth(m::ModelConfig)
#     G = G_matrix(m.dt)
#     num_objects = length(m.objects)
#     # On stocke des Vector{Union{Nothing, SVector}}
#     trajs = [Vector{Union{Nothing, SVector{5, Float32}}}(nothing, m.T) for _ in 1:num_objects]

#     for i in 1:num_objects
#         obj = m.objects[i]
#         # État: [x, y, vx, vy, omega]
#         x = [Float32(obj.pos[1]), Float32(obj.pos[2]), 
#              Float32(obj.vel[1]), Float32(obj.vel[2]), Float32(obj.omega)]
        
#         for t in 1:m.T
#             if t >= obj.t_birth && t <= obj.t_dead
#                 F = F_matrix(x[5], m.dt)
                
#                 # Bruit de process
#                 w = m.sigma_w * randn(Float32, 2)
#                 noise = G * w
                
#                 # Mise à jour des 4 premières composantes (cinématique)
#                 new_pos_vel = F * SVector{4, Float32}(x[1], x[2], x[3], x[4]) + noise
#                 x[1:4] .= new_pos_vel
                
#                 trajs[i][t] = SVector{5, Float32}(x...)
#             end
#         end
#     end
#     return trajs
# end

# function simulate_measurements(trajs, m::ModelConfig)
#     Z = Vector{Matrix{Float32}}(undef, m.T)
    
#     # Pré-calcul des grilles (équivalent meshgrid)
#     Xg = [Float32(i) for j in 0:m.Ny-1, i in 0:m.Nx-1]
#     Yg = [Float32(j) for j in 0:m.Ny-1, i in 0:m.Nx-1]
    
#     const_psf = (m.dx * m.dy * m.I0) / (2f0 * π * m.sigma_s^2)

#     for t in 1:m.T
#         S = zeros(Float32, m.Ny, m.Nx)
        
#         # Accumulation du signal des objets présents
#         for obj_idx in 1:length(trajs)
#             state = trajs[obj_idx][t]
#             if state !== nothing
#                 px, py = state[1], state[2]
#                 # Calcul vectorisé de la PSF
#                 @. S += const_psf * exp(-( (Xg - px)^2 + (Yg - py)^2 ) / (2f0 * m.sigma_s^2))
#             end
#         end
        
#         # Ajout du bruit blanc gaussien N(0,1)
#         Z[t] = S .+ randn(Float32, m.Ny, m.Nx)
#     end
#     return Z
# end

# end # module

module Simulator
include("model_yaml.jl")
using LinearAlgebra
using StaticArrays
using Random
using .ModelYAML  # Importation du module contenant la struct Model

export simulate_truth, simulate_measurements

# Matrice de transition Coordinated Turn (4x4 pour la cinématique)
function F_matrix(omega, dt)
    om = abs(omega) < 1f-6 ? 1f-6f0 : omega
    s, c = sin(om * dt), cos(om * dt)
    return @SMatrix [
        1.0f0  0.0f0  s/om        -(1f0-c)/om;
        0.0f0  1.0f0  (1f0-c)/om  s/om;
        0.0f0  0.0f0  c           -Float32(s);
        0.0f0  0.0f0  s           Float32(c)
    ]
end

# Matrice de bruit de process (4x2)
function G_matrix(dt)
    dt2 = (dt^2) / 2.0f0
    return @SMatrix [
        dt2    0.0f0;
        0.0f0  dt2;
        dt     0.0f0;
        0.0f0  dt
    ]
end

"""
Simule les trajectoires réelles à partir des objets définis dans le YAML.
"""
function simulate_truth(m)
    dt = m.dt
    σw = m.σw
    T = m.config["temporal"]["T"]
    G = G_matrix(dt)
    
    # Récupération des objets depuis la config YAML
    raw_objects = m.config["objects"]
    num_objects = length(raw_objects)
    
    # Liste de vecteurs de trajectoires (Union pour gérer les absences avant naissance/après mort)
    trajs = [Vector{Union{Nothing, SVector{5, Float32}}}(nothing, T) for _ in 1:num_objects]

    for i in 1:num_objects
        obj = raw_objects[i]
        
        # État initial complet [px, py, vx, vy, omega]
        # On accède aux clés du dictionnaire YAML
        x = Float32.([
            obj["position_init"][1], obj["position_init"][2],
            obj["velocity_init"][1], obj["velocity_init"][2],
            obj["omega_init"]
        ])
        
        t_birth = Int(obj["t_birth"])
        t_dead = Int(obj["t_dead"])
        
        for t in 1:T
            if t >= t_birth && t <= t_dead
                # On stocke l'état actuel avant propagation
                trajs[i][t] = SVector{5, Float32}(x...)
                
                # Propagation
                F = F_matrix(x[5], dt)
                w = σw * randn(Float32, 2)
                noise = G * w
                
                # Mise à jour de la cinématique (x, y, vx, vy)
                new_pos_vel = F * SVector{4, Float32}(x[1], x[2], x[3], x[4]) #+ noise
                x[1:4] .= new_pos_vel
                
                # Optionnel : update omega si sigma_u est défini, sinon constant
                # x[5] += ...
            end
        end
    end
    return trajs
end

"""
Génère les images de mesures Z (Heatmaps) à partir des trajectoires.
"""
function simulate_measurements(trajs, m)
    T = m.config["temporal"]["T"]
    Nx, Ny = m.Nx, m.Ny
    I0, σs = m.I0, m.σs
    
    # Z est un vecteur de matrices (images)
    Z = Vector{Matrix{Float32}}(undef, T)
    
    # Grilles pour le calcul de la PSF (indices de 0 à N-1 pour coller au Python)
    # On utilise Float32 pour la performance GPU ultérieure
    Xg = [Float32(i) for j in 0:Ny-1, i in 0:Nx-1]
    Yg = [Float32(j) for j in 0:Ny-1, i in 0:Nx-1]
    
    # Constante de normalisation de la PSF
    # Note : dx=1, dy=1 car basé sur les indices de pixels
    const_psf = (1.0f0 * 1.0f0 * I0) / (2f0 * π * σs^2)

    for t in 1:T
        # Signal pur (S)
        S = zeros(Float32, Ny, Nx)
        
        for obj_idx in 1:length(trajs)
            state = trajs[obj_idx][t]
            if state !== nothing
                px, py = state[1], state[2]
                # Calcul de la PSF cumulée
                @. S += const_psf * exp(-( (Xg - px)^2 + (Yg - py)^2 ) / (2f0 * σs^2))
            end
        end
        
        # Ajout du bruit de mesure Gaussien blanc N(0, 1)
        # Note : Si R est l'écart-type du bruit dans ton YAML, remplace randn par R * randn
        Z[t] = S .+ randn(Float32, Ny, Nx)
    end
    return Z
end

end # module