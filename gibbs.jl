using .ModelYAML
using CUDA, StaticArrays



struct VoLabel
    t_birth::Int
    id::Int
end

mutable struct MOTWorkspace
    T::Int
    max_tracks::Int
    num_particles::Int
    
    # [État(5), Particules(N), Temps(T), Tracks(K)]
    particles::CuArray{Float32, 4}

    
    # [Temps, Tracks] : Labels persistants (t_birth, id)
    labels::Matrix{VoLabel} 
    
    # Image du signal cumulé pour chaque t [Nx, Ny, T]
    S_total::CuArray{Float32, 3} # CACHING
    is_active::BitMatrix  # Matrix{Bool} de taille (T, max_tracks)

end

function MOTWorkspace(model::Model, T::Int, max_tracks::Int, num_particles::Int)
    particles = CUDA.zeros(Float32, 5, num_particles, T, max_tracks)
    labels = fill(VoLabel(-1, -1), T, max_tracks)
    S_total = CUDA.zeros(Float32, model.Nx, model.Ny, T)
    is_active = falses(T, max_tracks) # Initialement, aucun track n'est actif
    
    return MOTWorkspace(T, max_tracks, num_particles, particles, labels, S_total, is_active)
end




function init_pf_kernel!(particles, t, k, pmx, pmy, vmx, vmy, ωm, ppx, ppy, vpx, vpy, ωp, dt, n, has_m, has_p)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= n
        # Initialisation par défaut pour éviter les variables indéfinies
        fwd_px, fwd_py, fwd_vx, fwd_vy = 0f0, 0f0, 0f0, 0f0
        bwd_px, bwd_py, bwd_vx, bwd_vy = 0f0, 0f0, 0f0, 0f0

        # --- PREDICTION FORWARD ---
        if has_m
            # PROTECTION DIVISION PAR ZERO (ωm -> 0)
            abs_ω = max(abs(ωm), 1f-6) 
            # abs_ω = ωm

            s, c = sin(abs_ω * dt), cos(abs_ω * dt)
            
            fwd_px = pmx + (s/abs_ω)*vmx - ((1f0-c)/abs_ω)*vmy
            fwd_py = pmy + ((1f0-c)/abs_ω)*vmx + (s/abs_ω)*vmy
            fwd_vx = c*vmx - s*vmy
            fwd_vy = s*vmx + c*vmy
        end

        # --- PREDICTION BACKWARD ---
        if has_p
            abs_ω = max(abs(ωp), 1f-6)
            # abs_ω = ωp
            s, c = sin(abs_ω * -dt), cos(abs_ω * -dt)
            
            bwd_px = ppx + (s/abs_ω)*vpx - ((1f0-c)/abs_ω)*vpy
            bwd_py = ppy + ((1f0-c)/abs_ω)*vpx + (s/abs_ω)*vpy
            bwd_vx = c*vpx - s*vpy
            bwd_vy = s*vpx + c*vpy
        end

        # --- COMBINAISON ---
        if has_m && has_p
            particles[1, idx, t, k] = (fwd_px + bwd_px) * 0.5f0
            particles[2, idx, t, k] = (fwd_py + bwd_py) * 0.5f0
            particles[3, idx, t, k] = (fwd_vx + bwd_vx) * 0.5f0
            particles[4, idx, t, k] = (fwd_vy + bwd_vy) * 0.5f0
            particles[5, idx, t, k] = (ωm + ωp) * 0.5f0
        elseif has_m
            particles[1, idx, t, k] = fwd_px; particles[2, idx, t, k] = fwd_py
            particles[3, idx, t, k] = fwd_vx; particles[4, idx, t, k] = fwd_vy
            particles[5, idx, t, k] = ωm
        elseif has_p
            particles[1, idx, t, k] = bwd_px; particles[2, idx, t, k] = bwd_py
            particles[3, idx, t, k] = bwd_vx; particles[4, idx, t, k] = bwd_vy
            particles[5, idx, t, k] = ωp
        end

    end
    return nothing
end



function gromov_derivatives_kernel!(particles, t, k, Grad_L, Hess_L, Z, S_others, Nx, Ny, σs, I0, R, n)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= n
        @inbounds begin
            px = particles[1, idx, t, k]
            py = particles[2, idx, t, k]
            
            σs2 = σs * σs
            σs4 = σs2 * σs2
            norm_factor = I0 / (2.0f0 * Float32(pi) * σs2)
            inv_R = 1.0f0 / R
            
            gx, gy = 0.0f0, 0.0f0
            
            Hxx, Hxy, Hyy = 0.0f0, 0.0f0, 0.0f0
            
            win = Int(ceil(3.5f0 * σs))
             
            ix, iy = Int(floor(px)), Int(floor(py))
            
            @fastmath for ox in -win:win, oy in -win:win
                cx, cy = ix + ox, iy + oy
                if cx >= 1 && cx <= Nx && cy >= 1 && cy <= Ny
                    dx = cx - px
                    dy = cy - py
                    dist2 = dx*dx + dy*dy
                    
                    # A_m(x) :  PSF
                    Am = norm_factor * exp(-dist2 / (2.0f0 * σs2))
                    
                    # Résidu : Z_m - S_others_m
                    res = Z[cx, cy] - S_others[cx, cy, t]
                    
                    # Compute Grad
                    # d_Am_dx = Am * dx / σs2
                    term_grad = (res - Am) * Am / σs2
                    gx += term_grad * dx
                    gy += term_grad * dy
                    
                    # Compute Hess(Approximation  Gauss-Newton)
                    
                    # H_g ≈ - (1/R) * (∇Am) * (∇Am)^T
                    term_hess = -inv_R * (Am / σs2) * (Am / σs2)
                    
                    Hxx += term_hess * (dx * dx)
                    Hxy += term_hess * (dx * dy)
                    Hyy += term_hess * (dy * dy)
                end
            end
            
            Grad_L[1, idx] = gx * inv_R
            Grad_L[2, idx] = gy * inv_R
            
            Hess_L[1, idx] = Hxx
            Hess_L[2, idx] = Hxy
            Hess_L[3, idx] = Hyy
        end
    end
    return nothing
end



function pure_gromov_update_kernel!(particles, t, k, Grad_L, Hess_L, tau, dt_flow, n, pmx, pmy, var_prior)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= n
        @inbounds begin
            px = particles[1, idx, t, k]
            py = particles[2, idx, t, k]
            
            
            gx = Grad_L[1, idx]
            gy = Grad_L[2, idx]
            Hxx_L = Hess_L[1, idx]
            Hxy_L = Hess_L[2, idx]
            Hyy_L = Hess_L[3, idx]
            
            # Grad and Hess of prior
            # Hypo : Prior Gaussien P(x) = N(x; pm, var_prior)
            # ∇lp = -(x - μ) / σ²
            # ∇²lp = -1 / σ²
            inv_var = 1.0f0 / var_prior
            gx_P = -(px - pmx) * inv_var
            gy_P = -(py - pmy) * inv_var
            
            Hxx_P = -inv_var
            Hyy_P = -inv_var
            Hxy_P = 0.0f0
            
            # (H_pz = H_p + tau * H_g)
            Hxx = Hxx_P + tau * Hxx_L
            Hyy = Hyy_P + tau * Hyy_L
            Hxy = Hxy_P + tau * Hxy_L
            
            # Inv Hess 
            det = Hxx * Hyy - Hxy * Hxy
            
            # Régularisation (Tikhonov) au cas où le déterminant est trop proche de 0
            if abs(det) < 1e-6
                Hxx -= 0.0001f0
                Hyy -= 0.0001f0
                det = Hxx * Hyy - Hxy * Hxy
            end
            
            inv_det = 1.0f0 / det
            Inv_Hxx = Hyy * inv_det
            Inv_Hyy = Hxx * inv_det
            Inv_Hxy = -Hxy * inv_det
            
            # Compute drift (fz)
            # fz = [H_pz]^(-1) * (-∇l_g + Kz * ∇l_p)
            # En version déterministe (très stable), Kz ≈ 0, donc f_z = - [H_pz]^(-1) * (∇l_p + ∇l_g)
            
            force_x = -(gx_P + gx)
            force_y = -(gy_P + gy)
            
            fz_x = Inv_Hxx * force_x + Inv_Hxy * force_y
            fz_y = Inv_Hxy * force_x + Inv_Hyy * force_y
            
            particles[1, idx, t, k] += fz_x * dt_flow
            particles[2, idx, t, k] += fz_y * dt_flow
            
        end
    end
    return nothing
end





function particle_flow_step!(particles::CuArray{Float32, 4}, t::Int, k::Int, S_others::CuArray{Float32, 3}, z_t::CuArray{Float32, 2}, model::Model, bool_exist_prec::Bool, idx_VoLabel_prec::Int, bool_exist_succ::Bool, idx_VoLabel_succ::Int, is_birth::Bool, n_steps::Int)
    n = size(particles, 2)
    threads = 256
    blocks = ceil(Int, n/threads)
    dt_flow = 1.0f0 / n_steps
    
    # Pré-allocation (TODO : put in  MOTWorkspace)
    Grad_L = CUDA.zeros(Float32, 2, n) # Seulement besoin de x et y
    Hess_L = CUDA.zeros(Float32, 3, n) # Hxx, Hxy, Hyy
    
    function get_mean(t_idx, k_idx)
        m = dropdims(sum(view(particles, :, :, t_idx, k_idx), dims=2), dims=2) ./ Float32(n)
        return Array(m)
    end

    pmx, pmy, vmx, vmy, ωm = bool_exist_prec && idx_VoLabel_prec > 0 ? get_mean(t-1, idx_VoLabel_prec) : (0f0,0f0,0f0,0f0,0f0)
    ppx, ppy, vpx, vpy, ωp = bool_exist_succ && idx_VoLabel_succ > 0 ? get_mean(t+1, idx_VoLabel_succ) : (0f0,0f0,0f0,0f0,0f0)

    # Compute p_prior(Predict FW/BW)
    @cuda threads=threads blocks=blocks init_pf_kernel!(particles, t, k, pmx, pmy, vmx, vmy, ωm, ppx, ppy, vpx, vpy, ωp, model.dt, n, bool_exist_prec, bool_exist_succ)
    
    # var_prior
    var_prior = 5.0f0 

    for i in 0:n_steps-1
        tau = (i + 1.0f0) / n_steps
        
        # COMPUTE GRAD AND HESS
        @cuda threads=threads blocks=blocks gromov_derivatives_kernel!(
            particles, t, k, Grad_L, Hess_L, z_t, S_others, 
            model.Nx, model.Ny, model.σs, model.I0, model.R, n
        )

        
        @cuda threads=threads blocks=blocks pure_gromov_update_kernel!(
            particles, t, k, Grad_L, Hess_L, tau, dt_flow, n, pmx, pmy, var_prior
        )
    end
    
    CUDA.synchronize()
end





function s_total_kernel!(S_total, P, Nx, Ny, I0, σs, n)
    cx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    cy = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if cx <= Nx && cy <= Ny
        val = 0.0f0
        σs2 = σs * σs
        norm = I0 / (2f0 * π * σs2)
        
        for k in 1:n
            dx = cx - P[1, k]
            dy = cy - P[2, k]
            val += norm * exp(-(dx*dx + dy*dy) / (2f0 * σs2))
        end
        S_total[cx, cy] = val
    end
    return nothing
end

function method_kernel(P, Nx, Ny, I0, σs, n)
    output = CUDA.zeros(Float32, Nx, Ny)
    threads = (16, 16)
    blocks = (ceil(Int, Nx/threads[1]), ceil(Int, Ny/threads[2]))
    @cuda threads=threads blocks=blocks s_total_kernel!(output, P, Nx, Ny, I0, σs, n)
    return output
end

function compute_s_total!(S_total_t::CuArray{Float32, 2}, P::CuArray{Float32, 2}, model::Model)
    Nx, Ny = model.Nx, model.Ny
    I0, σs = model.I0, model.σs
    n = size(P, 2)  # nombre de pistes actives à ce t
    threads = (16, 16)
    blocks = (ceil(Int, Nx/threads[1]), ceil(Int, Ny/threads[2]))
    @cuda threads=threads blocks=blocks s_total_kernel!(S_total_t, P, Nx, Ny, I0, σs, n)
end

function update_s_total!(ws::MOTWorkspace, model::Model, t::Int)
    active_inds = findall(ws.is_active[t, :])
    if isempty(active_inds)
        ws.S_total[:, :, t] .= 0.0f0
        return
    end

    P_active = view(ws.particles, 1:2, :, t, active_inds)
    P_means_active = sum(P_active, dims=2) ./ size(P_active, 2)
    P_means_active = reshape(P_means_active, 2, :)  # 2 x n_active

    compute_s_total!(view(ws.S_total, :, :, t), P_means_active, model)
    #ws.S_total[:, :, t] .= method_kernel(P_means_active, model.Nx, model.Ny, model.I0, model.σs, size(P_means_active, 2))
end




function logsumexp(x)
    xmax = maximum(x)
    return xmax + log(sum(exp.(x .- xmax)))
end



function mos_gibbs_sampler!(t::Int, ws::MOTWorkspace, model::Model, Z_gpu_t::CuArray{Float32, 2})
    
    # L(X-) : Indices des tracks qui étaient actifs à t-1
    survivor_candidates = t > 1 ? findall(ws.is_active[t-1, :]) : Int[]
    
    # B : Indices disponibles pour la naissance (slots inactifs à t-1 et t)
    # On cherche des slots vides dans ws.is_active
    inactive_slots_t = findall(.!ws.is_active[t, :]) 
    num_births = min(length(inactive_slots_t), length(model.birth_means))
    
    birth_slots_candidates = inactive_slots_t[1:num_births]
    
    #Pour les labels surviant actif ie de t-1, on cheque si leur labels est aussi à t si c'est oui alors slot_survivoir c'est ce K sinon on prend un inactif

    # On regarde tous les IDs déjà attribués à travers le temps pour éviter les collisions
    all_ids = [l.id for l in ws.labels[t, :] if l.id != -1]
    next_id = isempty(all_ids) ? 0 : maximum(all_ids) + 1

   
    #Pour les birth 
    l_idx_prec = [-1 for _ in birth_slots_candidates]
    l_k = copy(birth_slots_candidates)
    l_idx_succ = [-1 for _ in birth_slots_candidates]
    # Pour les survivors, on cherche les indices de leurs labels à t-1 et t+1
    i = 1
    for k in survivor_candidates
        label = ws.labels[t-1, k]
        idx_prec = k # On sait que c'est le même index pour les survivors
        slot_t = findfirst(==(label), ws.labels[t, :])
        if isnothing(slot_t) 
            slot_t = inactive_slots_t[num_births + i] # avalaible slots after birth slots
            i += 1
        end
        idx_succ = t < model.T ? findfirst(==(label), ws.labels[t+1, :]) : -1
        if isnothing(idx_succ) 
            idx_succ = -1
        end
        push!(l_idx_prec, idx_prec)
        push!(l_k, slot_t)
        push!(l_idx_succ, idx_succ)
    end

    # Init with b(.) prior
    for (i, k) in enumerate(birth_slots_candidates)
        
        
        ws.particles[1, :, t, k] .= model.birth_means[i][1] 
        ws.particles[2, :, t, k] .= model.birth_means[i][2] 

        ws.particles[:, :, t, k] .+ 0.01f0 * CUDA.randn(Float32, 5, ws.num_particles)

        ws.labels[t, k] = VoLabel(t, next_id)
        next_id += 1
        ws.is_active[t, k] = true # On active le slot pour le calcul de S_total
    end

    ws.S_total[:, :, t] .= 0.0f0
    update_s_total!(ws, model, t) 



    for (idx_prec, k, idx_succ) in zip(l_idx_prec, l_k, l_idx_succ)
        

        is_birth_slot = k in birth_slots_candidates

        
        # Birth init
        if is_birth_slot 
            label = ws.labels[t, k] # Mise à jour du label local
            # println("Birth track $k at time $t with label (t_birth=$(label.t_birth), id=$(label.id))")
        else #survivor slot, on garde le label et les particules existants
            label = ws.labels[t-1, idx_prec] 
            # println("Survivor track $k at time $t with label (t_birth=$(label.t_birth), id=$(label.id))")
        end


        bool_exist_prec = idx_prec != -1
        bool_exist_succ = idx_succ != -1

        # compute r_prior
        if t == 1
            r_prior = bool_exist_succ ? 1.0f0 : model.Pb
        elseif t == ws.T
            r_prior = bool_exist_prec ? model.Ps : model.Pb
        else
            r_prior = bool_exist_succ ? 1.0f0 : (bool_exist_prec ? model.Ps : model.Pb)
        end

        # Compute p_prior and p_post
        particle_flow_step!(ws.particles, t, k, ws.S_total, Z_gpu_t, model, 
                            bool_exist_prec, idx_prec, 
                            bool_exist_succ, idx_succ, is_birth_slot, model.n_steps_pf)                

        # Compute r_post
        P_mean  = sum(view(ws.particles, 1:2, :, t, k), dims=2) ./ ws.num_particles
        h_new = method_kernel(P_mean, model.Nx, model.Ny, model.I0, model.σs, 1)

        # log g(z | x_l ∪ X_{bar l})
        log_g_with = -0.5f0 * sum((Z_gpu_t .- ws.S_total[:, :, t] .- h_new).^2) / model.R
        # log g(z | X_{bar l})
        log_g_without = -0.5f0 * sum((Z_gpu_t .- ws.S_total[:, :, t]).^2) / model.R
        
        log_u = log(r_prior) + log_g_with
        log_v = log(1 - r_prior) + log_g_without

      
        r_post = exp(log_u - logsumexp([log_u, log_v])) 
        
        
        # println("log_g_with = $log_g_with, log_g_without = $log_g_without, log_u = $log_u, log_v = $log_v")
        # println("Track $k at time $t with label (t_birth=$(label.t_birth), id=$(label.id)) has r_prior=$r_prior and r_post=$r_post")
        
        if rand() < r_post
            # println()
            # println()
            # println()
            # print("AAAAAACCCEEPPTED track $k at time $t with label (t_birth=$(label.t_birth), id=$(label.id)) and r_post=$r_post")
            ws.is_active[t, k] = true
            ws.labels[t, k] = label 
            ws.S_total[:, :, t] .+= h_new # CACHING
        else
            ws.is_active[t, k] = false
            ws.labels[t, k] = VoLabel(-1, -1) 
        end
    end
end









# function run_full_gibbs_MOT(model::Model, Z_gpu::Vector{CuArray{Float32, 2}})
function run_full_gibbs_MOT(model::Model, Z_gpu)

    T = model.T
    max_tracks = 50 # À ajuster selon tes besoins
    num_particles = 200
    
    # ALLOCATE GPU MEMORY 
    ws = MOTWorkspace(model, T, max_tracks, num_particles)
    
    # INIT ALGO 2 TODO
    
    
    for i in 1:model.gibbs_iterations
        println("Gibbs Iteration $i / $(model.gibbs_iterations)")
        # FW / BW 
        #time_order = (i % 2 == 1) ? (1:T) : (T:-1:1)
        time_order = 1:T
        for t in time_order
            mos_gibbs_sampler!(t, ws, model, Z_gpu[t])
        end
    end
    
    return ws
end























