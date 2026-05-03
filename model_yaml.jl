
module ModelYAML
    using YAML, StaticArrays

    export Model

    struct Model
        config::Dict{String, Any}
        Nx::Int; Ny::Int
        I0::Float32; σs::Float32; R::Float32
        Ps::Float32; Pb::Float32
        dt::Float32; σw::Float32
        C_birth::SMatrix{5, 5, Float32, 25}
        birth_means::Vector{SVector{5, Float32}}
        gibbs_iterations::Int
        T::Int
        n_particles::Int
        n_steps_pf::Int

        function Model(path::String)
            c = YAML.load_file(path)
            obj_means = [SVector{5, Float32}(Float32.(vcat(o["position_init"], o["velocity_init"], o["omega_init"]))) for o in c["objects"]]
            
            # Conversion propre de la matrice de covariance
            raw_cov = c["birth_general"]["C_birth"]
            C_mat = zeros(Float32, 5, 5)
            for i in 1:5, j in 1:5
                C_mat[i,j] = Float32(raw_cov[i][j])
            end
            C_b = SMatrix{5, 5, Float32}(C_mat)
            
            new(c, c["grid"]["Nx"], c["grid"]["Ny"], 
                Float32(c["psf"]["I0"]), Float32(c["psf"]["sigma_s"]), Float32(c["measurement"]["R"]),
                Float32(c["survival"]["Ps"]), Float32(c["birth_general"]["Pb"]),
                Float32(c["dynamics"]["dt"]), Float32(c["dynamics"]["sigma_w"]),
                C_b, obj_means, Int(c["gibbs"]["iterations"]), Int(c["temporal"]["T"]), Int(c["particle_flow"]["n_particles"]), Int(c["particle_flow"]["n_steps"]))
        end
    end

end # module
