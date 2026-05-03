using PlotlyJS
using Statistics

function plot_measurements(Z, times)
    n = length(times)
    
    # Création des traces (Heatmaps)
    # On spécifie l'index de l'axe pour chaque subplot (xaxis="x1", "x2", etc.)
    traces = [
        heatmap(
            z = Z[t + 1], 
            colorscale = "Viridis",
            xaxis = "x$i",
            yaxis = "y$i"
        ) for (i, t) in enumerate(times)
    ]

    # Configuration du Layout avec sous-intrigue (subplots)
    # On définit les domaines de chaque axe pour les aligner horizontalement
    layout_args = Dict{Symbol, Any}(
        :title => "Measurements across time",
        :width => 400 * n,
        :height => 450,
        :showlegend => false
    )

    # Calcul dynamique des positions des axes
    for i in 1:n
        spacing = 0.05
        width_per_plot = (1.0 - (spacing * (n - 1))) / n
        start_x = (i - 1) * (width_per_plot + spacing)
        
        layout_args[Symbol("xaxis$i")] = attr(domain=[start_x, start_x + width_per_plot], title="t=$(times[i])")
        layout_args[Symbol("yaxis$i")] = attr(domain=[0, 1], scaleanchor="x$i", scaleratio=1)
    end

    relayout = Layout(; layout_args...)
    
    return plot(traces, relayout)
end


function plot_max_intensity(Z)
    # 1. Calcul de la projection d'intensité maximale (MIP)
    # On empile les mesures le long d'une nouvelle dimension et on prend le max
    # Z est supposé être un vecteur de matrices (T, Nx, Ny)
    z_stack = stack(Z) # Crée un array 3D [Nx, Ny, T]
    z_max = dropdims(maximum(z_stack, dims=3), dims=3)

    # 2. Création de la Heatmap
    trace = heatmap(
        z = z_max,
        colorscale = "Viridis",
        colorbar = attr(title = "Amplitude")
    )

    # 3. Configuration du Layout
    layout = Layout(
        title = "Intensité maximale des mesures (0->T)",
        xaxis = attr(title = "X", constrain = "domain"),
        yaxis = attr(title = "Y", scaleanchor = "x", scaleratio = 1),
        width = 600,
        height = 600
    )

    return plot(trace, layout)
end




function plot_results_plotly(Z, true_trajs, estimated_X; Nx=100, Ny=100)
    # 1. Calcul de la Max Projection (MIP)
    Z_tensor = stack(Z) # Crée un array [Nx, Ny, T]
    z_max = dropdims(maximum(Z_tensor, dims=3), dims=3)

    # Création de la structure Subplots (1 ligne, 2 colonnes)
    p = make_subplots(
    rows=1, 
    cols=2,
    # Utilise des espaces pour créer une matrice 1x2 au lieu d'un vecteur
    subplot_titles=["Intensité maximale (0->T)" "Trajectoires estimées"], 
    horizontal_spacing=0.1
    )

    # --- SUBPLOT 1 : Heatmap ---
    trace_heat = heatmap(
        z=z_max', # Transpose car Plotly inverse X et Y par rapport aux matrices Julia
        colorscale="Viridis",
        colorbar=attr(title="Amplitude", x=0.45)
    )
    add_trace!(p, trace_heat, row=1, col=1)

    # --- SUBPLOT 2 : Trajectoires ---
    
    # 1. Vérité Terrain (Lignes grises en pointillés)
    for (i, traj) in enumerate(true_trajs)
        pts = filter(x -> x !== nothing, traj)
        if !isempty(pts)
            px = [p[1] for p in pts]
            py = [p[2] for p in pts]
            
            # Ligne de trajectoire
            add_trace!(p, scatter(
                x=px, y=py, mode="lines",
                line=attr(color="gray", dash="dash", width=1),
                opacity=0.5, name="Vérité $i", showlegend=(i==1)
            ), row=1, col=2)
            
            # Point de départ
            add_trace!(p, scatter(
                x=[px[1]], y=[py[1]], mode="markers",
                marker=attr(symbol="x", color="black", size=5),
                showlegend=false
            ), row=1, col=2)
        end
    end

    # 2. Estimations
    all_labels = unique(vcat([collect(keys(d)) for d in estimated_X]...))
    
    for (idx, lbl) in enumerate(all_labels)
        est_px = Float32[]
        est_py = Float32[]
        
        for t in 1:length(estimated_X)
            state = get(estimated_X[t], lbl, nothing)
            if state !== nothing
                push!(est_px, state[1])
                push!(est_py, state[2])
            end
        end

        if !isempty(est_px)
            # Ligne + points pour l'estimation
            add_trace!(p, scatter(
                x=est_px, y=est_py, mode="lines+markers",
                marker=attr(size=4),
                line=attr(width=2),
                name="Label $(lbl.id), $(lbl.t_birth)", showlegend=true
            ), row=1, col=2)
            
            # Annotation (Temps de naissance)
            # Note : Les annotations simples s'ajoutent souvent au layout final
        end
    end

    # Mise à jour du Layout
    relayout!(p,
        width=1200, height=600,
        xaxis1=attr(scaleanchor="y1", scaleratio=1), # Aspect ratio égal pour la heatmap
        xaxis2=attr(title="Position X", range=[1, Nx], scaleanchor="y2", scaleratio=1),
        yaxis2=attr(title="Position Y", range=[1, Ny]),
        template="plotly_white"
    )

    return p
end