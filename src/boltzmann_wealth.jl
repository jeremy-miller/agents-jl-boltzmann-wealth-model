module boltzmann_wealth

using Agents, CairoMakie, Random

mutable struct WealthAgent <: AbstractAgent
    id::Int
    pos::NTuple{2,Int}
    wealth::Int
end

function agent_step!(agent, model)
    agent.wealth == 0 && return
    neighboring_positions = collect(nearby_positions(agent.pos, model))
    push!(neighboring_positions, agent.pos)  # also consider current position
    random_pos = rand(model.rng, neighboring_positions)  # position to exchange with
    available_ids = ids_in_position(random_pos, model)
    if length(available_ids) > 0
        random_neighbor_agent = model[rand(model.rng, available_ids)]
        agent.wealth -= 1
        random_neighbor_agent.wealth += 1
    end
end

function wealth_model(; dims=(25, 25), wealth=1, M=1000)
    space = GridSpace(dims, periodic=true)
    model = ABM(WealthAgent, space; scheduler=Schedulers.randomly)
    for _ in 1:M
        add_agent!(model, wealth)
    end
    return model
end

function wealth_distribution(data, model, n)
    W = zeros(Int, size(model.space))
    for row in eachrow(filter(r -> r.step == n, data))  # iterate over rows at a specific step
        W[row.pos...] += row.wealth
    end
    return W
end

function make_heatmap(W)
    figure = Figure(; resolution=(600, 450))
    hmap_l = figure[1, 1] = Axis(figure)
    hmap = heatmap!(hmap_l, W; colormap=cgrad(:default))
    cbar = figure[1, 2] = Colorbar(figure, hmap; width=30)
    return figure
end

init_wealth = 4
model = wealth_model(; wealth=init_wealth)
adata = [:wealth, :pos]
data, _ = run!(model, agent_step!, 10; adata=adata, when=[1, 5, 9])
W1 = wealth_distribution(data, model, 9)
make_heatmap(W1)

end # module boltzmann_wealth
