#module DynamicCorporateModels

using PyPlot
using Optim
#using BlackBoxOptim
using LinearAlgebra, Statistics
using Parameters, QuantEcon, DataFrames, Plots, Random
using Interpolations
using BenchmarkTools


#Here we start putting the Neoclassical models and add extensions
function NeoClassical(;
#Flavours
    cash::Bool=false,
    costly_equity::Bool=true,
    asymmetric_investment::Bool=false,
    quadratic_investment::Bool=true,
    risk_neutral::Bool=true,
    systematic::Bool=true,
    idiosyncratic::Bool=false,
#Parameters
    α::Float64=0.3, #Elasticity of capital
    r::Float64=0.01, #Risk free rate
    f::Float64=0.0, #Fixed cost
    τ::Float64=0.3, # Tax rate
    δ::Float64=0.01, # Depreciation rate
    x_bar::Float64=2.0, #Unconditional mean of the systematic shock
    ρ_x::Float64=0.9, #Persistence of Systematic Shock
    σ_x::Float64=0.1, #Volatility of systematic shock innovations
    ρ_z::Float64=0.99, #Persistence of idiosyncratic shock
    σ_z::Float64=0.1, #Volatility of idiosyncratic shock innovations
    λ::Float64=0.001, #Equity issuance cost

# Sizes
    n_k::Int64=30,
    n_x::Int64=10,
#Adjustment costs
    η::Float64=1.0, # Map i×k -> real)
    ) # End of arguments
# Here we create the matrices with the value function and policy function
    V::Matrix=zeros(n_k, n_x)
    i_opt::Matrix=zeros(n_k, n_x)
    mc = tauchen(n_x, ρ_x, σ_x, x_bar);
    Π_x=mc.p;
    x_grid = collect(range(x_bar-3*σ_x/sqrt(1-ρ_x^2), x_bar+3*σ_x/sqrt(1-ρ_x^2), length = n_x));
    k_grid = collect(range(1.0, 400.0, length=n_k))

    # Dividend payout in the simplest case
    Ψ(i,k)=k*0.5*η*(i/k)^2
    e(x, k, i)=(1-τ)*(exp(x)*k^α-f-δ*k)-i-Ψ(i,k)
    function d(x, k, i)
        earnings=e(x,k,i)
        return earnings+λ*abs(earnings)*(earnings<0)
    end

    function TV!(V, i_opt)
        V_func=interpolate((k_grid, x_grid), V, Gridded(Linear()));
        V_func=extrapolate(V_func, (Interpolations.Flat(),  Interpolations.Flat()));
        for (i_k, k) in enumerate(k_grid)
            for (i_x, x) in enumerate(x_grid)
                obj(i)=-d(x, k, i)-(1.0/(1+r))*sum(Π_x[i_x, j]*V_func(k*(1-δ)+i, x_grid[j]) for j in 1:n_x)
                # Here we get the optimal one, we can use one dimensional optimization
                res=Optim.optimize(obj, -maximum(k_grid), maximum(k_grid))
                V[i_k, i_x]=-res.minimum
                i_opt[i_k, i_x]=res.minimizer
            end
        end
    end


    function solve_VFI(V, i_opt; max_iter::Int64=1000, max_tol::Float64=1e-6)
        it=1
        tol=Inf
        while tol>max_tol && it<=max_iter
            V_old=copy(V)
            TV!(V, i_opt)
            tol=maximum(abs(x - y) for (x, y) in zip(V_old, V))
            it=it+1
        end
        return V, i_opt
    end

    function simulate_model(V_opt::Matrix, i_opt::Matrix, T::Int64=100)
        #Solves the model
        V_func=interpolate((k_grid, x_grid), V, Gridded(Linear()));
        V_func=extrapolate(V_func, (Interpolations.Flat(),  Interpolations.Flat()));

        I_func=interpolate((k_grid, x_grid), i_opt, Gridded(Linear()));
        I_func=extrapolate(I_func, (Interpolations.Flat(),  Interpolations.Flat()));

        x_init = mean(x_grid) # Initial x
        k_init = mean(k_grid)
        x_init_ind = searchsortedfirst(x_grid, x_init)

        k_sim=zeros(T+1,1)
        k_sim[1]=k_init
        i_sim=zeros(T+1,1)
        d_sim=zeros(T+1,1)
        mc_x = MarkovChain(Π_x)
        x_sim_indices = simulate(mc_x, T+1; init = x_init_ind)
        x_sim_val = zeros(T)
        #Pre return
        r_sim=ones(T+1,1)

        for t in 1:T
            current_x=x_grid[x_sim_indices[t]]
            current_k=k_sim[t]
            k_sim[t+1]=(1-δ)*current_k+I_func(current_k, current_x)
            i_sim[t]=I_func(current_k, current_x)
            d_sim[t]=d(current_x, current_k, i_sim[t])
            r_sim[t+1]=V_func(k_sim[t+1], x_grid[x_sim_indices[t+1]])/(V_func(k_sim[t], x_grid[x_sim_indices[t]])+d_sim[t])
        end

        # Here we plot the outcome of the model

        pygui(true)
        subplot(511)
        PyPlot.plot(1:T, x_grid[x_sim_indices][1:T])
        PyPlot.ylim(x_grid[1],x_grid[n_x])
        xlabel("Time")
        ylabel("X")
        subplot(512)
        PyPlot.plot(1:T, k_sim[1:T])
        PyPlot.ylim(minimum(k_sim),maximum(k_sim))
        xlabel("Time")
        ylabel("K")
        subplot(513)
        PyPlot.plot(1:T, i_sim[1:T])
        PyPlot.ylim(minimum(i_sim),maximum(i_sim))
        xlabel("Time")
        ylabel("I")
        subplot(514)
        PyPlot.plot(1:T, d_sim[1:T])
        PyPlot.ylim(minimum(d_sim),maximum(d_sim))
        xlabel("Time")
        ylabel("D")
        subplot(515)
        PyPlot.plot(1:T, r_sim[1:T])
        PyPlot.ylim(minimum(r_sim),maximum(r_sim))
        xlabel("Time")
        ylabel("R")
        show()

    end

    return (solve = solve_VFI, simulate_model=simulate_model, V=V, i_opt=i_opt)

end

neoclassical=NeoClassical()

solve_model=neoclassical.solve
simulate_model=neoclassical.simulate_model
V=neoclassical.V
i_opt=neoclassical.i_opt

V_opt, i_opt=solve_model(V, i_opt)

simulate_model(V, i_opt)





#end # module
