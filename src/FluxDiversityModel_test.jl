export FluxPartModel_Q10_test

struct FluxPartModel_Q10_test <: EasyHybridModels
    CUE_chain::Flux.Chain
    CUE_predictors::AbstractArray{Symbol}

    forcing::AbstractArray{Symbol}

    gamma0
    k0
end

"""
FluxPartModel_Q10_test(RUE_predictors; gamma0=[0.5f0],k0=[0.5f0])
"""
function FluxPartModel_Q10_test(
    CUE_predictors::AbstractArray{Symbol}; 
    forcing=[:SW_IN, :TA], 
    gamma0=[0.5f0],
    k0=[0.5f0], 
    neurons=15)

    CUE_ch = DenseNN(length(CUE_predictors), length(CUE_predictors), neurons; activation=Flux.softplus)
    FluxPartModel_Q10_test(
        CUE_ch,
        CUE_predictors,
        forcing,
        gamma0,
        k0
    )
end

function (m::FluxPartModel_Q10_test)(dk, ::Val{:infer})
    CUE_input4Chain = select_predictors(dk, m.CUE_predictors)
    CUE = 1.0f0 * m.CUE_chain(CUE_input4Chain)

    Cs_in = select_variable(dk, m.forcing[1])
    Cb_in = select_variable(dk, m.forcing[2])

    Resp = (1.0f0-CUE) .* m.k0[1] .* Cs_in .* Cb_in .^ m.gamma0[1]
    #MinN = m.k0[1] .* Cs_in .* Cb_in .^ m.gamma0[1] .* (1.0f0/m.CN_s[1]-CUE/m.CN_b[1])

    return (; CUE, Resp=Resp)
end

function (m::FluxPartModel_Q10_test)(dk)
    res = m(dk, :infer)
    return res.Resp
end

"""
(m::`FluxPartModel_Q10_test`)(dk, infer::Symbol)
"""
function (m::FluxPartModel_Q10_test)(dk, infer::Symbol)
    return m(dk, Val(infer))
end

# Call @functor to allow for training the custom model
Flux.@functor FluxPartModel_Q10_test