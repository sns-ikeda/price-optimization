using Random, LinearAlgebra
using Pkg
Pkg.add("PyCall")

function calc_z(x, a, b, phi, g, M, K, D, TL)
    z, mt_z = Dict(), Dict()
    mk_x = Dict([(m, k) for m in M for k in K if get(x, (m, k), 0) >= 0.99])
    @assert length(mk_x) == length(M)

    for m in M
        t = 0
        while true
            linear_sum = sum(get(a, (m, mp, t), 0) * get(phi, (mp, get(mk_x, mp, 0)), 0) for mp in M)
            # + sum(get(a, (m, d, t), 0) * get(g, (m, d), 0) for d in get(D, m, []), init=0)

            if linear_sum < get(b, (m, t), 0)
                t = t * 2 + 1
            else
                t = t * 2 + 2
            end
            if t in get(TL, m, [])
                break
            end
            if t > 1000
                error("Infinite Loop Error")
            end
        end
        mt_z[m] = t
        z = merge(z, Dict((m, t) => if t == get(mt_z, m, 0) 1 else 0 end for t in get(TL, m, [])))
    end
    return z
end

function calc_obj(x, z, M, K, TL, P, beta, beta0, phi, g, D)
    mk_x = Dict([(m, k) for m in M for k in K if get(x, (m, k), 0) >= 0.99])
    mt_z = Dict([(m, t) for m in M for t in get(TL, m, []) if get(z, (m, t), 0) >= 0.99])
    @assert length(mt_z) == length(mk_x) == length(M)
    @assert keys(mk_x) == keys(mt_z)

    p = [get(P, (m, get(mk_x, m, 0)), 0) for m in keys(mk_x)]
    q = [
        sum(get(beta, (m, mp, t), 0) * get(phi, (mp, get(mk_x, mp, 0)), 0) for mp in keys(mt_z)) +
        # sum(get(beta, (m, d, t), 0) * get(g, (m, d), 0) for d in get(D, m, [])) +
        get(beta0, (m, t), 0)
        for (m, t) in pairs(mt_z)
    ]

    return dot(p, q)
end

function get_opt_prices(x, P)
    opt_prices = Dict([(k_tuple[1], get(P, k_tuple, 0)) for k_tuple in sort(collect(keys(x))) if round(Int, get(x, k_tuple, 0)) == 1])
    return opt_prices
end

function coordinate_descent(;M, K, P, x_init, a, b, phi, g, D, TL, beta, beta0, threshold=10)
    z_init = calc_z(x_init, a, b, phi, g, M, K, D, TL)
    best_obj = calc_obj(x_init, z_init, M, K, TL, P, beta, beta0, phi, g, D)
    best_x = deepcopy(x_init)

    break_count, total_count = 0, 0
    while true
        total_count += 1
        m = rand(M)
        x_m = zeros(length(K))
        objs = []
        for k in K
            x_m[k+1] = 1
            x = deepcopy(best_x)
            x = merge(x, Dict((m, k) => x_m[k+1] for k in K))
            z = calc_z(x, a, b, phi, g, M, K, D, TL)
            obj = calc_obj(x, z, M, K, TL, P, beta, beta0, phi, g, D)
            push!(objs, obj)
            if obj > best_obj
                best_obj = obj
                best_x = deepcopy(x)
                break_count = 0
            end
        end
        if best_obj >= maximum(objs)
            break_count += 1
        end

        if break_count >= threshold
            break
        end
        # @info "product: $m, best_obj_m: $best_obj"
    end
    opt_prices = get_opt_prices(best_x, P)
    return best_obj, opt_prices
end
