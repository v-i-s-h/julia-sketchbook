# Training simple LSTM

using Lux
using MLUtils, Optimisers, Zygote, NNlib, Random, Statistics

##
function get_dataloaders(; dataset_size=1000, sequence_length=50)
    # Create spirals
    data = [MLUtils.Datasets.make_spiral(sequence_length) for _ in 1:dataset_size]
    # Get labels
    labels = vcat(
        repeat([0.0f0], dataset_size ÷ 2),
        repeat([1.0f0], dataset_size ÷ 2)
    )
    clockwise_spirals = [
        reshape(d[1][:, 1:sequence_length], :, sequence_length, 1)
        for d in data[1:(dataset_size÷2)]
    ]
    anticlockwise_spirals = [
        reshape(d[1][:, (sequence_length+1):end], :, sequence_length, 1)
        for d in data[((dataset_size÷2)+1):end]
    ]
    x_data = Float32.(cat(clockwise_spirals..., anticlockwise_spirals...; dims=3))

    (x_train, y_train), (x_val, y_val) = splitobs((x_data, labels); at=0.80, shuffle=true)

    return (
        DataLoader(collect.((x_train, y_train)); batchsize=128, shuffle=true),
        DataLoader(collect.((x_val, y_val)); batchsize=128, shuffle=false)
    )
end

##
struct SpiralClassifier{L, C} <:
        Lux.AbstractExplicitContainerLayer{(:lstm_cell, :classifier)}
    lstm_cell::L
    classifier::C
end

function  SpiralClassifier(in_dims, hidden_dims, out_dims)
    return SpiralClassifier(
        LSTMCell(in_dims => hidden_dims),
        Dense(hidden_dims => out_dims, sigmoid)
    )
end

function (s::SpiralClassifier)(x::AbstractArray{T, 3}, 
                                ps::NamedTuple,
                                st::NamedTuple) where {T}
    x_init, x_rest = Iterators.peel(eachslice(x; dims=2))
    (y, carry), st_lstm = s.lstm_cell(x_init, ps.lstm_cell, st.lstm_cell)
    for x ∈ x_rest
        (y, carry), st_lstm = s.lstm_cell((x, carry), ps.lstm_cell, st_lstm)
    end
    y, st_classifier = s.classifier(y, ps.classifier, st.classifier)
    st = merge(st, (classifier=st_classifier, lstm_cell=st_lstm))
    return vec(y), st
end

##
function xlogy(x, y)
    result = x * log(y)
    return ifelse(iszero(x), zero(result), result)
end

function binarycrossentropy(y_pred, y_true)
    y_pred = y_pred .+ eps(eltype(y_pred))
    return mean(@. -xlogy(y_true, y_pred) - xlogy(1 - y_true, 1 - y_pred))
end

function compute_loss(x, y, model, ps, st)
    y_pred, st = model(x, ps, st)
    return binarycrossentropy(y_pred, y), y_pred, st
end

matches(y_pred, y_true) = sum((y_pred .> 0.50) .== y_true)
accuracy(y_pred, y_true) = matches(y_pred, y_true) / length(y_pred)

function create_optimiser(ps)
    opt = Optimisers.Adam(0.01f0)
    return Optimisers.setup(opt, ps)
end

##
function main()
    (train_loader, val_loader) = get_dataloaders()

    model = SpiralClassifier(2, 8, 1)
    rng = Random.default_rng()
    Random.seed!(rng, 0)
    ps, st = Lux.setup(rng, model)

    opt_state = create_optimiser(ps)

    for epoch in 1:25
        for (x, y) ∈ train_loader
            (loss, pred, st), back = pullback(
                    p -> compute_loss(x, y, model, p, st),
                    ps
                )
            gs = back((one(loss), nothing, nothing))[1]
            opt_state, ps = Optimisers.update(opt_state, ps, gs)

            println("Epoch [$epoch]: Loss $loss")
        end

        st_ = Lux.testmode(st)
        for (x, y) in val_loader
            (loss, y_pred, st_) = compute_loss(x, y, model, ps, st_)
            acc = accuracy(y_pred, y)
            println("Validation: Loss $loss    Accuracy: $acc")
        end
    end
end

##
main()


