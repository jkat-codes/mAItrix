## test forward prop
def test_forward_prop():
    layers = [X_train.shape[0], 100, 200, Y_train.shape[0]]
    params = initialize_params(layers)
    aL, forward_cache = forward_prop(X_train, params, "relu")

    for l in range(len(params) // 2 + 1):
        print("Shape of A" + str(l) + " :", forward_cache["A" + str(l)].shape)


## test back prop and observe shapes
def test_back_prop():
    layers = [X_train.shape[0], 100, 200, Y_train.shape[0]]
    params = initialize_params(layers)
    aL, forward_cache = forward_prop(X_train, params, "relu")

    for l in range(len(params) // 2 + 1):
        print("Shape of A" + str(l) + " :", forward_cache["A" + str(l)].shape)

    grads = back_prop(
        forward_cache["A" + str(3)], Y_train, params, forward_cache, "relu"
    )

    print("\n")

    for l in reversed(range(1, len(grads) // 3 + 1)):
        print("Shape of dZ" + str(l) + " :", grads["dZ" + str(l)].shape)
        print("Shape of dW" + str(l) + " :", grads["dW" + str(l)].shape)
        print("Shape of dB" + str(l) + " :", grads["db" + str(l)].shape, "\n")
