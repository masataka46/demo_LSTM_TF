import numpy as np

# def make_sin_data(data_per_cycle=200, n_cycle=5, train_ratio=0.8):
#     np.random.seed(0)
#
#     n_data = n_cycle * data_per_cycle
#     theta = np.linspace(0., n_cycle * (2. * np.pi), num=n_data)
#
#     X = np.sin(theta) + 0.1 * (2. * np.random.rand(n_data) - 1.)
#     X /= np.std(X)
#     X = X.astype(np.float32)
#
#     n_train = int(n_data * train_ratio)
#     X_train, X_test = X[:n_train], X[n_train:]
#
#     return X_train, X_test

def make_sin_data2(data_per_cycle=200, n_cycle=5, train_ratio=0.8):
    np.random.seed(0)

    n_data = n_cycle * data_per_cycle
    theta = np.linspace(0., n_cycle * (2. * np.pi), num=n_data)

    sin_x = np.sin(theta)
    X = sin_x + 0.1 * (2. * np.random.rand(n_data) - 1.)
    X /= np.std(X)
    X = X.astype(np.float32)

    sin_x /= np.std(X)
    sin_x = sin_x.astype(np.float32)

    n_train = int(n_data * train_ratio)
    X_train, X_test = X[:n_train], X[n_train:]

    sin_x_train, sin_x_test = sin_x[:n_train], sin_x[n_train:]

    return X_train, X_test, sin_x_train, sin_x_test

if __name__ == '__main__':
    X_train, X_test, sin_x_train, sin_x_test = make_sin_data2()
    #print(X_train)
    print("X_train.shape", X_train.shape)
    print("np.max(X_train)", np.max(X_train))
    print("np.min(X_train)", np.min(X_train))
    print("np.mean(X_train)", np.mean(X_train))

    print("sin_x_test, ", sin_x_test)
