import numpy as np
import matplotlib.pyplot as plt
from make_sin_data import make_sin_data2

def draw_graph(y_probs, sin_x_test, data_num=20, sequence_len=100):
    print("np.max(y_probs), ", np.max(y_probs))
    print("np.max(sin_x_test), ", np.max(sin_x_test))
    print("y_probs.shape", y_probs.shape)
    print("sin_x_test.shape", sin_x_test.shape)
    sin_x_test_list = []
    for i in range(len(y_probs)):
        sin_x_test_list.append(sin_x_test[i:i + sequence_len])

    sin_x_test_np = np.array(sin_x_test_list, dtype=np.float32)
    print("sin_x_test_np.shape", sin_x_test_np.shape)
    sin_x_test_np_1_data = sin_x_test_np[data_num]
    print("sin_x_test_np_1_data.shape", sin_x_test_np_1_data.shape)
    y_probs_1_data = y_probs[data_num + 1]
    print("y_probs_1_data.shape", y_probs_1_data.shape)

    sin_x_y_probs = np.concatenate((sin_x_test_np_1_data[:50], y_probs_1_data))
    print("sin_x_y_probs.shape", sin_x_y_probs.shape)
    x_cord = np.arange(data_num * 0.01, 1 + data_num * 0.01, 0.01)
    print("x_cord.shape", x_cord.shape)
    plt.plot(x_cord, sin_x_test_np_1_data, linestyle="solid")
    plt.plot(x_cord, sin_x_y_probs, linestyle="dashed")
    plt.show()

if __name__ == '__main__':
    X_train, X_test, sin_x_train, sin_x_test = make_sin_data2()
    prob_list = []
    print("np.max(sin_x_test), ", np.max(sin_x_test))
    for i in range(100):
        prob_list.append(X_test[i + 50:i + 100])

    prob_np = np.array(prob_list, dtype=np.float32)

    draw_graph(prob_np, sin_x_test, data_num=0)

