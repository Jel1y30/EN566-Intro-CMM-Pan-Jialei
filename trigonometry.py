import numpy as np
import matplotlib.pyplot as plt
import argparse

def compute_functions(values, functions):
    data = {'x': values}
    if 'cos' in functions:
        data['cos'] = np.cos(values)
    if 'sin' in functions:
        data['sin'] = np.sin(values)
    if 'sinc' in functions:
        data['sinc'] = np.sinc(values)
    return data

def plot_functions(data):
    plt.figure()
    for key, val in data.items():
        if key != 'x':
            plt.plot(data['x'], val, label=key)
    plt.title("Trigonometric Functions")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()

def write_to_file(filename, data):
    with open(filename, 'w') as f:
        f.write(','.join(data.keys()) + '\n')
        for i in range(len(data['x'])):
            f.write(','.join(str(data[key][i]) for key in data.keys()) + '\n')

def read_from_file(filename):
    with open(filename, 'r') as f:
        keys = f.readline().strip().split(',')
        data = {key: [] for key in keys}
        for line in f:
            for key, val in zip(keys, line.strip().split(',')):
                data[key].append(float(val))
    return data

def save_plot(data, format):
    plt.figure()
    for key, val in data.items():
        if key != 'x':
            plt.plot(data['x'], val, label=key)
    plt.title("Trigonometric Functions")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"plot.{format}", format=format)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot and manipulate trigonometric functions.")
    parser.add_argument('--function', required=True, action='append', choices=['sin', 'cos', 'sinc'])
    parser.add_argument('--write', type=str, help="Filename to write the data.")
    parser.add_argument('--read_from_file', type=str, help="Filename to read the data from.")
    parser.add_argument('--print', type=str, choices=['jpeg', 'eps', 'pdf'], help="Save the plot in the specified format.")

    args = parser.parse_args()

    if args.read_from_file:
        data = read_from_file(args.read_from_file)
    else:
        values = np.arange(-10, 10.05, 0.05)
        data = compute_functions(values, args.function)

    if args.write:
        write_to_file(args.write, data)

    if args.print:
        save_plot(data, args.print)
    else:
        plot_functions(data)
