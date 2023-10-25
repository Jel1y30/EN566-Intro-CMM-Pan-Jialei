import matplotlib.pyplot as plt
import argparse

def plot_functions(functions, x, y_values):
    for idx, func in enumerate(functions):
        plt.plot(x, y_values[idx], label=func)
    plt.xlabel('x')
    plt.ylabel('Function value')
    plt.title(', '.join(functions).capitalize())
    plt.legend()
    plt.grid(True)
    plt.show()

def read_from_file(filename):
    with open(filename, 'r') as f:
        x = []
        y_values = []
        for line in f.readlines():
            values = list(map(float, line.strip().split()))
            x.append(values[0])
            y_values.append(values[1:])
    return x, y_values

def main():
    parser = argparse.ArgumentParser(description="Trigonometric function plotter from file.")
    parser.add_argument('--read_from_file', type=str, required=True, help='File to read ASCII table from')
    parser.add_argument('--function', type=str, required=True, help='Function(s) to plot. Available: cos,sin,sinc')
    args = parser.parse_args()

    x, y_values = read_from_file(args.read_from_file)
    functions = [f.strip() for f in args.function.split(',')]
    
    plot_functions(functions, x, y_values)

if __name__ == '__main__':
    main()

