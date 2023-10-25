import numpy as np
import matplotlib.pyplot as plt
import argparse

def plot_functions(functions, x, y_values, formats):
    for idx, func in enumerate(functions):
        plt.plot(x, y_values[idx], label=func)
    plt.xlabel('x')
    plt.ylabel('Function value')
    plt.title(', '.join(functions).capitalize())
    plt.legend()
    plt.grid(True)
    for fmt in formats:
        plt.savefig(f"plot.{fmt}", format=fmt)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Trigonometric function plotter with save option.")
    parser.add_argument('--function', type=str, required=True, help='Function(s) to plot. Available: cos,sin,sinc')
    parser.add_argument('--print', type=str, required=True, help='Save plot to file. Format: jpeg,eps,pdf')
    args = parser.parse_args()

    x = np.arange(-10, 10.05, 0.05)
    functions = [f.strip() for f in args.function.split(',')]
    y_values = []
    
    for func in functions:
        if func == 'cos':
            y_values.append(np.cos(x))
        elif func == 'sin':
            y_values.append(np.sin(x))
        elif func == 'sinc':
            y_values.append(np.sinc(x))
        else:
            print(f"Function {func} not recognized!")
            return
    
    formats = [f.strip() for f in args.print.split(',')]
    plot_functions(functions, x, y_values, formats)

if __name__ == '__main__':
    main()

