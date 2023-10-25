import numpy as np
import argparse

def write_to_file(filename, x, y_values):
    with open(filename, 'w') as f:
        for i in range(len(x)):
            f.write(f"{x[i]:.2f} " + ' '.join([f"{y[i]:.5f}" for y in y_values]) + "\n")

def main():
    parser = argparse.ArgumentParser(description="Trigonometric function data writer.")
    parser.add_argument('--function', type=str, required=True, help='Function(s) to write. Available: cos,sin,sinc')
    parser.add_argument('--write', type=str, required=True, help='File to write ASCII table')
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

    write_to_file(args.write, x, y_values)

if __name__ == '__main__':
    main()
