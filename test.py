from barrier_func_method import barrier_method
from main import f, g

import openpyxl


def f_range(start, stop, step):
    while start <= stop:
        yield start
        start += step


def test(sheet):
    x0 = [0.3, 0.5, 0.1, 0.2, 0.2]
    t0 = 2
    gamma = 10

    row, col = 1, 1
    sheet.cell(row=row, column=col).value = 'set of starting points'
    row += 1
    for i, xi in enumerate(x0):
        print(f'x{i + 1} is fixed' + 100 * '*' + '\n')
        sheet.cell(row=row, column=col).value = f'x{i + 1} variable is fixed'
        sheet.cell(row=row, column=2).value = 'Number of iterations'
        row += 1
        for j in f_range(-10, 10, 0.1):
            x0[i] = j
            if g.check_point_strict(x0):
                print(f'{x0} {g.get_func()(x0)} \n')
                xk, k = barrier_method(f, g, x0, t0, gamma)
                if xk != [0, 0, 0, 0, 0]:
                    sheet.cell(row=row, column=col).value = f'({x0[0]:.1f}, {x0[1]:.1f}, {x0[2]:.1f}, {x0[3]:.1f}, {x0[4]:.1f})'
                    sheet.cell(row=row, column=2).value = k
                    row += 1
        x0[i] = xi


def main():
    wb = openpyxl.Workbook()
    sheet = wb.active
    test(sheet)
    wb.save(filename="excel.xlsx")
    return 0


if __name__ == '__main__':
    main()
