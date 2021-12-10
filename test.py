from barrier_func_method import barrier_method
from main import f, g

import openpyxl
from decimal import *

getcontext().prec = 4


def f_range(start: Decimal, stop: Decimal, step: Decimal):
    while start <= stop:
        yield Decimal(start)
        start += step


def test(sheet):
    x0 = [Decimal(0.1), Decimal(0.5), Decimal(0.1), Decimal(0.2), Decimal(0.2)]
    r0 = Decimal(100)
    c = Decimal(0.7)

    row, col = 1, 1
    sheet.cell(row=row, column=col).value = 'set of starting points'
    row += 1
    for i, xi in enumerate(x0):
        sheet.cell(row=row, column=col).value = f'x{i + 1} variable is fixed'
        sheet.cell(row=row, column=2).value = 'Number of iterations'
        row += 1
        for j in f_range(Decimal(-10), Decimal(10), Decimal(0.1)):
            x0[i] = j
            if g.check_point_strict(x0):
                xk, k = barrier_method(f, g, x0, r0, c)
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
