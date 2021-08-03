import math
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
from tabulate import tabulate
from termcolor import cprint

### Основные функции:###
def lagrange(array_x, array_y, cur_x):
    lag = 0
    lagrangians = []
    for j in range(len(array_y)):
        multiplying = 1
        for i in range(len(array_x)):
            if i != j:
                multiplying *= (cur_x - array_x[i]) / (array_x[j] - array_x[i])
        lagrangians.append(array_y[j] * multiplying)
        lag += array_y[j] * multiplying
    return lag, lagrangians




def interpolateForward(t, differencesByOrder, order, previous):
    valueY = differencesByOrder[0][previous]
    for i in range(1, order):
        valueY += ((t / math.factorial(i)) * differencesByOrder[i][previous])
        t *= (t - i)
    return valueY


def interpolateBackward(t, differencesByOrder, order):
    valueY = differencesByOrder[0][order]
    for i in range(1, order):
        valueY += (t / math.factorial(i)) * differencesByOrder[i][order - i]
        t *= (t + i)
    return valueY


def newton_polynomial(array_x, array_y, cur_x):
    m = len(array_x)
    last = 0
    for i in range(len(array_x) - 1):
        if (cur_x >= array_x[i] and array_x[i + 1] >= cur_x):
            last = i
            break
    if (cur_x >= array_x[-1]):
        last = len(array_x) - 1
    table_dif = list()
    order = 0
    y_i = list()
    for i in range(len(array_y)):
        y_i.append(array_y[i])
    table_dif.append(y_i)
    while order < len(array_y) - 1:
        order += 1
        delta_Y = list()
        previous_delta_Y = table_dif[order - 1]
        for i in range(len((previous_delta_Y)) - 1):
            delta_Y.append(previous_delta_Y[i + 1] - previous_delta_Y[i])
        table_dif.append(delta_Y)
    if (last < len(array_x) / 2):
        t = (cur_x - array_x[last]) / (array_x[1] - array_x[0])

        return interpolateForward(t, table_dif, (len(array_y) - last - 1), last), None
    else:
        if (last == (len(array_x) - 1)):
            next = last
        else:
            next = last + 1
        t = (cur_x - array_x[next]) / (array_x[1] - array_x[0])
        return interpolateBackward(t, table_dif, next), None

def check_and_draw(x, y, approximate_function, title, point):
    fig, ax = plt.subplots()
    xnew = np.linspace(np.min(x), np.max(x), 100)
    ynew = [approximate_function(x, y, i)[0] for i in xnew]
    plt.plot(x, y, '.', color='blue', label='узлы инторполяции')
    plt.plot(xnew, ynew, color='green', label='y = F(x)')
    plt.plot(point[0], point[1], 'o', color='red', markersize=12, label='результат инторполяции')
    plt.title(title)
    ax.legend()
    plt.grid(True)
    plt.show()


### второстепенные:###
def print_matrix(input_matrix):
    cprint(tabulate(input_matrix,
                    tablefmt="grid", floatfmt="2.5f"))


def write_number(s='', integer=False, check=None):
    if check is None:
        check = [False, 0, 0]
    flag = True
    while flag:
        flag = False
        try:
            if integer:
                val = int(input(s))
            else:
                val = float(input(s))
            if check[0] and (val < check[1] or val > check[2]):
                raise ValueError
        except ValueError:
            flag = True
            if check[0]:
                cprint(
                    f'\nВведите значение точки интерполяции, что лежит между узлами интерполяции : [{check[1]}; {check[2]}]\n',
                    attrs=['bold'])
    return val


def parse():
    a = genfromtxt('test.csv', delimiter=',')
    if True in np.isnan(a) or a.shape[0] != 2:
        raise ValueError
    return a


def equation():
    while 1:
        cprint("\nВыберете функцию:\n"
               "\t1. sin(x)\n"
               "\t2. sqrt(x)\n"
               "\t3. e^x\n", attrs=['bold'])
        method = int(input().strip())
        cprint("\nВведите границы интерполяции через пробел - на интервале [-10;10]:\n", attrs=['bold'])
        borders = list(input().strip().split(" "))

        if len(borders) == 2 and (float(borders[0].strip()) < float(borders[1].strip())):
            a = float(borders[0].strip())
            b = float(borders[1].strip())
            data = []
            cprint("\nВведите количество интерполяционных узлов (от 1 до 100):\n", attrs=['bold'])
            number_of_data = write_number('', integer=True, check=[False, 0, 100])
            x = np.linspace(a, b, number_of_data)
            if method == 1:
                for i in range(number_of_data):
                    data.append([x[i], math.sin(x[i])])
            elif method == 2:
                for i in range(number_of_data):
                    data.append([x[i], math.sqrt(x[i])])
            elif method == 3:
                for i in range(number_of_data):
                    data.append([x[i], math.pow(math.e, x[i])])
            break
        else:
            cprint('Нижняя граница должна быть меньше верхней!')
            continue
    return np.array(data).transpose()


def write_values():
    print()
    n = write_number(s='Введите количество точек: ', integer=True)
    print()
    a = []
    for i in range(int(n)):
        a.append([write_number('X='), write_number('Y=')])
        print()
    return np.array(a).transpose()


def main_func():
    again = True
    while again:
        again = False
        cprint(
            'Интерполяцию будем производить по:\n\t0 - по данным из файла (таблица x,y)\n\t1 - по данным с клавиатуры (таблица x,y)\n\t2 -  выбранной функции\n',
            attrs=['bold'])
        chosen = input()
        if chosen.strip() == '0':
            data = parse()
        elif chosen.strip() == '1':
            data = write_values()
        elif chosen.strip() == '2':
            data = equation()
        else:
            print('Некорректный ответ !\n')
            again = True
    cprint('\nУзлы интерполяции ( верхняя строка - Y, нижняя - соответствующие X ):', attrs=['bold'])
    print_matrix(data)

    cprint('\nВведите значение точки интерполяции: \n', attrs=['bold'])
    cur_x = write_number('', check=[True, min(data[0]), max(data[0])])

    lag, lagrangians = lagrange(data[0], data[1], cur_x)
    cprint('\nИнтерполяция посредством многочлена Лагранжа:', attrs=['bold'])
    cprint(F' Ответ = {lag}')
    check_and_draw(data[0], data[1], lagrange, 'Многочлен Лагранжа', [cur_x, lag])
    print()

    newtone_answer = newton_polynomial(data[0], data[1], cur_x)[0]
    cprint(f'\nИнтерполяция посредством многочлена Ньютона:', attrs=['bold'])
    cprint(f'Ответ = {newtone_answer}')
    check_and_draw(data[0], data[1], newton_polynomial, 'Многочлен Ньютона', [cur_x, newtone_answer])


main_func()
