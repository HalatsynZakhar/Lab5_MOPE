from math import sqrt
from random import randint
import numpy as np
import copy
import scipy.stats


def r(x: float) -> float:
    """Точність округлення"""
    x = round(x, 4)
    if float(x) == int(x):
        return int(x)
    else:
        return x


def par(a: float) -> str:
    """Для вивіду. Негативні числа закидає в скобки и округлює"""
    if a < 0:
        return "(" + str(r(a)) + ")"
    else:
        return str(r(a))


def average(list: list, name: str) -> int or float:
    """Середнє значення з форматованним вивідом для будь-якого листа"""
    print("{} = ( ".format(name), end="")
    average = 0
    for i in range(len(list)):
        average += list[i]
        if i == 0:
            print(r(list[i]), end="")
        else:
            print(" + ", end="")
            print(par(list[i]), end="")
    average /= len(list)
    print(" ) / {} = {} ".format(len(list), r(average)))
    return average


def printf(name: str, value: int or float):
    """Форматованний вивід змінної з округленням"""
    print("{} = {}".format(name, r(value)))


def matrixplan(xn_factor: list, x_min: int, x_max: int) -> list:
    """Заповнює матрицю планування згідно нормованої"""
    xn_factor_experiment = []
    for i in range(len(xn_factor)):
        if xn_factor[i] == -1:
            xn_factor_experiment.append(x_min)
        elif xn_factor[i] == 1:
            xn_factor_experiment.append(x_max)
    return xn_factor_experiment


def MatrixExper(x_norm: list, x_min: list, x_max: list) -> list:
    """Генеруємо матрицю планування згідно нормованної"""
    x_factor_experiment = []
    for i in range(len(x_norm)):
        x_factor_experiment.append(matrixplan(x_norm[i], x_min[i], x_max[i]))
    return x_factor_experiment


def generate_y(y_min, y_max, n, m) -> list:
    """Генерує функції відгугу за вказанним діапозоном"""
    list = []
    for i in range(m):
        list.append([randint(y_min, y_max + 1) for i in range(n)])
    return list


def a_n_funct(xn_factor_experiment: list, y_average_list: list) -> list:
    """Рахує а1, а2, а3 з форматованним вивідом"""
    a_n = []
    for i in range(len(xn_factor_experiment)):
        a_n.append(0)
        print("a{} = ( ".format(i + 1), end="")
        for j in range(len(xn_factor_experiment[i])):
            a_n[i] += xn_factor_experiment[i][j] * y_average_list[j]
            if j == 0:
                print("{}*{}".format(r(xn_factor_experiment[i][j]), par(y_average_list[j])), end="")
            else:
                print(" + {}*{}".format(par(xn_factor_experiment[i][j]), par(y_average_list[j])), end="")
        a_n[i] /= len(xn_factor_experiment[i])
        print(" ) / {} = {} ".format(len(xn_factor_experiment[i]), r(a_n[i])))
    return a_n


def a_nn_funct(xn_factor_experiment: list) -> list:
    """Рахує а11, а22, а33 з форматованим вивідом"""
    a_nn = []
    for i in range(len(xn_factor_experiment)):
        a_nn.append(0)
        print("a{}{} = ( ".format(i + 1, i + 1), end="")
        for j in range(len(xn_factor_experiment[i])):
            a_nn[i] += xn_factor_experiment[i][j] ** 2
            if j == 0:
                print("{}^2".format(par(xn_factor_experiment[i][j])), end="")
            else:
                print(" + {}^2".format(par(xn_factor_experiment[i][j])), end="")
        a_nn[i] /= len(xn_factor_experiment[i])
        print(" ) / {} = {} ".format(len(xn_factor_experiment[i]), r(a_nn[i])))
    return a_nn


def a_mn_funct(x_factor_experiment: list) -> list:
    """Рахує a12, a21, a13, a31, a23, a32"""
    a_mn = []
    list_range = [[0, 1], [1, 2], [2, 0]]

    for i, j in list_range:
        a_mn.append(0)
        print("a{}{} = ( ".format(i + 1, j + 1), end="")
        for k in range(len(x_factor_experiment[i])):
            a_mn[i] += x_factor_experiment[i][k] * x_factor_experiment[j][k]
            if k == 0:
                print("{}*{}".format(r(x_factor_experiment[i][k]), par(x_factor_experiment[j][k])), end="")
            else:
                print(" + {}*{}".format(r(x_factor_experiment[i][k]), par(x_factor_experiment[j][k])), end="")
        a_mn[i] /= len(x_factor_experiment[i])
        print(" ) / {} = {} ".format(len(x_factor_experiment[i]), r(a_mn[i])))

    return a_mn


def dispers(y: list, y_average_list: list, m) -> list:
    """Рахує s2 для усіх рядків. Повертає масив значень"""
    s2_y_row = []

    for i in range(len(y_average_list)):
        s2_y_row.append(0)
        print("s2_y_row{} = ( ".format(i + 1), end="")
        for j in range(3):
            s2_y_row[i] += (y[j][i] - y_average_list[i]) ** 2
            if j == 0:
                print("({} - {})^2".format(r(y[j][i]), par(y_average_list[i])), end="")
            else:
                print(" + ({} - {})^2".format(r(y[j][i]), par(y_average_list[i])), end="")
        s2_y_row[i] /= m
        print(" ) / {} = {} ".format(m, r(s2_y_row[i])))

    return s2_y_row


def beta(x_norm: list, y_average_list: list) -> list:
    """Рахує Бета критерия Стюдента. Повертає масив значень"""
    beta_list = []

    for i in range(len(x_norm)):
        beta_list.append(0)
        print("Beta{} = ( ".format(i + 1), end="")
        for j in range(len(x_norm[i])):
            beta_list[i] += y_average_list[j] * x_norm[i][j]
            if j == 0:
                print("{}*{}".format(r(y_average_list[j]), par(x_norm[i][j])), end="")
            else:
                print(" + {}*{}".format(r(y_average_list[j]), par(x_norm[i][j])), end="")
        beta_list[i] /= len(x_norm[0])
        print(" ) / {} = {} ".format(len(x_norm[0]), r(beta_list[i])))

    return beta_list


def t(beta_list: list, s_BetaS) -> list:
    """Рахує t критерія Стюдента. Повертає масив значень"""
    t_list = []
    for i in range(len(beta_list)):
        t_list.append(abs(beta_list[i]) / s_BetaS)
        print("t{} = {}/{} = {}".format(i, r(abs(beta_list[i])), par(s_BetaS), par(t_list[i])))
    return t_list


def s2_od_func(y_average_list, y_average_row_Student, m, N, d):
    """Вираховує сігму в квадраті для критерія Фішера"""
    s2_od = 0
    print("s2_od = ( ", end="")
    for i in range(len(y_average_list)):
        s2_od += (y_average_row_Student[i] - y_average_list[i]) ** 2
        if i == 0:
            print("({} - {})^2".format(r(y_average_row_Student[i]), par(y_average_list[i])), end="")
        else:
            print(" + ({} - {})^2".format(r(y_average_row_Student[i]), par(y_average_list[i])), end="")
    s2_od *= m / (N - d)
    print(" ) * {}/({} - {}) = {} ".format(m, N, d, r(s2_od)))
    return s2_od


x_min = [-4, -6, -7]  # Задані за умовою значення. Варіант 206
x_max = [4, 7, 10]

x_average_min = average(x_min, "X_average_min")  # Середнє Х макс и мин
x_average_max = average(x_max, "X_average_max")

m = 3  # За замовчуванням
q = 0.05  # рівень значимості
y_max = round(200 + x_average_max)  # Максимальні і мінімальні значення для генерації функції відгуку
printf("Y_max", y_max)
y_min = round(200 + x_average_min)
printf("Y_min", y_min)

l = 1.215

x_0_i = [(x_min[i] + x_max[i]) / 2 for i in range(len(x_min))]
delta_x_i = [(x_max[i] - x_0_i[i]) for i in range(len(x_min))]

x_norm = [[-1, -1, -1, -1, +1, +1, +1, +1],
          [-1, -1, +1, +1, -1, -1, +1, +1],
          [-1, +1, -1, +1, -1, +1, -1, +1]]

x_factor_experiment = MatrixExper(x_norm, x_min, x_max)

x_norm = [[+1, +1, +1, +1, +1, +1, +1, +1],  # додаємо перший стовпчик з одиницям
          [-1, -1, -1, -1, +1, +1, +1, +1],
          [-1, -1, +1, +1, -1, -1, +1, +1],
          [-1, +1, -1, +1, -1, +1, -1, +1]]

x_norm[0].extend([1 for i in range(7)])
x_norm[1].extend([-l, l, 0, 0, 0, 0, 0])  # там у дужкал ель (L), а не 1
x_norm[2].extend([0, 0, -l, l, 0, 0, 0])
x_norm[3].extend([0, 0, 0, 0, -l, l, 0])

x_norm.append([x_norm[1][i] * x_norm[2][i] for i in range(len(x_norm[0]))])  # додаємо ефект взаимодії і квадрат
x_norm.append([x_norm[1][i] * x_norm[3][i] for i in range(len(x_norm[0]))])
x_norm.append([x_norm[2][i] * x_norm[3][i] for i in range(len(x_norm[0]))])
x_norm.append([x_norm[1][i] * x_norm[2][i] * x_norm[3][i] for i in range(len(x_norm[0]))])
x_norm.append([x_norm[1][i] ** 2 for i in range(len(x_norm[0]))])
x_norm.append([x_norm[2][i] ** 2 for i in range(len(x_norm[0]))])
x_norm.append([x_norm[2][i] ** 2 for i in range(len(x_norm[0]))])

x_factor_experiment[0].extend([-l * delta_x_i[0], l * delta_x_i[0], x_0_i[0], x_0_i[0], x_0_i[0], x_0_i[0], x_0_i[0]])
x_factor_experiment[1].extend([x_0_i[1], x_0_i[1], -l * delta_x_i[1], l * delta_x_i[1], x_0_i[1], x_0_i[1], x_0_i[1]])
x_factor_experiment[2].extend([x_0_i[2], x_0_i[2], x_0_i[2], x_0_i[2], -l * delta_x_i[2], l * delta_x_i[2], x_0_i[2]])

N = len(x_factor_experiment[0])
count = 0
while True:  # Вихід тільки якщо задовольняються критерії
    count +=1

    y = generate_y(y_min, y_max, len(x_factor_experiment[0]), m)  # генеруємо значення функції відгуку

    y_average_list = []  # cереднє значення рядка Y
    for i in range(len(y[0])):
        y_average_list.append(
            average([y[j][i] for j in range(m)], "y_average_{}row".format(i + 1)))  # рахую середнє У у рядках

    y_average_average = average(y_average_list, "Y_average_average")  # середнє середніх значень Y

    x_exper_neight = []  # урахування еферкту взаємодії і квадратних членів. В массиві стовпці.
    for j in range(7):
        x_exper_neight.append([])
        for i in range(len(x_factor_experiment[0])):
            if j == 0:
                x_exper_neight[j].append(x_factor_experiment[0][i] * x_factor_experiment[1][i])  # x1x2
            if j == 1:
                x_exper_neight[j].append(x_factor_experiment[0][i] * x_factor_experiment[2][i])  # x1x3
            if j == 2:
                x_exper_neight[j].append(x_factor_experiment[1][i] * x_factor_experiment[2][i])  # x2x3
            if j == 3:
                x_exper_neight[j].append(
                    x_factor_experiment[0][i] * x_factor_experiment[1][i] * x_factor_experiment[2][i])  # x1x2x3
            if j == 4:
                x_exper_neight[j].append(x_factor_experiment[0][i] ** 2)  # x1**2
            if j == 5:
                x_exper_neight[j].append(x_factor_experiment[1][i] ** 2)  # x2**2
            if j == 6:
                x_exper_neight[j].append(x_factor_experiment[2][i] ** 2)  # x3**2

    x_i_list = []  # об ’эднанний лист факторів і ефекту взяємодії
    x_i_list.extend(x_factor_experiment)
    x_i_list.extend(x_exper_neight)

    list_mx = []  # середнє по усім стовпцям
    for i in range(10):
        list_mx.append(average(x_i_list[i], "m{}".format(i + 1)))

    my = y_average_average  # cереднє всіх значень функції

    """Генерація таблиці зі зручним виводом"""
    name = ["N", "x1", "x2", "x3", "x12", "x23", "x13", "x123", "x1^2", "x2^2", "x3^2"]
    name.extend(["y{}".format(i + 1) for i in range(m)])
    name.append("y_mid")

    for j in range(12 + m):
        print("|{: ^9}|".format(name[j]), end="")
    print()
    for i in range(12 + m):
        print("|{: ^9}|".format("---------"), end="")
    print()
    for i in range(15):
        print("|{: ^9}|".format(i + 1), end="")
        for j in range(10):
            print("|{: ^9}|".format(r(x_i_list[j][i])), end="")
        for j in range(m):
            print("|{: ^9}|".format(r(y[j][i])), end="")
        print("|{: ^9}|".format(r(y_average_list[i])), end="")
        print()
    for i in range(12 + m):
        print("|{: ^9}|".format("---------"), end="")
    print()

    print("|{: ^9}|".format("Mid"), end="")
    for i in list_mx:
        print("|{: ^9}|".format(r(i)), end="")
    for i in y:
        print("|{: ^9}|".format(r(sum(i) / len(i))), end="")
    print("|{: ^9}|".format(r(y_average_average)), end="")
    """Кінець генерації таблиці"""

    a_n = []  # a1, a2, a3......., a10
    for i in range(len(x_i_list)):
        a_n.append(sum(x_i_list[i][k] * y_average_list[i] for k in range(N)) / N)

    a_nm = []  # a11, a12, a12...., a10 10
    for i in range(len(x_i_list)):
        a_nm.append([])
        for j in range(len(x_i_list)):
            a_nm[i].append("")
            a_nm[i][j] = sum(x_i_list[i][k] * x_i_list[j][k] for k in range(N)) / N

    line1_10 = []  # знаменник для визначення коеф.

    line1_10.append([1])  # Заповнюю першу строку (інші строки - циклом)
    line1_10[0].extend([list_mx[i] for i in range(7)])
    line1_10[0].extend([0, 0, 0])

    for i in range(10):  # заповнення інших рядків матриці
        line1_10.append([list_mx[i]])
        for j in range(10):
            line1_10[i + 1].append(a_nm[j][i])

    vilni = [my]  # стовпець вільних членів
    vilni.extend([a_n[i] for i in range(10)])

    numer = [copy.deepcopy(line1_10) for i in range(11)]  # містить зараз 11 копій знаменника
    for i in range(11):  # доступаємось до кожного нумератора
        for j in range(11):
            numer[i][j][i] = vilni[j]  # міняє потрібний стовпець. Тепер у массиві нимер двувимірні массиви чисельників

    denominator = np.array(line1_10)  # знаменник
    numerator = []  # масив, що зберігає чисельники (двовимірні массиви)
    for i in range(len(numer)):
        numerator.append(np.array(numer[i]))

    print("\nЗнаменник")
    for i in denominator:
        for j in i:
            print("|{: ^9}|".format(round(j, 1)), end="")
        print()

    for k in range(len(numerator)):
        print("\nЧисельник {}".format(k + 1))
        for i in numerator[k]:
            for j in i:
                print("|{: ^9}|".format(round(j, 1)), end="")
            print()

    try:
        b = []
        for i in range(len(numerator)):
            b.append(np.linalg.det(numerator[i]) / np.linalg.det(denominator))
    except:
        print("Невизначена помилка під час роботи з матрицями")

    print(
        "\ny = {} + {}*x1 + {}*x2 + {}*x3 + {}*x1*x2 + {}*x1*x3 + {}*x2*x3 + {}*x1*x2*x3 + {}*x1^2 + {}*x2*2 + {}*x3^2\n"
        .format(par(b[0]), par(b[1]), par(b[2]), par(b[3]), par(b[4]), par(b[5]), par(b[6]), par(b[7]), par(b[8]),
                par(b[9]), par(b[10])))

    y_control = []  # Розраховуємо утворені значення для перевірки з середніми
    for i in range(len(x_i_list[0])):
        temp = 0
        temp = b[0] + b[1] * x_i_list[0][i] + b[2] * x_i_list[1][i] + b[3] * x_i_list[2][i] + b[4] * x_i_list[3][i] \
               + b[5] * x_i_list[4][i] + b[6] * x_i_list[5][i] + b[7] * x_i_list[6][i] \
               + b[8] * x_i_list[7][i] + b[9] * x_i_list[8][i] + b[10] * x_i_list[9][i]
        y_control.append(temp)

    for i in range(len(y_control)):
        print("Отримане значення: {}.\t\t\tСередне значення рядка Y: {}".format(r(y_control[i]), r(y_average_list[i])))
    print()

    s2_list = dispers(y, y_average_list, m)  # дисперсії по рядках

    Gp = max(s2_list) / sum(s2_list)
    print("Gp = (max(s2) / sum(s2)) = {}".format(par(Gp)))
    print("f1=m-1={} ; f2=N=4 Рівень значимості приймемо 0.05.".format(m))
    f1 = m - 1
    f2 = N
    p = 0.95
    q = 1 - p
    Gt_tableN15 = {1: 0.4709, 2: 0.3346, 3: 0.2758, 4: 0.2419, 5: 0.2159, 6: 0.2034, 7: 0.1911, 8: 0.1815, 9: 0.1736,
                   10: 0.1671, 16: 0.1429, 36: 0.1144, 144: 0.0889, "inf": 0.667}  # f2 = 15, рівень знач. 0.05
    if f1 <= 10:
        Gt = Gt_tableN15[f1]  # табличне значення критерію Кохрена при N=4, f1=2, рівень значимості 0.05
    elif f1 <= 16:
        Gt = Gt_tableN15[16]
    elif f1 <= 36:
        Gt = Gt_tableN15[36]
    elif f1 <= 144:
        Gt = Gt_tableN15[144]
    else:
        Gt = Gt_tableN15["inf"]
    printf("Gt", Gt)
    if Gp <= Gt:
        Krit_Kohr = "Однор" + " m=" + str(m)
        print("Дисперсія однорідна")

    else:
        Krit_Kohr = "Не однор."
        print("Дисперсія неоднорідна\n\n\n\n")
        print("m+1")
        m += 1

        continue  # цикл починається знову, якщо неоднор. Якщо однорідне, то цикл продовжується

    print("\nКритерію Стьюдента\n")

    """Генерація таблиці зі зручним виводом"""
    print("Нормовані фактори")
    name = ["N", "x0", "x1", "x2", "x3", "x12", "x23", "x13", "x123", "x1^2", "x2^2", "x3^2"]
    name.extend(["y{}".format(i + 1) for i in range(m)])
    name.append("y_mid")

    for j in range(13 + m):
        print("|{: ^9}|".format(name[j]), end="")
    print()
    for i in range(13 + m):
        print("|{: ^9}|".format("---------"), end="")
    print()
    for i in range(15):
        print("|{: ^9}|".format(i + 1), end="")
        for j in range(11):
            print("|{: ^9}|".format(r(x_norm[j][i])), end="")
        for j in range(m):
            print("|{: ^9}|".format(r(y[j][i])), end="")
        print("|{: ^9}|".format(r(y_average_list[i])), end="")
        print()
    for i in range(13 + m):
        print("|{: ^9}|".format("---------"), end="")
    print()

    """Кінець генерації таблиці"""

    s2_B = sum(s2_list) / len(s2_list)
    printf("s2_B", s2_B)

    s2_BetaS = s2_B / (N * m)
    printf("s2_BetaS", s2_BetaS)

    s_BetaS = sqrt(s2_BetaS)
    printf("s_betaS", s_BetaS)

    beta_list = beta(x_norm, y_average_list)  # значенння B0, B1, B2, B3....

    t_list = t(beta_list, s_BetaS)  # t0, t1, t2, t3

    f3 = (m - 1) * N  # N завжди 15
    t_tabl = scipy.stats.t.ppf((1 + (1 - q)) / 2, f3)  # табличне значення за критерієм Стюдента
    printf("t_tabl", t_tabl)

    b_list_St = []
    print("Утворене рівняння регресії: Y = ", end="")
    for i in range(len(t_list)):
        """Форматованне виведення рівняння зі значущими коеф. Не знач. пропускаються"""
        b_list_St.append(0)
        if t_list[i] > t_tabl:
            b_list_St[i] = b[i]
            if i == 0:
                print("{}".format(r(b[i])), end="")
            else:
                print(" + {}*{}".format(par(b[i]), name[i + 1]), end="")
    print()

    # Порівняння результатів
    y_average_row_Student = []
    dodanki = []  # для гарного виведення, буде зберігати текст
    for i in range(len(x_i_list[0])):
        for j in range(len(b_list_St)):
            if j == 0:
                dodanki.append("{}".format(r(b_list_St[j])))  # додає доданок до виведення, якщо він не нуль
            else:
                if b_list_St[j] == 0:
                    dodanki.append("")  # додає доданок до виведення як пустий, якщо він нуль
                else:
                    dodanki.append(" + {}*{}".format(par(b_list_St[j]), x_i_list[j - 1][i]))
        y_average_row_Student.append(0)
        y_average_row_Student[i] = b[0] + b[1] * x_i_list[0][i] + b[2] * x_i_list[1][i] + b[3] * x_i_list[2][i] + b[4] * \
                                   x_i_list[3][i] \
                                   + b[5] * x_i_list[4][i] + b[6] * x_i_list[5][i] + b[7] * x_i_list[6][i] \
                                   + b[8] * x_i_list[7][i] + b[9] * x_i_list[8][i] + b[10] * x_i_list[9][i]

        if abs(y_average_row_Student[i] - y_average_list[i]) >= 20:
            print("Yrow{} = {}{}{}{} = \033[31m {}\t\t\t\033[0mY_average_{}row = \033[31m {}\033[0m".format(
                i + 1, dodanki[0], dodanki[1], dodanki[2], dodanki[3],
                r(y_average_row_Student[i]), i + 1, r(y_average_list[i])))
        elif abs(y_average_row_Student[i] - y_average_list[i]) >= 10:
            print("Yrow{} = {}{}{}{} = {}\t\t\tY_average_{}row =  {}".format(
                i + 1, dodanki[0], dodanki[1], dodanki[2], dodanki[3],
                r(y_average_row_Student[i]), i + 1, r(y_average_list[i])))
            print("Результат приблизно (+-10) збігається! (Рівень значимості 0.05)")
        else:
            print("Yrow{} = {}{}{}{} = {}\t\t\tY_average_{}row =  {}".format(
                i + 1, dodanki[0], dodanki[1], dodanki[2], dodanki[3],
                r(y_average_row_Student[i]), i + 1, r(y_average_list[i])))
            print("Результат приблизно (+-10) збігається! (Рівень значимості 0.05)")
        dodanki.clear()

    print("Критерій Фішера")
    d = len(b_list_St) - b_list_St.count(0)
    f4 = N - d
    s2_od = s2_od_func(y_average_list, y_average_row_Student, m, N, d)

    Fp = s2_od / s2_B
    print("Fp = {} / {} = {}".format(r(s2_od), par(s2_B), r(Fp)))

    F_table = scipy.stats.f.ppf(1 - q, f4, f3)
    printf("F_table", F_table)

    if Fp > F_table:
        print("За критерієм Фішера рівняння регресії неадекватно оригіналу при рівні значимості 0.05")
        Krit_Fish = "Не адекв. "

        print("\nПочаток\n")
        continue  # знову зі збільшенням m
    else:
        print("За критерієм Фішера рівняння регресії адекватно оригіналу при рівні значимості 0.05")
        printf("Номер спроби:", count)
        Krit_Fish = "Адекв."

        break  # якщо программа дійшла до цієї точки, то все виконано вірно, критерії задовольняють умову
