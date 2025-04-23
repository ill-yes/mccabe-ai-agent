def hoch_komplexe_funktion_eins(x):
    result = 0
    for i in range(x):
        if i % 2 == 0:
            result += i
            if i % 3 == 0:
                result += 1
                if i % 5 == 0:
                    result -= 2
                else:
                    if i % 7 == 0:
                        result += 3
                    elif i % 11 == 0:
                        result -= 4
                    else:
                        result += 5
            else:
                if i % 4 == 0:
                    result -= 6
                elif i % 6 == 0:
                    result += 7
                elif i % 9 == 0:
                    result += 8
        else:
            result -= i
            if i % 3 == 1:
                result += 9
            elif i % 5 == 2:
                result -= 10
            else:
                for j in range(3):
                    if j == 0:
                        result += 11
                    elif j == 1:
                        result -= 12
                    else:
                        result += 13
    return result

def hoch_komplexe_funktion_zwei(y):
    total = 1
    i = 0
    while i < y:
        if i % 2 == 1:
            if i % 3 == 2:
                if i % 4 == 1:
                    total += 2
                elif i % 5 == 0:
                    total -= 3
                else:
                    total += 4
            elif i % 6 == 3:
                total -= 5
            else:
                if i % 7 == 2:
                    total += 6
                elif i % 8 == 1:
                    total += 7
                elif i % 9 == 0:
                    total += 8
        else:
            if i % 10 == 5:
                total -= 9
            elif i % 11 == 4:
                total += 10
            elif i % 12 == 3:
                total -= 11
            else:
                for k in range(2):
                    if k == 0:
                        total += 12
                    else:
                        total -= 13
        i += 1
    return total
