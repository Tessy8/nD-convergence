def parberry_comparators(n):
    k = 1
    while k < n:
        k *= 2

    pairs = []

    p = k // 2
    while p >= 1:
        a = 0
        while a < n:
            for b in range(p):
                i = a + b
                j = a + b + p
                if j < n:
                    pairs.append((i, j))
            a += 2 * p
        p //= 2

    p = k // 2
    while p >= 1:
        q = k // 2
        while q >= 2 * p:
            c = 0
            while c < n:
                for d in range(p):
                    i = c + d + p
                    j = c + d + q
                    if j < n:
                        pairs.append((i, j))
                c += 2 * p
            q //= 2
        p //= 2

    return pairs
