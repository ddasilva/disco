

# Block size or GPU computations
BLOCK_SIZE = 256

# Number of elements in the state vector
NSTATE = 5

class RK45Coeffs:
    """
    Coefficients of the RK45 Algorithm
    """

    # Cash-Karp Coefficients
    a2 = 1 / 5
    a3 = 3 / 10
    a4 = 3 / 5
    a5 = 1
    a6 = 7 / 8

    b21 = 1 / 5
    b31 = 3 / 40
    b32 = 9 / 40
    b41 = 3 / 10
    b42 = -9 / 10
    b43 = 6 / 5
    b51 = -11 / 54
    b52 = 5 / 2
    b53 = -70 / 27
    b54 = 35 / 27
    b61 = 1631 / 55296
    b62 = 175 / 512
    b63 = 575 / 13824
    b64 = 44275 / 110592
    b65 = 253 / 4096

    c1 = 37 / 378
    c3 = 250 / 621
    c4 = 125 / 594
    c5 = 0
    c6 = 512 / 1771

    d1 = 2825 / 27648
    d3 = 18575 / 48384
    d4 = 13525 / 55296
    d5 = 277 / 14336
    d6 = 1 / 4
