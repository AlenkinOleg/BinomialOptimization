import numpy as np

function_names = ['allinit', 'beale', 'eg1', 'engval2', 'pspdoc', 'rosenbrock', 'scalable',\
                  'michalewicz', 'ackley1', 'gsobol', 'biggs5', 'michalewicz5', 'sqmichalewicz5']

function_dimensions = {'allinit': 4, 'beale': 2, 'eg1': 3, 'engval2': 3, 'pspdoc': 4, 'biggs5': 5}

def functionDoESpecial(X, functionName):
    x = np.array(X, copy=True)

    sqr = lambda x: x**2

    if functionName.lower() == 'allinit':
        functionName = 'allinit'
        inputDimension = 4

        x[:, 0] = 2 * x[:, 0] - 1
        x[:, 1] = x[:, 1] + 1
        x[:, 2] = -x[:, 2] + 1
        x[:, 1] = 2 * x[:, 1]

        y = -4 - (-(sqr(x[:, 0]) + sqr(x[:, 1]) + sqr(x[:, 2] + x[:, 3]) + x[:, 2] + sqr(np.sin(x[:, 2])) + \
                  sqr(x[:, 0]) * sqr(x[:, 1]) + x[:, 3] + sqr(np.sin(x[:, 2])) + sqr(-1 + x[:, 3]) + \
                  sqr(sqr(x[:, 1])) + sqr(sqr(x[:, 2]) + sqr(x[:, 0] + x[:, 3])) + sqr(-4 + sqr(np.sin(x[:, 3])) + \
                                                                                       sqr(x[:, 1]) * sqr(x[:, 2]) + x[:, 0]) + np.sin(x[:, 3])**4))


    elif functionName.lower() == 'beale':
        functionName = 'beale'
        inputDimension = 2

        x = 2 * x - 1

        y = -(-(sqr(-1.5 + x[:, 0] * (1 - x[:, 1])) + sqr(-2.25 + (1 - sqr(x[:, 1])) * x[:, 0]) + \
              sqr(-2.625 + (1 - x[:, 1]**3 * x[:, 0]))))

    elif functionName.lower() == 'eg1':
        functionName = 'eg1'
        inputDimension = 3

        x[:, 0] = 2 * x[:, 0] - 1
        x[:, 1] = 2 * x[:, 1] - 1
        x[:, 2] = x[:, 2] + 1

        y = -(-(sqr(x[:, 0]) + (x[:, 1] * x[:, 2])**4 + x[:, 0] * x[:, 2] + np.sin(x[:, 0] + x[:, 2]) * x[:, 1] + x[:, 1]))

    elif functionName.lower() == 'engval2':
        functionName = 'engval2'
        inputDimension = 3

        x = 2 * x - 1

        y = -(-(sqr(-1 + sqr(x[:, 0]) + sqr(x[:, 1]) + sqr(x[:, 2])) + sqr(-1 + sqr(x[:, 0]) + sqr(x[:, 1]) + \
              sqr(-2 + x[:, 2])) + sqr(-1 + x[:, 0] + x[:, 1] + x[:, 2]) + sqr(1 + x[:, 0] + x[:, 1] - x[:, 2]) + \
        sqr(-36 + 3 * sqr(x[:, 1]) + x[:, 0]**3 + sqr(1 - x[:, 0] + 5 * x[:, 2]))))


    elif functionName.lower() ==  'pspdoc':
        functionName = 'pspdoc'
        inputDimension = 4

        x[:, 0] = -x[:, 0] - 1
        x[:, 1:4] = 2 * x[:, 1:4] - 1

        y = -(-(np.sqrt(1 + sqr(x[:, 0]) + sqr(x[:, 1] - x[:, 2])) + np.sqrt(1 + sqr(x[:, 1]) + sqr(x[:, 2] - x[:, 3]))))


    elif functionName.lower() ==  'rosenbrock':
        functionName = 'rosenbrock'

        c = 2.048
        x = c * (2 * x - 1)
        y = np.sum((1 - x[:, :-1])**2 + \
                   100 * (x[:, 1:] - x[:, :-1]**2)**2, axis=1)


    elif functionName.lower() == 'scalable':
        functionName = 'scalable'

        P = 50000.0
        E = 200.0 * 10**9
        LTOT = 500.0
        SIGMA_BAR = 14000.0
        Y_BAR = 0.5
        beams = 3 * x + 1
        [numberPoints, nbeams] = beams.shape
        H = np.zeros(beams.shape)
        L = np.zeros(beams.shape)
        I = np.zeros(beams.shape)
        M = np.zeros(beams.shape)
        SIGMA = np.zeros(beams.shape)
        Y = np.zeros(beams.shape)
        YPRIME = np.zeros(beams.shape)
        V = 0

        for i in range(nbeams):
            H[:, i] = 20 * beams[:, i]
            L[:, i] = 100.0
            I[:, i] = beams[:, i] * (H[:, i]**3) / 12.0
            sum0 = 0
            for j in range(i + 1):
                sum0 = sum0 + L[:, j]

            M[:, i] = P * (LTOT + L[:, i] - sum0)
            SIGMA[:, i] = M[:, i] * H[:, i] / (2 * I[:, i])
            V = V + beams[:, i] * H[:, i] * L[:, i]
            if i == 0:
                YPRIME[:, 0] = ((P * L[:, i]) / (E * I[:, i])) * (LTOT + (L[:, i] / 2.0) - L[:, i])
            else:
                sum0 = 0
                for j in range(i + 1):
                    sum0 = sum0 + L[:, j]

                YPRIME[:, i] = ((P * L[:, i]) / (E * I[:, i])) * (LTOT + (L[:, i] / 2.0) - sum0) + YPRIME[:, i - 1]

            if i == 0:
                Y[:, 0] = ((P * L[:, i] * L[:, i]) / (2 * E * I[:, i])) * (LTOT + (2 * L[:, i] / 3.0) - L[:, i])
            else:
                sum0 = 0
                for j in range(i + 1):
                    sum0 = sum0 + L[:, j]

                Y[:, i] = ((P * L[:, i] * L[:, i]) / (2 * E * I[:, i])) * (LTOT + (2 * L[:, i] / 3.0) - sum0) + YPRIME[:, i - 1] * L[:, i] + Y[:, i]

        SIGMA = SIGMA / SIGMA_BAR
        y = Y[:, -1] / Y_BAR

        y = np.hstack((V.reshape(-1, 1), SIGMA, y.reshape(-1, 1)))


    elif functionName.lower() == 'michalewicz':
        functionName = 'michalewicz';

        # rescale to 0<x<np.pi
        x = np.pi * x
        numberDimensions = x.shape[1]

        y = 0
        for i in range(numberDimensions):
            y = y + np.sin(x[:, i]) * np.sin(x[:, i]**2 / np.pi)


    elif functionName.lower() == 'ackley1':
        functionName = 'ackley1'

        x = 32.768 * (2 * x - 1)
        numberDimensions = x.shape[1]
        n = numberDimensions
        a = 20
        b = 0.2
        c = 2 * np.pi
        s1 = 0
        s2 = 0
        for i in range(numberDimensions):
            s1 = s1 + x[:, i]**2
            s2 = s2 + np.cos(c * x[:, i])

        y = -a * np.exp(-b * np.sqrt(1.0 / n * s1)) + a + np.exp(1)

    elif functionName.lower() == 'gsobol':
        functionName = 'gSobol'

        # x in [0,1]
        numberDimensions = x.shape[1]
        a = [4.5, 4.5, 1, 0, 1, 9, 0,9]
        y = 1
        for i in range(numberDimensions):
            y = y * (np.abs(4 * x[:, i] - 2) + a[i]) / (1 + a[i])

    elif functionName.lower() == 'biggs5':
        functionName = 'biggs5'
        inputDimension = 5
        # x in [0,1]
        x = np.hstack([x, 3 * np.ones((x.shape[0], 1))])
        y = -(-(sqr(-1.07640035028567 + np.exp(-0.1 * x[:, 0]) * x[:, 2] - np.exp(-0.1 * x[:, 1]) * x[:, 3] + \
              np.exp(-0.1 * x[:, 4]) * x[:, 5]) + sqr(-1.49004122924658 + np.exp(-0.2 * x[:, 0]) * x[:, 2] - \
              np.exp(-0.2 * x[:, 1]) * x[:, 3] + np.exp(-0.2 * x[:, 4]) * x[:, 5]) + sqr(-1.395465514579 + \
              np.exp(-0.3 * x[:, 0]) * x[:, 2] - np.exp(-0.3 * x[:, 1]) * x[:, 3] + np.exp(-0.3 * x[:, 4]) * x[:, 5]) + \
              sqr(-1.18443140557593 + np.exp(-0.4 * x[:, 0]) * x[:, 2] - np.exp(-0.4 * x[:, 1]) * x[:, 3] + \
                  np.exp(-0.4 * x[:, 4]) * x[:, 5]) + sqr(-0.978846774427044 + np.exp(-0.5 * x[:, 0]) * x[:, 2] - \
                  np.exp(-0.5 * x[:, 1]) * x[:, 3] + np.exp(-0.5 * x[:, 4]) * x[:, 5]) + sqr(-0.808571735078932 + \
                  np.exp(-0.6 * x[:, 0]) * x[:, 2] - np.exp(-0.6 * x[:, 1]) * x[:, 3] + np.exp(-0.6 * x[:, 4]) * x[:, 5]) + \
                  sqr(-0.674456081839291 + np.exp(-0.7 * x[:, 0]) * x[:, 2] - np.exp(-0.7 * x[:, 1]) * x[:, 3] + \
                      np.exp(-0.7 * x[:, 4]) * x[:, 5]) + sqr(-0.569938262912808 + np.exp(-0.8 * x[:, 0]) * x[:, 2] - \
                      np.exp(-0.8 * x[:, 1]) * x[:, 3] + np.exp(-0.8 * x[:, 4]) * x[:, 5]) + sqr(-0.487923778062043 + \
                      np.exp(-0.9 * x[:, 0]) * x[:, 2] - np.exp(-0.9 * x[:, 1]) * x[:, 3] + np.exp(-0.9 * x[:, 4]) * x[:, 5]) + \
                      sqr(-0.422599358188832 + np.exp(-x[:, 0]) * x[:, 2] - np.exp(-x[:, 1]) * x[:, 3] + np.exp(-x[:, 4]) * x[:, 5]) + \
                      sqr(-0.369619594903334 + np.exp(-1.1 * x[:, 0]) * x[:, 2] - np.exp(-1.1 * x[:, 1]) * x[:, 3] + \
                          np.exp(-1.1 * x[:, 4]) * x[:, 5]) + sqr(-0.325852731997495 + np.exp(-1.2 * x[:, 0]) * x[:, 2] - \
                          np.exp(-1.2 * x[:, 1]) * x[:, 3] + np.exp(-1.2 * x[:, 4]) * x[:, 5]) + sqr(-0.28907018464926 + \
                          np.exp(-1.3 * x[:, 0]) * x[:, 2] - np.exp(-1.3 * x[:, 1]) * x[:, 3] + np.exp(-1.3 * x[:, 4]) * x[:, 5])));

    elif functionName.lower() == 'michalewicz5':
        functionName = 'michalewicz5'

        # rescale to 0<x<np.pi
        x = np.pi * x
        numberDimensions = x.shape[1]

        y = 0
        for i in range(numberDimensions):
            y = y + np.sin(x[:, i]) * np.sin(x[:, i]**2 / np.pi)

    elif functionName.lower() == 'sqmichalewicz5':
        functionName = 'sqMichalewicz5'

        # rescale to 0<x<np.pi
        x = np.pi * x
        numberDimensions = x.shape[1]

        y = 0
        for i in range(numberDimensions):
            y = y + np.sin(x[:, i]) * np.sin(x[:, i]**2 / np.pi)**2

    elif functionName.lower() == 'rosenbrock5':
        functionName = 'rosenbrock5'

        c = 2.048
        x = c * (2 * x - 1)
        y = np.sum((1 - x[:, :-1])**2 + \
                   100 * (x[:, 1:] - x[:, :-1]**2)**2, axis=1)


    return y