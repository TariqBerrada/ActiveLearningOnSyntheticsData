import numpy as np

import matplotlib.pyplot as plt
plt.style.use('bmh')


ymin = -10
ymax = 5
xmin, xmax = -10, 10

def b1(x):
    return 2 + 0.32*x*np.cos(.73*x-.32)

def b2(x):
    return 0.4*np.sin(1.87*x + .2)*x - 4

x = np.linspace(-10, 10, 100)

bound1 = b1(x)
bound2 = b2(x)
if __name__ == '__main__':
    plt.figure()
    plt.plot(x, bound1, '--', label='_nolegend_')
    plt.plot(x, bound2, '--', label='_nolegend_')

    plt.fill_between(x, bound2, bound1, color = 'r', alpha = .3)
    plt.fill_between(x, bound1, ymax, color = 'b', alpha = .3)
    plt.fill_between(x, ymin, bound1, color='g', alpha=.3)

    plt.legend(['zone 1', 'zone 2', 'zone 3'])
    plt.show()
