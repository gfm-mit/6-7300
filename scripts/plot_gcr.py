import sys
import os
import pathlib

import seaborn as sns
from matplotlib import pyplot as plt

sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))

from domain_specific.evalf import evalf
from domain_specific.x0 import generate_stochastic_inputs
from domain_specific.jacobian import finiteDifferenceJacobian

if __name__ == '__main__':
    x0, p, u = generate_stochastic_inputs(30)
    J = finiteDifferenceJacobian(evalf, x0, p, u, delta=1e-6)
    sns.heatmap(J, center=0)
    plt.show()