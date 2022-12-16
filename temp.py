from seldonian.utils.io_utils import load_pickle
from seldonian.utils.plot_utils import plot_gradient_descent
import matplotlib.pyplot as plt


cs_file = 'logs/candidate_selection_log51.p'
solution_dict = load_pickle(cs_file)

fig = plot_gradient_descent(solution_dict,
    primary_objective_name='log loss',
    save=False,savename='test.png')
plt.show()