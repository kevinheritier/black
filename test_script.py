from analytics.portfolio_views import *
import time
import matplotlib.pyplot as plt


file_path = 'ProblemTest.xlsx'
file_path_Portfolio = 'ProblemTest - Portfolio.csv'
file_path_Portfolio2 = 'ProblemTest - Portfolio2.csv'
file_path_VCV = 'ProblemTest - VCV Matrix.csv'
file_path_views = 'ProblemTest - Views.csv'
# views = Views()
# views.read_csv_views(file_path_views, r_name='R', c_name='C')
ptf = Portfolio()
ptf2 = Portfolio()
# print(views._df)
ptf.read_csv_ptf(file_path_Portfolio, r_name='R', w_name='W')
ptf.read_csv_cov(file_path_VCV)
ptf2.read_csv_ptf(file_path_Portfolio2, r_name='R', w_name='W')
ptf2.read_csv_cov(file_path_VCV)
# print(ptf.df)
# print(ptf._cov)
ptf.optim_w()
# print(np.linalg.inv(ptf2._cov.values))
# print(ptf2.r.values)
# w = np.linalg.inv(ptf2._cov.values) @ ptf2.r.values
# print(w)
ptf2.optim_w()

# print(ptf._cov)

# print(ptf.df)

# print(sum(ptf.df.loc['w']))


# problem = PortfolioProblem(ptf, views)
# new_ptf = problem.post_portfolio(omega_analytical=True, tau=1)
#new_ptf2 = problem.post_portfolio(omega_analytical=True, tau=0.02)

# print(new_ptf.df)
# print(new_ptf2.df)

# fig, (axr, axw) = plt.subplots(2, 1)


reverse_problem = ReversePortfolioProblem(ptf, ptf2)
views = reverse_problem.compute_views(tau=0.025, conf=0.5)
print(views)
problem = PortfolioProblem(ptf, views)
new_ptf = problem.post_portfolio(omega_analytical=True, tau=0.025)

print(ptf.df)
print(ptf2.df)
print(new_ptf.df)
# print(reverse_problem.Omega)
# print(problem.compute_Omega_analytical(tau=0.025))
#print(reverse_problem.Q - views.df.r)

# newr_asarray = new_ptf.df.loc['r'].values
# neww_asarray = new_ptf.df.loc['w'].values
# r_asarray = ptf.df.loc['r'].values
# w_asarray = ptf.df.loc['w'].values
# asset_asarray = ptf.df.keys()

# xloc = np.arange(len(asset_asarray))
# widthb = 0.3

# axr.bar(xloc - widthb / 2, r_asarray, widthb,
#         label='Market Portfolio', color='aquamarine')
# axr.bar(xloc + widthb / 2, newr_asarray, widthb,
#         label='Tactic Portfolio', color='deepskyblue')
# axw.bar(xloc - widthb / 2, w_asarray, widthb,
#         label='Market Portfolio', color='aquamarine')
# axw.bar(xloc + widthb / 2, neww_asarray, widthb,
#         label='Tactic Portfolio', color='deepskyblue')

# axr.set_ylabel('Excess return')
# axr.set_xticks(xloc)
# axr.set_xticklabels([])
# axr.legend()
# axr.set_title('Black-Litterman')

# axw.set_ylabel('Weight')
# axw.set_xticks(xloc)
# axw.set_xticklabels(asset_asarray, rotation=30)

# plt.show()
