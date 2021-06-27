from analytics.portfolio_views import *
import time
import matplotlib.pyplot as plt


file_path = 'ProblemTest.xlsx'
file_path_Portfolio = 'ProblemTest - Portfolio.csv'
file_path_VCV = 'ProblemTest - VCV Matrix.csv'
file_path_views = 'ProblemTest - Views.csv'
views = Views()
views.read_csv_views(file_path_views, r_name='R', c_name='C')
ptf = Portfolio()
# print(views._df)
ptf.read_csv_ptf(file_path_Portfolio, r_name='R', w_name='W')
ptf.read_csv_cov(file_path_VCV)
# print(ptf.df)
# print(ptf._cov)
ptf.optim_w()
# print(ptf.df)
# print(ptf._cov)

# print(ptf.df)

# print(sum(ptf.df.loc['w']))


problem = PortfolioProblem(ptf, views)
new_ptf = problem.post_portfolio(omega_analytical=True, tau=1)
#new_ptf2 = problem.post_portfolio(omega_analytical=True, tau=0.02)

# print(new_ptf.df)
# print(new_ptf2.df)

fig, (axr, axw) = plt.subplots(2, 1)

newr_asarray = new_ptf.df.loc['r'].values
neww_asarray = new_ptf.df.loc['w'].values
r_asarray = ptf.df.loc['r'].values
w_asarray = ptf.df.loc['w'].values
asset_asarray = ptf.df.keys()

xloc = np.arange(len(asset_asarray))
widthb = 0.3

axr.bar(xloc - widthb / 2, r_asarray, widthb,
        label='Market Portfolio', color='aquamarine')
axr.bar(xloc + widthb / 2, newr_asarray, widthb,
        label='Tactic Portfolio', color='deepskyblue')
axw.bar(xloc - widthb / 2, w_asarray, widthb,
        label='Market Portfolio', color='aquamarine')
axw.bar(xloc + widthb / 2, neww_asarray, widthb,
        label='Tactic Portfolio', color='deepskyblue')

axr.set_ylabel('Excess return')
axr.set_xticks(xloc)
axr.set_xticklabels([])
axr.legend()
axr.set_title('Black-Litterman')

axw.set_ylabel('Weight')
axw.set_xticks(xloc)
axw.set_xticklabels(asset_asarray, rotation=30)

plt.show()
