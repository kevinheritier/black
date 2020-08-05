from portfolio_views import *

d = {
    'equities': dict(r=0.07),
    'fixed_income': dict(r=0.03),
}

port2 = Portfolio(d,
                  cov=pd.DataFrame(
                      np.array([[0.28, 0], [0, 0.12]]), columns=d.keys(), index=d.keys()),
                  kappa=1)

port2.imp_kap()
print(port2.kappa)

views = Views()
df = pd.DataFrame(
    {
        'fixed_income': [0, 1],
        'equities': [1, 0],
        'r': [0.05, 0.02],
        'c': [0.5, 0.5]
    })
views.add_views(df)


problem = PortfolioProblem(port2, views)

print(port2.w)
p = problem.post_ret100_k(views[0], inplace=False)
print(p.w)
print(problem.w_pk(views[0]))
#print(problem.f_k(0.3, port2, views[0]))
res = minimize(problem.f_k, 0.3, args=(port2, views[0]))
print(res)
