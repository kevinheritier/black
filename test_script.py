from portfolio_views import *

d = {
    'equities': dict(r=0.07),
    'fixed_income': dict(r=0.03),
}

port2 = Portfolio(d,
                  cov=pd.DataFrame(
                      np.array([[1.2, -0.1], [-0.1, 0.3]]), columns=d.keys(), index=d.keys()),
                  kappa=0.1)

views = Views()
df = pd.DataFrame(
    {
        'fixed_income': [0, 1, 1],
        'equities': [1, 0, 0],
        'r': [0.05, 0.01, 0.01],
        'c': [1.0, 1.0, 1.5]
    })
views.add_views(df)


problem = PortfolioProblem(port2, views)

for view in views:
    problem.post_ret100_k(view)
