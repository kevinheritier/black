from portfolio_views import *

d = {
    'equities': dict(r=0.07),
    'fixed_income': dict(r=0.03),
}

port2 = Portfolio(d,
                  cov=pd.DataFrame(
                      np.array([[0.04, 0], [0, 0.01]]), columns=d.keys(), index=d.keys()),
                  kappa=1)

port2.imp_kap()


v = Views()
df = pd.DataFrame(
    {
        'fixed_income': [0, 1],
        'equities': [1, 0],
        'r': [0.05, 0.01],
        'c': [1.0, 1.0]
    })
v.add_views(df)


problem = PortfolioProblem(port2, v)
# for k in range(len(v)):
#     problem.post_ret100_k(k, )

# port2.display()

problem.post_ret100_k(0)
