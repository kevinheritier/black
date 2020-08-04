from portfolio_views import *

d = {
    'equities': dict(r=0.07),
    'fixed_income': dict(r=0.03),
}

port2 = Portfolio(d,
<<<<<<< HEAD
                  cov=pd.DataFrame(
                      np.array([[1.2, -0.1], [-0.1, 0.3]]), columns=d.keys(), index=d.keys()),
=======
                  cov=pd.DataFrame(np.array([[1.2, -0.1],[-0.1,0.3]]), columns=d.keys(), index=d.keys()),
>>>>>>> remotes/origin/master
                  kappa=0.1)

v = Views()
df = pd.DataFrame(
<<<<<<< HEAD
    {
        'fixed_income': [0, 1, 1],
        'equities': [1, 0, 0],
        'r': [0.05, 0.01, 0.01],
        'c': [1.0, 1.0, 1.5]
    })
=======
{
    'fixed_income': [0, 1, 1],
    'equities': [1, 0, 0],
    'r': [0.05, 0.01, 0.01],
    'c': [1.0, 1.0, 1.5]
})
>>>>>>> remotes/origin/master
v.add_views(df)


problem = PortfolioProblem(port2, v)
# for k in range(len(v)):
#     problem.post_ret100_k(k, )
<<<<<<< HEAD
=======





>>>>>>> remotes/origin/master
