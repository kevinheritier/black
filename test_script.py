from portfolio_views import *

d = {
    'equities': dict(r=0.1),
    'fixed_income': dict(r=0.05),
}

port2 = Portfolio(d,
                  cov=pd.DataFrame(
                      np.array([[0.28, 0], [0, 0.14]]), columns=d.keys(), index=d.keys()),
                  kappa=1)

port2.imp_kap()
# print(port2.kappa)

views = Views()
df = pd.DataFrame(
    {
        'fixed_income': [0, 1],
        'equities': [1, 0],
        'r': [0.05, 0.025],
        'c': [0.1, 0.1]
    })
views.add_views(df)

problem = PortfolioProblem(port2, views)

# print(port2.w)
#p = problem.post_ret100_k(views[0], inplace=False)
# print(p.w)
print(problem.w_pk(views[0]))
print(problem.w_pk(views[1]))
#print(problem.f_k(0.3, port2, views[0]))
Omega = np.diag([minimize(problem.f_k, 0.3, args=(port2, v)).x[0]
                 for v in views])
print(Omega)

InvSig = np.linalg.inv(1 * port2.cov)
InvOme = np.linalg.inv(Omega)
first_term = np.linalg.inv(InvSig + np.dot(views.P.T, InvOme).dot(views.P))
second_term = np.dot(InvSig, port2.r) + \
    np.dot(views.P.T, InvOme).dot(views.df.r)
E = np.dot(first_term, second_term)
print(E)
NCov = port2.cov + \
    np.linalg.inv(InvSig + np.dot(views.P.T, InvOme).dot(views.P))
print(NCov)
print(np.linalg.solve(port2.kappa * NCov, E))
