import sympy
from sympy import pprint
from sympy import init_printing
init_printing() 

w1 = sympy.symbols('w1')
w2 = sympy.symbols('w2')
z = sympy.symbols('z')

y1 = w1*z
y2 = w2*z

pi_1_star = sympy.symbols('pi_1')
pi_2_star = sympy.symbols('pi_2')
pi_1 = sympy.exp(y1)/(sympy.exp(y1)+sympy.exp(y2))
pi_2 = sympy.exp(y2)/(sympy.exp(y1)+sympy.exp(y2))

loss = pi_1_star*sympy.log(pi_1)# + pi_2_star*sympy.log(pi_2)
pprint(sympy.diff(loss, z).simplify())
