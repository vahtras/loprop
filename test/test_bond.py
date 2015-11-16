import unittest
import numpy as np

str_nobond = """AU
3 1 2 1
1     0.00000000     0.00000000     0.00000000    -0.66387672     0.00000000    -0.00000000     0.34509720     3.78326969    -0.00000000    -0.00000000     3.96610412     0.00000000     3.52668267     0.00000000    -0.00000000    -2.98430053     0.00000000    -0.00000000     0.00000000    -0.00000000     1.26744725    -0.00000000     2.16730601
1     1.43043000     0.00000000     1.10716000     0.33193836    -0.16057903    -0.00000000    -0.11299312     1.55235099    -0.00000000     1.15495299     0.60859677    -0.00000000     1.21104235    -4.46820475     0.00000000    -4.55909022    -0.05601735     0.00000000    -3.72029878    -0.00000000     0.46039909    -0.00000000    -2.40410436
1    -1.43043000     0.00000000     1.10716000     0.33193836     0.16057903    -0.00000000    -0.11299312     1.55235099    -0.00000000    -1.15495299     0.60859677     0.00000000     1.21104235     4.46820475    -0.00000000    -4.55909022     0.05601735     0.00000000     3.72029878    -0.00000000     0.46039909    -0.00000000    -2.40410436

Time used in Loprop              :      0.45 (cpu)       0.11 (wall)
"""

str_bond ="""AU
5 1 22 1
1     0.00000000     0.00000000     0.00000000    -0.66387672     0.00000000    -0.00000000     0.41788500     1.19165567     0.00000000     0.00000000     2.74891057     0.00000000     1.33653383     0.00000000     0.00000000     4.18425484     0.00000000    -0.00000000    -0.00000000    -0.00000000     0.19037387     0.00000000     5.96033807
1     0.71521500     0.00000000     0.55358000     0.00000000    -0.06567795    -0.00000000    -0.07278780     2.59161403    -0.00000000     1.21719355     1.98015668    -0.00000000     2.19014883    -7.24839104     0.00000000    -7.16855538     0.59534043     0.00000000    -5.74640170    -0.00000000     1.07707338    -0.00000000    -3.79303206
1     1.43043000     0.00000000     1.10716000     0.33193836    -0.12774005     0.00000000    -0.07659922     0.25654398     0.00000000     0.16487465    -0.00000000    -0.00000000     0.11596794    -0.84400923     0.00000000    -0.97481253    -0.35368757    -0.00000000    -0.84709793     0.00000000    -0.07813759     0.00000000    -0.50758833
1    -0.71521500     0.00000000     0.55358000     0.00000000     0.06567795    -0.00000000    -0.07278780     2.59161403    -0.00000000     1.21719355    -1.98015668     0.00000000     2.19014883     7.24839104    -0.00000000    -7.16855538    -0.59534043     0.00000000     5.74640170    -0.00000000     1.07707338    -0.00000000    -3.79303206
1    -1.43043000     0.00000000     1.10716000     0.33193836     0.12774005     0.00000000    -0.07659922     0.25654398    -0.00000000    -0.16487465     0.00000000     0.00000000     0.11596794     0.84400923    -0.00000000    -0.97481253     0.35368757     0.00000000     0.84709793    -0.00000000    -0.07813759    -0.00000000    -0.50758833

Time used in Loprop              :      0.45 (cpu)       0.11 (wall)
"""

class TestBond(unittest.TestCase):


    def test_bond_nobond_properties(self):
        #a0 = 0.52917721092
        a0 = 1.0

#Read in string that is for no bonds output
        lines = [line for line in str_bond.split('\n') if len(line.split()) > 10 ]

        n_bond = np.array(   [8.0, 0.0, 1.0, 0.0, 1.0], dtype = float )
        r_bond = a0 * np.array(   [l.split()[1:4] for l in lines ], dtype = float)
        q_bond = np.array(   [l.split()[4] for l in lines], dtype = float)
        d_bond = np.array(   [l.split()[5:8] for l in lines], dtype = float)
        a_bond = np.array(   [l.split()[8:15] for l in lines], dtype = float)
        b_bond = np.array(   [l.split()[15:26] for l in lines], dtype = float)

#Read in string that is for bonds output -b
        lines = [line for line in str_nobond.split('\n') if len(line.split()) > 10 ]

        n_nobond = np.array( [8.0, 1.0, 1.0], dtype = float )
        r_nobond = a0 * np.array( [l.split()[1:4] for l in lines ], dtype = float)
        q_nobond = np.array( [l.split()[4] for l in lines], dtype = float)
        d_nobond = np.array( [l.split()[5:8] for l in lines], dtype = float)
        a_nobond = np.array( [l.split()[8:15] for l in lines], dtype = float)
        b_nobond = np.array( [l.split()[15:26] for l in lines], dtype = float)

#Total dipole moment should be the same
        coc_bond = np.einsum( 'ij,i', r_bond , n_bond ) / n_bond.sum()
        coc_nobond = np.einsum( 'ij,i', r_nobond , n_nobond ) / n_nobond.sum()

        np.testing.assert_allclose( coc_bond, coc_nobond )

        a_tot_bond = np.sum(a_bond)
        a_tot_nobond = np.sum(a_nobond)

        np.testing.assert_allclose( a_tot_bond, a_tot_nobond )

        b_tot_bond = np.sum(b_bond)
        b_tot_nobond = np.sum(b_nobond)

        np.testing.assert_allclose( b_tot_bond, b_tot_nobond )

        dip_bond = np.einsum( 'ij,i', (r_bond - coc_bond), q_bond ) + d_bond.sum(axis=0)
        dip_nobond = np.einsum( 'ij,i', (r_nobond - coc_nobond), q_nobond ) + d_nobond.sum(axis = 0 )
