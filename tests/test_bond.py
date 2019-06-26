import numpy as np

STR_NOBOND = """AU
3 1 2 1
1     0.00000000     0.00000000     0.00000000    -0.66387672     0.00000000    -0.00000000     0.34509720     3.78326969    -0.00000000    -0.00000000     3.96610412     0.00000000     3.52668267     0.00000000    -0.00000000    -2.98430053     0.00000000    -0.00000000     0.00000000    -0.00000000     1.26744725    -0.00000000     2.16730601
1     1.43043000     0.00000000     1.10716000     0.33193836    -0.16057903    -0.00000000    -0.11299312     1.55235099    -0.00000000     1.15495299     0.60859677    -0.00000000     1.21104235    -4.46820475     0.00000000    -4.55909022    -0.05601735     0.00000000    -3.72029878    -0.00000000     0.46039909    -0.00000000    -2.40410436
1    -1.43043000     0.00000000     1.10716000     0.33193836     0.16057903    -0.00000000    -0.11299312     1.55235099    -0.00000000    -1.15495299     0.60859677     0.00000000     1.21104235     4.46820475    -0.00000000    -4.55909022     0.05601735     0.00000000     3.72029878    -0.00000000     0.46039909    -0.00000000    -2.40410436

Time used in Loprop              :      0.45 (cpu)       0.11 (wall)
"""

STR_BOND = """AU
5 1 22 1
1     0.00000000     0.00000000     0.00000000    -0.66387672     0.00000000    -0.00000000     0.41788500     1.19165567     0.00000000     0.00000000     2.74891057     0.00000000     1.33653383     0.00000000     0.00000000     4.18425484     0.00000000    -0.00000000    -0.00000000    -0.00000000     0.19037387     0.00000000     5.96033807
1     0.71521500     0.00000000     0.55358000     0.00000000    -0.06567795    -0.00000000    -0.07278780     2.59161403    -0.00000000     1.21719355     1.98015668    -0.00000000     2.19014883    -7.24839104     0.00000000    -7.16855538     0.59534043     0.00000000    -5.74640170    -0.00000000     1.07707338    -0.00000000    -3.79303206
1     1.43043000     0.00000000     1.10716000     0.33193836    -0.12774005     0.00000000    -0.07659922     0.25654398     0.00000000     0.16487465    -0.00000000    -0.00000000     0.11596794    -0.84400923     0.00000000    -0.97481253    -0.35368757    -0.00000000    -0.84709793     0.00000000    -0.07813759     0.00000000    -0.50758833
1    -0.71521500     0.00000000     0.55358000     0.00000000     0.06567795    -0.00000000    -0.07278780     2.59161403    -0.00000000     1.21719355    -1.98015668     0.00000000     2.19014883     7.24839104    -0.00000000    -7.16855538    -0.59534043     0.00000000     5.74640170    -0.00000000     1.07707338    -0.00000000    -3.79303206
1    -1.43043000     0.00000000     1.10716000     0.33193836     0.12774005     0.00000000    -0.07659922     0.25654398    -0.00000000    -0.16487465     0.00000000     0.00000000     0.11596794     0.84400923    -0.00000000    -0.97481253     0.35368757     0.00000000     0.84709793    -0.00000000    -0.07813759    -0.00000000    -0.50758833

Time used in Loprop              :      0.45 (cpu)       0.11 (wall)
"""


class TestBondH2O:
    """H2O tests bonded versus non-bonden results"""

    def setup(self):
        # Read in string that is for no bonds output
        lines = [line for line in STR_BOND.split("\n") if len(line.split()) > 10]
        a0 = 1.0

        self.n_bond = np.array([8.0, 0.0, 1.0, 0.0, 1.0], dtype=float)
        self.r_bond = a0 * np.array([l.split()[1:4] for l in lines], dtype=float)
        self.q_bond = np.array([l.split()[4] for l in lines], dtype=float)
        self.d_bond = np.array([l.split()[5:8] for l in lines], dtype=float)
        self.a_bond = np.array([l.split()[8:15] for l in lines], dtype=float)
        self.b_bond = np.array([l.split()[15:26] for l in lines], dtype=float)

        self.coc_bond = np.einsum("ij,i", self.r_bond, self.n_bond) / self.n_bond.sum()

        # Read in string that is for bonds output -b
        lines = [line for line in STR_NOBOND.split("\n") if len(line.split()) > 10]

        self.n_nobond = np.array([8.0, 1.0, 1.0], dtype=float)
        self.r_nobond = a0 * np.array([l.split()[1:4] for l in lines], dtype=float)
        self.q_nobond = np.array([l.split()[4] for l in lines], dtype=float)
        self.d_nobond = np.array([l.split()[5:8] for l in lines], dtype=float)
        self.a_nobond = np.array([l.split()[8:15] for l in lines], dtype=float)
        self.b_nobond = np.array([l.split()[15:26] for l in lines], dtype=float)

        self.coc_nobond = (
            np.einsum("ij,i", self.r_nobond, self.n_nobond) / self.n_nobond.sum()
        )

    def test_bond_nobond_properties(self):
        """Center-of-charge equality"""
        np.testing.assert_allclose(self.coc_bond, self.coc_nobond)

    def test_a(self):
        """Polarizability equality"""
        a_tot_bond = np.sum(self.a_bond)
        a_tot_nobond = np.sum(self.a_nobond)

        np.testing.assert_allclose(a_tot_bond, a_tot_nobond)

    def test_b(self):
        """Hyperpolarizability equality"""
        b_tot_bond = np.sum(self.b_bond)
        b_tot_nobond = np.sum(self.b_nobond)

        np.testing.assert_allclose(b_tot_bond, b_tot_nobond)

    def test_dip(self):
        """Dipole equality"""
        dip_bond = np.einsum(
            "ij,i", (self.r_bond - self.coc_bond), self.q_bond
        ) + self.d_bond.sum(axis=0)
        dip_nobond = np.einsum(
            "ij,i", (self.r_nobond - self.coc_nobond), self.q_nobond
        ) + self.d_nobond.sum(axis=0)
        np.testing.assert_allclose(dip_bond, dip_nobond)


class TestBondH2S:
    """H2O tests bonded versus non-bonden results"""

    def setup(self):
        # Read in string that is for no bonds output
        lines = [line for line in STR_BOND.split("\n") if len(line.split()) > 10]
        a0 = 1.0

        self.n_bond = np.array([16.0, 0.0, 1.0, 0.0, 1.0], dtype=float)
        self.r_bond = a0 * np.array([l.split()[1:4] for l in lines], dtype=float)
        self.q_bond = np.array([l.split()[4] for l in lines], dtype=float)
        self.d_bond = np.array([l.split()[5:8] for l in lines], dtype=float)
        self.a_bond = np.array([l.split()[8:15] for l in lines], dtype=float)
        self.b_bond = np.array([l.split()[15:26] for l in lines], dtype=float)

        self.coc_bond = np.einsum("ij,i", self.r_bond, self.n_bond) / self.n_bond.sum()

        # Read in string that is for bonds output -b
        lines = [line for line in STR_NOBOND.split("\n") if len(line.split()) > 10]

        self.n_nobond = np.array([16.0, 1.0, 1.0], dtype=float)
        self.r_nobond = a0 * np.array([l.split()[1:4] for l in lines], dtype=float)
        self.q_nobond = np.array([l.split()[4] for l in lines], dtype=float)
        self.d_nobond = np.array([l.split()[5:8] for l in lines], dtype=float)
        self.a_nobond = np.array([l.split()[8:15] for l in lines], dtype=float)
        self.b_nobond = np.array([l.split()[15:26] for l in lines], dtype=float)

        self.coc_nobond = (
            np.einsum("ij,i", self.r_nobond, self.n_nobond) / self.n_nobond.sum()
        )

    def test_bond_nobond_properties(self):
        """Center-of-charge equality"""
        np.testing.assert_allclose(self.coc_bond, self.coc_nobond)

    def test_a(self):
        """Polarizability equality"""
        a_tot_bond = np.sum(self.a_bond)
        a_tot_nobond = np.sum(self.a_nobond)

        np.testing.assert_allclose(a_tot_bond, a_tot_nobond)

    def test_b(self):
        """Hyperpolarizability equality"""
        b_tot_bond = np.sum(self.b_bond)
        b_tot_nobond = np.sum(self.b_nobond)

        np.testing.assert_allclose(b_tot_bond, b_tot_nobond)

    def test_dip(self):
        """Dipole equality"""
        dip_bond = np.einsum(
            "ij,i", (self.r_bond - self.coc_bond), self.q_bond
        ) + self.d_bond.sum(axis=0)
        dip_nobond = np.einsum(
            "ij,i", (self.r_nobond - self.coc_nobond), self.q_nobond
        ) + self.d_nobond.sum(axis=0)
        np.testing.assert_allclose(dip_bond, dip_nobond)
