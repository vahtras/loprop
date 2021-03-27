import pathlib

import numpy as np
import pytest

from loprop.core import penalty_function, AU2ANG, pairs
from loprop.dalton import MolFragDalton
from loprop.linalg import triangular_symmetric

from .common import LoPropTestCase
from . import h2o_rot_data as ref

thisdir = pathlib.Path(__file__).parent
case = "h2o_rot"
tmpdir = thisdir / case / "tmp"


@pytest.fixture
def molfrag(request):
    cls = request.param
    return cls(tmpdir, freqs=(0.0,), pf=penalty_function(2.0 / AU2ANG ** 2))


@pytest.mark.parametrize("molfrag", [MolFragDalton], ids=["dalton"], indirect=True)
class TestNew(LoPropTestCase):

    # def setup(self):
    #    molfrag = MolFrag(tmpdir, freqs=(0, ), pf=penalty_function(2.0/AU2ANG**2))
    #    molfragaxDiff = None

    # def tearDown(self):
    #    pass

    def test_nuclear_charge(self, molfrag):
        Z = molfrag.Z
        self.assert_allclose(Z, ref.Z)

    def test_coordinates_au(self, molfrag):
        R = molfrag.R
        self.assert_allclose(R, ref.R)

    def test_default_gauge(self, molfrag):
        self.assert_allclose(molfrag.Rc, ref.Rc)

    def test_total_charge(self, molfrag):
        Qtot = molfrag.Qab.sum()
        self.assert_allclose(Qtot, ref.Qtot)

    def test_charge(self, molfrag):
        Qaa = molfrag.Qa
        self.assert_allclose(ref.Q, Qaa)

    def test_total_dipole(self, molfrag):
        self.assert_allclose(molfrag.Dtot, ref.Dtot)

    def test_dipole_allbonds(self, molfrag):
        D = np.zeros(ref.D.shape)
        Dab = molfrag.Dab
        for ab, a, b in pairs(molfrag.noa):
            D[:, ab] += Dab[:, a, b]
            if a != b:
                D[:, ab] += Dab[:, b, a]
        self.assert_allclose(D, ref.D)

    def test_dipole_allbonds_sym(self, molfrag):
        Dsym = molfrag.Dsym
        self.assert_allclose(Dsym, ref.D)

    def test_dipole_nobonds(self, molfrag):
        Daa = molfrag.Dab.sum(axis=2)
        self.assert_allclose(Daa, ref.Daa)

    def test_quadrupole_total(self, molfrag):
        rRab = np.zeros((6, molfrag.noa, molfrag.noa))
        RRab = np.zeros((6, molfrag.noa, molfrag.noa))
        Rabc = 1.0 * molfrag.Rab
        for a in range(molfrag.noa):
            for b in range(molfrag.noa):
                Rabc[a, b, :] -= molfrag.Rc
        for a in range(molfrag.noa):
            for b in range(molfrag.noa):
                ij = 0
                for i in range(3):
                    for j in range(i, 3):
                        rRab[ij, a, b] = (
                            molfrag.Dab[i, a, b] * Rabc[a, b, j]
                            + molfrag.Dab[j, a, b] * Rabc[a, b, i]
                        )
                        RRab[ij, a, b] = (
                            molfrag.Qab[a, b]
                            * (molfrag.R[a, i] - molfrag.Rc[i])
                            * (molfrag.R[b, j] - molfrag.Rc[j])
                        )
                        ij += 1
        QUcab = molfrag.QUab + rRab + RRab
        QUc = QUcab.sum(axis=2).sum(axis=1)
        self.assert_allclose(QUc, ref.QUc)

    def test_nuclear_quadrupole(self, molfrag):
        QUN = molfrag.QUN
        self.assert_allclose(QUN, ref.QUN)

    def test_quadrupole_allbonds(self, molfrag):
        QU = np.zeros(ref.QU.shape)
        QUab = molfrag.QUab
        for ab, a, b in pairs(molfrag.noa):
            QU[:, ab] += QUab[:, a, b]
            if a != b:
                QU[:, ab] += QUab[:, b, a]
        self.assert_allclose(QU, ref.QU)

    def test_quadrupole_allbonds_sym(self, molfrag):
        QUsym = molfrag.QUsym
        self.assert_allclose(QUsym, ref.QU)

    def test_quadrupole_nobonds(self, molfrag):
        QUaa = (molfrag.QUab + molfrag.dQUab).sum(axis=2)
        self.assert_allclose(QUaa, ref.QUaa)

    def test_Fab(self, molfrag):
        Fab = molfrag.Fab
        self.assert_allclose(Fab, ref.Fab)

    def test_molcas_shift(self, molfrag):
        Fab = molfrag.Fab
        Lab = Fab + molfrag.sf(Fab)
        self.assert_allclose(Lab, ref.Lab)

    def test_total_charge_shift(self, molfrag):
        dQ = molfrag.dQa[0].sum(axis=0)
        dQref = [0.0, 0.0, 0.0]
        self.assert_allclose(dQref, dQ)

    def test_atomic_charge_shift(self, molfrag):
        dQa = molfrag.dQa[0]
        dQaref = (ref.dQa[:, 1::2] - ref.dQa[:, 2::2]) * (1 / (2 * ref.ff))

        self.assert_allclose(dQa, dQaref, atol=0.006)

    def test_lagrangian(self, molfrag):
        # values per "perturbation" as in atomic_charge_shift below
        la = molfrag.la[0]
        laref = (ref.la[:, 0:6:2] - ref.la[:, 1:6:2]) * (1 / (2 * ref.ff))
        # The sign difference is because mocas sets up rhs with opposite sign
        self.assert_allclose(-laref, la, atol=100)

    def test_bond_charge_shift(self, molfrag):
        dQab = molfrag.dQab[0]
        noa = molfrag.noa

        dQabref = (ref.dQab[:, 1:7:2] - ref.dQab[:, 2:7:2]) * (1 / (2 * ref.ff))
        dQabcmp = np.zeros((3, 3))
        ab = 0
        for a in range(noa):
            for b in range(a):
                dQabcmp[ab, :] = dQab[a, b, :]
                ab += 1
        # The sign difference is because mocas sets up rhs with opposite sign
        self.assert_allclose(-dQabref, dQabcmp, atol=0.006)

    def test_bond_charge_shift_sum(self, molfrag):
        dQa = molfrag.dQab[0].sum(axis=1)
        dQaref = molfrag.dQa[0]
        self.assert_allclose(dQa, dQaref)

    def test_polarizability_total(self, molfrag):

        Am = molfrag.Am[0]
        self.assert_allclose(Am, ref.Am, 0.015)

    def test_polarizability_allbonds_molcas_internal(self, molfrag):

        O = ref.O
        H1O = ref.H1O
        H1 = ref.H1
        H2O = ref.H2O
        H2H1 = ref.H2H1
        H2 = ref.H2
        rMP = ref.rMP

        RO, RH1, RH2 = molfrag.R
        ROx, ROy, ROz = RO
        RH1x, RH1y, RH1z = RH1
        RH2x, RH2y, RH2z = RH2

        ihff = 1 / (2 * ref.ff)

        q, x, y, z = range(4)
        dx1, dx2, dy1, dy2, dz1, dz2 = 1, 2, 3, 4, 5, 6
        o, h1o, h1, h2o, h2h1, h2 = range(6)

        Oxx = ihff * (rMP[x, dx1, o] - rMP[x, dx2, o])
        Oyx = (
            ihff
            * (rMP[y, dx1, o] - rMP[y, dx2, o] + rMP[x, dy1, o] - rMP[x, dy2, o])
            / 2
        )
        Oyy = ihff * (rMP[y, dy1, o] - rMP[y, dy2, o])
        Ozx = (
            ihff
            * (rMP[z, dx1, o] - rMP[z, dx2, o] + rMP[x, dz1, o] - rMP[x, dz2, o])
            / 2
        )
        Ozy = (
            ihff
            * (rMP[z, dy1, o] - rMP[z, dy2, o] + rMP[y, dz1, o] - rMP[y, dz2, o])
            / 2
        )
        Ozz = ihff * (rMP[z, dz1, o] - rMP[z, dz2, o])
        H1Oxx = ihff * (
            rMP[x, dx1, h1o]
            - rMP[x, dx2, h1o]
            - (rMP[q, dx1, h1o] - rMP[q, dx2, h1o]) * (RH1x - ROx)
        )
        H1Oyx = ihff * (
            (rMP[y, dx1, h1o] - rMP[y, dx2, h1o] + rMP[x, dy1, h1o] - rMP[x, dy2, h1o])
            / 2
            - (rMP[q, dx1, h1o] - rMP[q, dx2, h1o]) * (RH1y - ROy)
            #      - (rMP[0, dy1, h1o] - rMP[0, dy2, h1o])*(RH1x-ROx) THIS IS REALLY... A BUG?
        )
        H1Oyy = ihff * (
            rMP[y, dy1, h1o]
            - rMP[y, dy2, h1o]
            - (rMP[q, dy1, h1o] - rMP[q, dy2, h1o]) * (RH1y - ROy)
        )
        H1Ozx = ihff * (
            (rMP[z, dx1, h1o] - rMP[z, dx2, h1o] + rMP[x, dz1, h1o] - rMP[x, dz2, h1o])
            / 2
            - (rMP[q, dx1, h1o] - rMP[q, dx2, h1o]) * (RH1z - ROz)
            #             - (rMP[q, dz1, h1o] - rMP[q, dz2, h1o])*(RH1x-ROx) #THIS IS REALLY... A BUG?
        )
        H1Ozy = ihff * (
            (rMP[z, dy1, h1o] - rMP[z, dy2, h1o] + rMP[y, dz1, h1o] - rMP[y, dz2, h1o])
            / 2
            - (rMP[q, dy1, h1o] - rMP[q, dy2, h1o]) * (RH1z - ROz)
            #     - (rMP[q, dz1, h1o] - rMP[q, dz2, h1o])*(RH1y-ROy) THIS IS REALLY... A BUG?
        )
        H1Ozz = ihff * (
            rMP[z, dz1, h1o]
            - rMP[z, dz2, h1o]
            - (rMP[q, dz1, h1o] - rMP[q, dz2, h1o]) * (RH1z - ROz)
        )
        H1xx = ihff * (rMP[x, dx1, h1] - rMP[x, dx2, h1])
        H1yx = (
            ihff * (rMP[y, dx1, h1] - rMP[y, dx2, h1])
            + ihff * (rMP[x, dy1, h1] - rMP[x, dy2, h1])
        ) / 2
        H1yy = ihff * (rMP[y, dy1, h1] - rMP[y, dy2, h1])
        H1zx = (
            ihff * (rMP[z, dx1, h1] - rMP[z, dx2, h1])
            + ihff * (rMP[x, dz1, h1] - rMP[x, dz2, h1])
        ) / 2
        H1zy = (
            ihff * (rMP[z, dy1, h1] - rMP[z, dy2, h1])
            + ihff * (rMP[y, dz1, h1] - rMP[y, dz2, h1])
        ) / 2
        H1zz = ihff * (rMP[z, dz1, h1] - rMP[z, dz2, h1])
        H2Oxx = ihff * (
            rMP[x, dx1, h2o]
            - rMP[x, dx2, h2o]
            - (rMP[q, dx1, h2o] - rMP[q, dx2, h2o]) * (RH2x - ROx)
        )
        H2Oyx = ihff * (
            (rMP[y, dx1, h2o] - rMP[y, dx2, h2o] + rMP[x, dy1, h2o] - rMP[x, dy2, h2o])
            / 2
            - (rMP[q, dx1, h2o] - rMP[q, dx2, h2o]) * (RH2y - ROy)
            #      - (rMP[q, dy1, h1o] - rMP[q, dy2, h1o])*(RH2x-ROx) THIS IS REALLY... A BUG?
        )
        H2Oyy = ihff * (
            rMP[y, dy1, h2o]
            - rMP[y, dy2, h2o]
            - (rMP[q, dy1, h2o] - rMP[q, dy2, h2o]) * (RH2y - ROy)
        )
        H2Ozx = ihff * (
            (rMP[z, dx1, h2o] - rMP[z, dx2, h2o] + rMP[x, dz1, h2o] - rMP[x, dz2, h2o])
            / 2
            - (rMP[q, dx1, h2o] - rMP[q, dx2, h2o]) * (RH2z - ROz)
            #             - (rMP[q, dz1, h1o] - rMP[q, dz2, h1o])*(RH2x-ROx) #THIS IS REALLY... A BUG?
        )
        H2Ozy = ihff * (
            (rMP[z, dy1, h2o] - rMP[z, dy2, h2o] + rMP[y, dz1, h2o] - rMP[y, dz2, h2o])
            / 2
            - (rMP[q, dy1, h2o] - rMP[q, dy2, h2o]) * (RH2z - ROz)
            #     - (rMP[q, dz1, h2o] - rMP[q, dz2, h2o])*(RH2y-ROy) THIS IS REALLY... A BUG?
        )
        H2Ozz = ihff * (
            rMP[z, dz1, h2o]
            - rMP[z, dz2, h2o]
            - (rMP[q, dz1, h2o] - rMP[q, dz2, h2o]) * (RH2z - ROz)
        )
        H2H1xx = ihff * (
            rMP[x, dx1, h2h1]
            - rMP[x, dx2, h2h1]
            - (rMP[q, dx1, h2h1] - rMP[q, dx2, h2h1]) * (RH2x - RH1x)
        )
        H2H1yx = ihff * (
            (
                rMP[y, dx1, h2h1]
                - rMP[y, dx2, h2h1]
                + rMP[x, dy1, h2h1]
                - rMP[x, dy2, h2h1]
            )
            / 2
            - (rMP[q, dx1, h2h1] - rMP[q, dx2, h2h1]) * (RH1y - ROy)
            #      - (rMP[q, dy1, h2h1] - rMP[q, dy2, h2h1])*(RH1x-ROx) THIS IS REALLY... A BUG?
        )
        H2H1yy = ihff * (
            rMP[y, dy1, h2h1]
            - rMP[y, dy2, h2h1]
            - (rMP[q, dy1, h2h1] - rMP[q, dy2, h2h1]) * (RH2y - RH1y)
        )
        H2H1zx = ihff * (
            (
                rMP[z, dx1, h2h1]
                - rMP[z, dx2, h2h1]
                + rMP[x, dz1, h2h1]
                - rMP[x, dz2, h2h1]
            )
            / 2
            - (rMP[q, dx1, h2h1] - rMP[q, dx2, h2h1]) * (RH1z - ROz)
            #     - (rMP[q, dz1, h2h1] - rMP[q, dz2, h2h1])*(RH1x-ROx) #THIS IS REALLY... A BUG?
        )
        H2H1zy = ihff * (
            (
                rMP[z, dy1, h2h1]
                - rMP[z, dy2, h2h1]
                + rMP[y, dz1, h2h1]
                - rMP[y, dz2, h2h1]
            )
            / 2
            - (rMP[q, dy1, h2h1] - rMP[q, dy2, h2h1]) * (RH1z - ROz)
            #     - (rMP[q, dz1, h2h1] - rMP[q, dz2, h2h1])*(RH1y-RO[1]) THIS IS REALLY... A BUG?
        )
        H2H1zz = ihff * (
            rMP[z, dz1, h2h1]
            - rMP[z, dz2, h2h1]
            - (rMP[q, dz1, h2h1] - rMP[q, dz2, h2h1]) * (RH2z - RH1z)
        )
        H2xx = ihff * (rMP[x, dx1, h2] - rMP[x, dx2, h2])
        H2yx = (
            ihff * (rMP[y, dx1, h2] - rMP[y, dx2, h2])
            + ihff * (rMP[x, dy1, h2] - rMP[x, dy2, h2])
        ) / 2
        H2yy = ihff * (rMP[y, dy1, h2] - rMP[y, dy2, h2])
        H2zx = (
            ihff * (rMP[z, dx1, h2] - rMP[z, dx2, h2])
            + ihff * (rMP[x, dz1, h2] - rMP[x, dz2, h2])
        ) / 2
        H2zy = (
            ihff * (rMP[z, dy1, h2] - rMP[z, dy2, h2])
            + ihff * (rMP[y, dz1, h2] - rMP[y, dz2, h2])
        ) / 2
        H2zz = ihff * (rMP[z, dz1, h2] - rMP[z, dz2, h2])

        self.assert_allclose(O[0], Oxx, text="Oxx")
        self.assert_allclose(O[1], Oyx, text="Oyx")
        self.assert_allclose(O[2], Oyy, text="Oyy")
        self.assert_allclose(O[3], Ozx, text="Ozx")
        self.assert_allclose(O[4], Ozy, text="Ozy")
        self.assert_allclose(O[5], Ozz, text="Ozz")
        self.assert_allclose(H1O[0], H1Oxx, text="H1Oxx")
        self.assert_allclose(H1O[1], H1Oyx, text="H1Oyx")
        self.assert_allclose(H1O[2], H1Oyy, text="H1Oyy")
        self.assert_allclose(H1O[3], H1Ozx, text="H1Ozx")
        self.assert_allclose(H1O[4], H1Ozy, text="H1Ozy")
        self.assert_allclose(H1O[5], H1Ozz, text="H1Ozz")
        self.assert_allclose(H1[0], H1xx, text="H1xx")
        self.assert_allclose(H1[1], H1yx, text="H1yx")
        self.assert_allclose(H1[2], H1yy, text="H1yy")
        self.assert_allclose(H1[3], H1zx, text="H1zx")
        self.assert_allclose(H1[4], H1zy, text="H1zy")
        self.assert_allclose(H1[5], H1zz, text="H1zz")
        self.assert_allclose(H2O[0], H2Oxx, text="H2Oxx")
        self.assert_allclose(H2O[1], H2Oyx, text="H2Oyx")
        self.assert_allclose(H2O[2], H2Oyy, text="H2Oyy")
        self.assert_allclose(H2O[3], H2Ozx, text="H2Ozx")
        self.assert_allclose(H2O[4], H2Ozy, text="H2Ozy")
        self.assert_allclose(H2O[5], H2Ozz, text="H2Ozz")
        self.assert_allclose(H2H1[0], H2H1xx, text="H2H1xx")
        self.assert_allclose(H2H1[1], H2H1yx, text="H2H1yx")
        self.assert_allclose(H2H1[2], H2H1yy, text="H2H1yy")
        self.assert_allclose(H2H1[3], H2H1zx, text="H2H1zx")
        self.assert_allclose(H2H1[4], H2H1zy, text="H2H1zy")
        self.assert_allclose(H2H1[5], H2H1zz, text="H2H1zz")
        self.assert_allclose(H2[0], H2xx, text="H2xx")
        self.assert_allclose(H2[1], H2yx, text="H2yx")
        self.assert_allclose(H2[2], H2yy, text="H2yy")
        self.assert_allclose(H2[3], H2zx, text="H2zx")
        self.assert_allclose(H2[4], H2zy, text="H2zy")
        self.assert_allclose(H2[5], H2zz, text="H2zz")

    def test_altint(self, molfrag):
        R = molfrag.R
        rMP = ref.rMP
        diff = [(1, 2), (3, 4), (5, 6)]
        bonds = (1, 3, 4)
        ablab = ("O", "H1O", "H1", "H2O", "H2H1", "H2")
        ijlab = ("xx", "yx", "yy", "zx", "zy", "zz")

        pol = np.zeros((6, molfrag.noa * (molfrag.noa + 1) // 2))
        for ab, a, b in pairs(molfrag.noa):
            for ij, i, j in pairs(3):
                i1, i2 = diff[i]
                j1, j2 = diff[j]
                pol[ij, ab] += (
                    rMP[i + 1, j1, ab]
                    - rMP[i + 1, j2, ab]
                    + rMP[j + 1, i1, ab]
                    - rMP[j + 1, i2, ab]
                ) / (4 * ref.ff)
                if ab in bonds:
                    pol[ij, ab] -= (
                        (R[a][i] - R[b][i])
                        * (rMP[0, j1, ab] - rMP[0, j2, ab])
                        / (2 * ref.ff)
                    )
                self.assert_allclose(
                    ref.Aab[ij, ab], pol[ij, ab], text="%s%s" % (ablab[ab], ijlab[ij])
                )

    def test_polarizability_allbonds_atoms(self, molfrag):

        Aab = molfrag.Aab[0]  # + molfrag.dAab[0]
        noa = molfrag.noa

        Acmp = np.zeros(ref.Aab.shape)

        ab = 0
        lower = np.tril_indices(3)
        for a in range(noa):
            for b in range(a):
                Asab = (Aab[:, :, a, b] + Aab[:, :, b, a])
                Asab = .5 * (Asab + Asab.T)
                Acmp[:, ab] = Asab[lower]
                ab += 1
            Asaa = Aab[:, :, a, a]
            Asaa = .5 * (Asaa + Asaa.T)
            Acmp[:, ab] = Asaa[lower]
            ab += 1
        # atoms
        self.assert_allclose(ref.Aab[:, 0], Acmp[:, 0], atol=0.005)
        self.assert_allclose(ref.Aab[:, 2], Acmp[:, 2], atol=0.005)
        self.assert_allclose(ref.Aab[:, 5], Acmp[:, 5], atol=0.005)

    def test_polarizability_allbonds_bonds(self, molfrag):

        Aab = molfrag.Aab[0] + molfrag.dAab[0] * .5
        noa = molfrag.noa

        Acmp = np.zeros(ref.Aab.shape)

        ab = 0
        lower = np.tril_indices(3)
        for a in range(noa):
            for b in range(a):
                Asab = (Aab[:, :, a, b] + Aab[:, :, b, a])
                Asab = .5 * (Asab + Asab.T)
                Acmp[:, ab] = Asab[lower]
                ab += 1
            Asaa = Aab[:, :, a, a]
            Asaa = .5 * (Asaa + Asaa.T)
            Acmp[:, ab] = Asaa[lower]
            ab += 1
        # atoms
        self.assert_allclose(ref.Aab[:, 1], Acmp[:, 1], atol=0.150, err_msg="H1O")
        self.assert_allclose(ref.Aab[:, 3], Acmp[:, 3], atol=0.150, err_msg="H2O")
        self.assert_allclose(ref.Aab[:, 4], Acmp[:, 4], atol=0.005, err_msg="H2H1")

    def test_polarizability_nobonds(self, molfrag):

        Aab = molfrag.Aab[0] + molfrag.dAab[0] * .5
        noa = molfrag.noa

        Acmp = np.zeros((6, noa))
        Aa = Aab.sum(axis=3)

        for a in range(noa):
            Acmp[:, a] = triangular_symmetric(Aa[:, :, a])

        # atoms
        self.assert_allclose(Acmp, ref.Aa, atol=0.07)

    def test_potfile_PAn0(self, molfrag):
        PAn0 = molfrag.output_potential_file(maxl=-1, pol=0, hyper=0)
        assert PAn0 == ref.PAn0

    def test_potfile_PA00(self, molfrag):
        PA00 = molfrag.output_potential_file(maxl=0, pol=0, hyper=0)
        assert PA00 == ref.PA00

    def test_potfile_PA10(self, molfrag):
        PA10 = molfrag.output_potential_file(maxl=1, pol=0, hyper=0)
        assert PA10 == ref.PA10

    def test_potfile_PA20(self, molfrag):
        PA20 = molfrag.output_potential_file(maxl=2, pol=0, hyper=0)
        assert PA20 == ref.PA20

    def test_potfile_PA21(self, molfrag):
        PA21 = molfrag.output_potential_file(maxl=2, pol=1, hyper=0)
        assert PA21 == ref.PA21

    def test_potfile_PA22(self, molfrag):
        PA22 = molfrag.output_potential_file(maxl=2, pol=2, hyper=0)
        assert PA22 == ref.PA22
