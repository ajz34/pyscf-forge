import unittest
from pyscf.dh.dipole.dh.udh import UDHPolar, UDHDipole
import numpy as np


class TestUDHPolar(unittest.TestCase):
    def test_xyg3(self):
        from pyscf import gto, dft, scf
        np.set_printoptions(10, suppress=True, linewidth=150)
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G", charge=1, spin=1, verbose=0).build()
        mf = UDHPolar(mol, xc="XYG3").run()
        print(mf.de)

        REF = np.array(
            [[3.6687010571, -0., -0.6378231004],
             [-0., 2.639405801, -0.],
             [-0.6378231004, -0., 3.3387967725]])
        self.assertTrue(np.allclose(mf.de, REF, atol=1e-6, rtol=1e-4))

        """
        # generation of numerical result

        def dipole_with_dipole_field(t, h):
            mf_scf = dft.UKS(mol, xc="B3LYPg").density_fit("cc-pVDZ-jkfit")
            mf_scf.get_hcore = lambda *args, **kwargs: scf.hf.get_hcore(mol) - h * mol.intor("int1e_r")[t]
            mf_scf.run(conv_tol=1e-12)
            mf_resp = UDHDipole(mf_scf, xc="XYG3")
            return mf_resp.make_dipole()

        dip_array = np.zeros((2, 3, 3))
        h = 3e-4
        for idx, h in [(0, h), [1, -h]]:
            for t in (0, 1, 2):
                dip_array[idx, t] = dipole_with_dipole_field(t, h)
        pol_num = (dip_array[0] - dip_array[1]) / (2 * h)
        print(pol_num)
        print(mf.de - pol_num)

        self.assertTrue(np.allclose(mf.de, pol_num, atol=1e-6, rtol=1e-4))
        """

    def test_xygjos(self):
        from pyscf import gto, dft, scf
        np.set_printoptions(10, suppress=True, linewidth=150)
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G", charge=1, spin=1, verbose=0).build()
        mf = UDHPolar(mol, xc="XYGJ-OS").run()
        print(mf.de)

        REF = np.array(
            [[3.671872874, 0., -0.6339445187],
             [0., 2.6680054163, -0.],
             [-0.6339445186, -0., 3.343974718]])
        self.assertTrue(np.allclose(mf.de, REF, atol=1e-6, rtol=1e-4))

        """
        # generation of numerical result

        def dipole_with_dipole_field(t, h):
            mf_scf = dft.UKS(mol, xc="B3LYPg").density_fit("cc-pVDZ-jkfit")
            mf_scf.get_hcore = lambda *args, **kwargs: scf.hf.get_hcore(mol) - h * mol.intor("int1e_r")[t]
            mf_scf.run(conv_tol=1e-12)
            mf_resp = UDHDipole(mf_scf, xc="XYGJ-OS")
            return mf_resp.make_dipole()

        dip_array = np.zeros((2, 3, 3))
        h = 3e-4
        for idx, h in [(0, h), [1, -h]]:
            for t in (0, 1, 2):
                dip_array[idx, t] = dipole_with_dipole_field(t, h)
        pol_num = (dip_array[0] - dip_array[1]) / (2 * h)
        print(pol_num)
        print(mf.de - pol_num)

        self.assertTrue(np.allclose(mf.de, pol_num, atol=1e-6, rtol=1e-4))
        """

    def test_rsdh(self):
        from pyscf import gto, dft, scf
        np.set_printoptions(10, suppress=True, linewidth=150)
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G", charge=1, spin=1, verbose=0).build()
        mf = UDHPolar(mol, xc="RS-PBE-P86").run()
        print(mf.de)

        REF = np.array(
            [[3.5978514035, -0., -0.626831992],
             [-0., 2.6971818494, 0.],
             [-0.6268319914, 0., 3.2736321596]])
        self.assertTrue(np.allclose(mf.de, REF, atol=1e-6, rtol=1e-4))

        """
        # generation of numerical result

        def dipole_with_dipole_field(t, h):
            mf_resp = UDHDipole(mol, xc="RS-PBE-P86").build_scf()
            mf_resp.scf.get_hcore = lambda *args, **kwargs: scf.hf.get_hcore(mol) - h * mol.intor("int1e_r")[t]
            mf_resp.scf.conv_tol = 1e-12
            return mf_resp.make_dipole()

        dip_array = np.zeros((2, 3, 3))
        h = 3e-4
        for idx, h in [(0, h), [1, -h]]:
            for t in (0, 1, 2):
                dip_array[idx, t] = dipole_with_dipole_field(t, h)
        pol_num = (dip_array[0] - dip_array[1]) / (2 * h)
        print(pol_num)
        print(mf.de - pol_num)

        self.assertTrue(np.allclose(mf.de, pol_num, atol=1e-6, rtol=1e-4))
        """
