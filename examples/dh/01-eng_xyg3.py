"""
Single point energy of XYG3
"""

from pyscf import gto, dh


if __name__ == "__main__":
    # Restricted
    mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="cc-pVTZ").build()
    mf = dh.DH(mol, xc="XYG3").run()
    print("Total energy of XYG3:", mf.e_tot)  # -76.4221128758632

    # Unrestricted
    mol = gto.Mole(atom="O; H 1 0.94", basis="cc-pVTZ", spin=1).build()
    mf = dh.DH(mol, xc="XYG3").run()
    print("Total energy of XYG3:", mf.e_tot)  # -75.72651276402937
