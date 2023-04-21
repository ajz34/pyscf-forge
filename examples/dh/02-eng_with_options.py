"""
Common options for DH evaluation
"""

from pyscf import gto, dh, df, dft


if __name__ == "__main__":
    # change RI basis set
    mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="cc-pVTZ").build()
    mf = dh.DH(mol, xc="XYG3").build_scf(auxbasis_jk=df.aug_etb(mol)).run(auxbasis_ri="aug-cc-pVTZ-ri")
    print("Total energy of XYG3 (auxjk: aug_etb, auxri: aVTZ-ri):", mf.e_tot)  # -76.422117115225

    # change to conventional integral
    mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="cc-pVTZ").build()
    mf = dh.DH(mol, xc="XYG3").build_scf(route_scf="Conv").run(route_mp2="Conv")
    print("Total energy of XYG3 (Conventional ERI):", mf.e_tot)  # -76.42211926603686

    # add frozen core
    mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="cc-pVTZ").build()
    mf = dh.DH(mol, xc="XYG3").run(frozen="FreezeG2")
    print("Total energy of XYG3 (Frozen Core):", mf.e_tot)  # -76.41721994022949

    # self-define functional
    mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="cc-pVTZ").build()
    xc = ("B3LYPg", "0.8033*HF - 0.0140*Slater + 0.2107*B88, 0.6789*LYP + 0.3211*MP2")
    mf = dh.DH(mol, xc=xc).run(frozen="FreezeG2")
    print("Total energy of XYG3 (Frozen Core):", mf.e_tot)  # -76.41721994022949

    # build DH object with converged SCF object
    mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="cc-pVTZ").build()
    mf_scf = dft.RKS(mol, xc="B3LYPg").run()
    mf = dh.DH(mf_scf, xc="XYG3", route_scf="Conv").run(route_mp2="RI")  # -76.422119266037
