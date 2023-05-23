""" Unrestricted Ring-CCD. """


from pyscf.dh.energy.ring_ccd.rring_ccd import RRingCCDConv
from pyscf.dh.util import pad_omega
from pyscf import ao2mo, lib
import numpy as np

α, β = 0, 1
αα, αβ, ββ = 0, 1, 2


def kernel_energy_uring_ccd_conv(
        mo_energy, mo_coeff, eri_or_mol, mo_occ,
        tol_e=1e-8, tol_amp=1e-6, max_cycle=64,
        diis_start=3, diis_space=6,
        verbose=lib.logger.NOTE):
    """ dRPA evaluation by ring-CCD with conventional integral (unrestricted).

    Parameters
    ----------
    mo_energy : list[np.ndarray]
        Molecular orbital energy levels.
    mo_coeff : list[np.ndarray]
        Molecular coefficients.
    eri_or_mol : np.ndarray or gto.Mole
        ERI that is recognized by ``pyscf.ao2mo.general``.
    mo_occ : list[np.ndarray]
        Molecular orbital occupation numbers.

    tol_e : float
        Threshold of ring-CCD energy difference while in DIIS update.
    tol_amp : float
        Threshold of L2 norm of ring-CCD amplitude while in DIIS update.
    max_cycle : int
        Maximum iteration of ring-CCD iteration.
    diis_start : int
        Start iteration number of DIIS.
    diis_space : int
        Space of DIIS.
    verbose : int
        Verbose level for PySCF.
    """
    log = lib.logger.new_logger(verbose=verbose)
    # log.warn("Conventional integral of MP2 is not recommended!\n"
    #          "Use density fitting approximation is recommended.")
    log.info(f"[INFO] Ring-CCD iteration, tol_e {tol_e:9.4e}, tol_amp {tol_amp:9.4e}, max_cycle {max_cycle:3d}")

    # preparation
    mask_occ = [mo_occ[σ] != 0 for σ in (α, β)]
    mask_vir = [mo_occ[σ] == 0 for σ in (α, β)]
    eo = [mo_energy[σ][mask_occ[σ]] for σ in (α, β)]
    ev = [mo_energy[σ][mask_vir[σ]] for σ in (α, β)]
    Co = [mo_coeff[σ][:, mask_occ[σ]] for σ in (α, β)]
    Cv = [mo_coeff[σ][:, mask_vir[σ]] for σ in (α, β)]
    nocc = tuple([mask_occ[σ].sum() for σ in (α, β)])
    nvir = tuple([mask_vir[σ].sum() for σ in (α, β)])

    # ao2mo preparation
    log.info("[INFO] Start ao2mo")
    g_ovov = [
        ao2mo.general(eri_or_mol, (Co[σ], Cv[σ], Co[ς], Cv[ς])).reshape(nocc[σ], nvir[σ], nocc[ς], nvir[ς])
        for (σ, ς) in ((α, α), (α, β), (β, β))]
    D_ovov = [
        lib.direct_sum("i - a + j - b -> iajb", eo[σ], ev[σ], eo[ς], ev[ς])
        for (σ, ς) in ((α, α), (α, β), (β, β))]
    log.info("[INFO] Finish ao2mo")

    # convert to dim_ov
    dim_ov = [nocc[σ] * nvir[σ] for σ in (α, β)]
    B = [g_ovov[σς].reshape(dim_ov[σ], dim_ov[ς]) for (σ, ς, σς) in ((α, α, αα), (α, β, αβ), (β, β, ββ))]
    D = [D_ovov[σς].reshape(dim_ov[σ], dim_ov[ς]) for (σ, ς, σς) in ((α, α, αα), (α, β, αβ), (β, β, ββ))]

    def update_T(T, B, D):
        B_αα, B_αβ, B_ββ = B
        D_αα, D_αβ, D_ββ = D
        T_αα, T_αβ, T_ββ = T
        B_βα, D_βα, T_βα = B_αβ.T, D_αβ.T, T_αβ.T
        BT_αα = B_αα @ T_αα + B_αβ @ T_βα
        BT_αβ = B_αα @ T_αβ + B_αβ @ T_ββ
        BT_βα = B_βα @ T_αα + B_ββ @ T_βα
        BT_ββ = B_βα @ T_αβ + B_ββ @ T_ββ
        # T_new_αα = - 1 / D_αα * (
        #     + B_αα - B_αα @ T_αα - B_αβ @ T_βα - T_αα @ B_αα - T_αβ @ B_βα
        #     + T_αα @ B_αα @ T_αα + T_αα @ B_αβ @ T_βα + T_αβ @ B_βα @ T_αα + T_αβ @ B_ββ @ T_βα)
        # T_new_αβ = - 1 / D_αβ * (
        #     + B_αβ - B_αα @ T_αβ - B_αβ @ T_ββ - T_αα @ B_αβ - T_αβ @ B_ββ
        #     + T_αα @ B_αα @ T_αβ + T_αα @ B_αβ @ T_ββ + T_αβ @ B_βα @ T_αβ + T_αβ @ B_ββ @ T_ββ)
        # T_new_ββ = - 1 / D_ββ * (
        #     + B_ββ - B_ββ @ T_ββ - B_βα @ T_αβ - T_ββ @ B_ββ - T_βα @ B_αβ
        #     + T_ββ @ B_ββ @ T_ββ + T_ββ @ B_βα @ T_αβ + T_βα @ B_αβ @ T_ββ + T_βα @ B_αα @ T_αβ)
        T_new_αα = - 1 / D_αα * (B_αα - BT_αα - BT_αα.T + T_αα @ BT_αα + T_αβ @ BT_βα)
        T_new_αβ = - 1 / D_αβ * (B_αβ - BT_αβ - BT_βα.T + T_αα @ BT_αβ + T_αβ @ BT_ββ)
        T_new_ββ = - 1 / D_ββ * (B_ββ - BT_ββ - BT_ββ.T + T_βα @ BT_αβ + T_ββ @ BT_ββ)
        return [T_new_αα, T_new_αβ, T_new_ββ]

    # begin diis
    T_new = [np.zeros_like(B[σς]) for σς in (αα, αβ, ββ)]
    eng_os = eng_ss = eng_drpa = 0
    diis = [lib.diis.DIIS() for _ in (αα, αβ, ββ)]
    for σς in (αα, αβ, ββ):
        diis[σς].space = diis_space
    converged = False
    for epoch in range(max_cycle):
        T_old, T_new = T_new, update_T(T_new, B, D)
        eng_old = eng_drpa
        if epoch > diis_start:
            T_new = [diis[σς].update(T_new[σς]) for σς in (αα, αβ, ββ)]
        eng_os = - np.einsum("AB, AB -> ", T_new[αβ], B[αβ])
        eng_ss = (
            - 0.5 * np.einsum("AB, AB -> ", T_new[αα], B[αα])
            - 0.5 * np.einsum("AB, AB -> ", T_new[ββ], B[ββ]))
        eng_drpa = eng_os + eng_ss
        err_amp = [np.linalg.norm(T_new[σς] - T_old[σς]) for σς in (αα, αβ, ββ)]
        err_e = abs(eng_drpa - eng_old)
        log.info(
            f"[INFO] Ring-CCD energy in iter {epoch:3d}: {eng_drpa:20.12f}, "
            f"amplitude L2 error {err_amp}, eng_err {err_e:9.4e}")
        if max(err_amp) < tol_amp and err_e < tol_e:
            converged = True
            log.info("[INFO] Ring-CCD amplitude converges.")
            break
    if not converged:
        log.warn(f"Ring-CCD not converged in {max_cycle} iterations!")

    results = dict()
    results["eng_corr_RING_CCD_OS"] = eng_os
    results["eng_corr_RING_CCD_SS"] = eng_ss
    results["eng_corr_RING_CCD"] = eng_drpa
    results["converged_RING_CCD"] = converged
    log.info(f"[RESULT] Energy corr ring-CCD of same-spin: {eng_os  :18.10f}")
    log.info(f"[RESULT] Energy corr ring-CCD of oppo-spin: {eng_ss  :18.10f}")
    log.info(f"[RESULT] Energy corr ring-CCD of total    : {eng_drpa:18.10f}")
    return results


class URingCCDConv(RRingCCDConv):

    @property
    def restricted(self):
        return False

    def driver_eng_ring_ccd(self, **kwargs):
        mask = self.get_frozen_mask()
        mol = self.mol
        mo_coeff_act = [self.mo_coeff[σ][:, mask[σ]] for σ in (α, β)]
        mo_energy_act = [self.mo_energy[σ][mask[σ]] for σ in (α, β)]
        mo_occ_act = [self.mo_occ[σ][mask[σ]] for σ in (α, β)]
        # eri generator
        eri_or_mol = self.scf._eri if self.omega == 0 else mol
        eri_or_mol = eri_or_mol if eri_or_mol is not None else mol
        results = self.kernel_energy_ring_ccd(
            mo_energy=mo_energy_act,
            mo_coeff=mo_coeff_act,
            eri_or_mol=eri_or_mol,
            mo_occ=mo_occ_act,
            tol_e=self.conv_tol,
            tol_amp=self.conv_tol_amp,
            max_cycle=self.max_cycle,
            diis_start=self.diis_start,
            diis_space=self.diis_space,
            verbose=self.verbose
        )
        # pad omega
        results = {pad_omega(key, self.omega): val for (key, val) in results.items()}
        self.results.update(results)
        return results

    kernel_energy_ring_ccd = staticmethod(kernel_energy_uring_ccd_conv)
    kernel = driver_eng_ring_ccd


if __name__ == '__main__':
    def main_1():
        from pyscf import gto, scf
        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G").build()
        mf = scf.RHF(mol).run()
        mf_r = RRingCCDConv(mf).run()
        print(mf_r.results)

        mf = mf.to_uhf()
        mf_u = URingCCDConv(mf).run()
        print(mf_u.results)

        mol = gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="6-31G", charge=1, spin=1).build()
        mf = scf.UHF(mol).run()
        mf_u = URingCCDConv(mf).run(frozen=[[0], [1, 2]])
        print(mf_u.results)

    main_1()
