from pyscf.dh.util import pylibxc
import pyscf.dh.util.pylibxc.flags
from pyscf.dh.util.pylibxc import LibXCFunctional

from pyscf import dft
from pyscf.dft import xc_deriv
import numpy as np
import itertools


def eval_xc_eff_ssr_generator(name_code, name_fr, name_sr, omega=0.7, cutoff=1e-13):
    """ Generator for function of scaled short-range DFT integrals.

    Parameters
    ----------
    name_code : str
        xc functional to be scaled.
    name_fr : str
        Full-range functional of the scale. Must be LDA.
    name_sr : str
        Short-range functional of the scale. Must be LDA.
    omega : float
        Range-separate parameter.
    cutoff : float
        Cutoff of exc_fr. If grid value of exc_fr is too small, then the returned exc on this grid will set to be zero.

    Returns
    -------
    callable
        A function shares the same function signature with dft.numint.NumInt.eval_xc_eff.
    """

    # original numint object
    ni = dft.numint.NumInt()
    # currently only LDA is supported
    assert ni._xc_type(name_fr) == "LDA"
    assert ni._xc_type(name_sr) == "LDA"
    # the xc type concern in this scaled short-range term
    xc_type_code = ni._xc_type(name_code)

    def eval_xc_eff(numint, xc_code, rho, *args, xctype=None, **kwargs):
        # for unrestricted methods, input rho can be tuple
        rho = np.asarray(rho)

        if xctype is None:
            xctype = numint._xc_type(xc_code)
        if xctype == 'LDA':
            spin_polarized = rho.ndim >= 2
        else:
            spin_polarized = rho.ndim == 3

        if not spin_polarized:
            return eval_xc_eff_spin_nonpolarized(numint, xc_code, rho, xctype=xctype, *args, **kwargs)
        else:
            return eval_xc_eff_spin_polarized(numint, xc_code, rho, xctype=xctype, *args, **kwargs)

    def eval_xc_eff_spin_nonpolarized(numint, xc_code, rho, deriv=1, omega=omega, xctype=None, *_args, **_kwargs):
        # the xc type of the whole numint object
        if xctype is None:
            xctype = numint._xc_type(xc_code)
        if xctype == "LDA":
            assert rho.ndim == 1
            rho0 = rho.copy()
        else:
            assert rho.ndim == 2
            rho0 = rho[0].copy()
        # if the xc type concern is LDA, then only extract the density grid (instead of its derivatives)
        if xc_type_code == "LDA":
            rho = rho0
        # evaluate xc grids by original numint object
        exc_code, vxc_code, fxc_code, kxc_code = ni.eval_xc_eff(name_code, rho, deriv=deriv)
        exc_fr, vxc_fr, fxc_fr, kxc_fr = ni.eval_xc_eff(name_fr, rho0, deriv=deriv)
        exc_sr, vxc_sr, fxc_sr, kxc_sr = ni.eval_xc_eff(name_sr, rho0, deriv=deriv, omega=omega)
        # avoid too small denominator (must set grid values to zero on these masks)
        mask = abs(exc_fr) < cutoff
        exc_fr[mask] = cutoff
        rho0[mask] = cutoff
        # handle exc, vxc, fxc, kxc
        ratio = exc_sr / exc_fr
        exc = vxc = fxc = kxc = None
        if deriv >= 0:
            exc = exc_code * ratio
            exc[mask] = 0
        if deriv >= 1:
            vxc = vxc_code.copy()
            vxc[0] = (
                + vxc_code[0] * exc_sr / exc_fr
                + exc_code * vxc_sr[0] / exc_fr
                - exc_code * exc_sr * vxc_fr[0] / exc_fr ** 2
            )
            vxc[1:] *= ratio
            vxc[:, mask] = 0
        if deriv >= 2:
            # derivative of (c * s / f), c -> code, s -> short-range, f -> full-range
            fxc = fxc_code.copy()
            r = rho0
            c, dc, ddc = exc_code * r, vxc_code[0], fxc_code[0, 0]
            pc, dpc = vxc_code[1:], fxc_code[0, 1:]
            s, ds, dds = exc_sr * r, vxc_sr[0], fxc_sr[0, 0]
            f, df, ddf = exc_fr * r, vxc_fr[0], fxc_fr[0, 0]
            fxc *= ratio
            fxc[0, 0] = (
                + (c * dds + 2 * dc * ds + ddc * s) / f
                - (c * s * ddf + 2 * c * ds * df + 2 * dc * s * df) / f**2
                + 2 * c * s * df**2 / f**3
            )
            fxc[0, 1:] = (
                + dpc * s / f
                + pc * ds / f
                - pc * s * df / f**2
            )
            fxc[1:, 0] = fxc[0, 1:]
            fxc[:, :, mask] = 0
        if deriv >= 3:
            # derivative of (c * s / f), c -> code, s -> short-range, f -> full-range
            kxc = kxc_code.copy()
            r = rho0
            c, dc, ddc, dddc = exc_code * r, vxc_code[0], fxc_code[0, 0], kxc_code[0, 0, 0]
            pc, dpc, ddpc = vxc_code[1:], fxc_code[0, 1:], kxc_code[0, 0, 1:]
            ppc, dppc = fxc_code[1:, 1:], kxc_code[0, 1:, 1:]
            s, ds, dds, ddds = exc_sr * r, vxc_sr[0], fxc_sr[0, 0], kxc_sr[0, 0, 0]
            f, df, ddf, dddf = exc_fr * r, vxc_fr[0], fxc_fr[0, 0], kxc_fr[0, 0, 0]

            kxc *= ratio
            kxc[0, 0, 0] = (
                + (c * ddds + 3 * dc * dds + 3 * ddc * ds + dddc * s) / f
                - (
                    + c * s * dddf + 3 * c * ds * ddf + 3 * c * dds * df + 3 * dc * s * ddf
                    + 6 * dc * ds * df + 3 * ddc * s * df) / f**2
                + 6 * (c * ds * df**2 + dc * s * df**2 + c * s * df * ddf) / f**3
                - 6 * c * s * df**3 / f**4
            )
            kxc[0, 0, 1:] = kxc[0, 1:, 0] = kxc[1:, 0, 0] = (
                + (pc * dds + 2 * dpc * ds + ddpc * s) / f
                - (pc * s * ddf + 2 * pc * ds * df + 2 * dpc * s * df) / f**2
                + 2 * pc * s * df**2 / f**3
            )
            kxc[0, 1:, 1:] = kxc[1:, 0, 1:] = kxc[1:, 1:, 0] = (
                + dppc * s / f
                + ppc * ds / f
                - ppc * s * df / f**2
            )
            kxc[:, :, :, mask] = 0
        return exc, vxc, fxc, kxc

    def eval_xc_eff_spin_polarized(numint, xc_code, rho, deriv=1, omega=omega, xctype=None, *_args, **_kwargs):
        # the xc type of the whole numint object
        if xctype is None:
            xctype = numint._xc_type(xc_code)
        if xctype == "LDA":
            assert rho.ndim == 2
            rho0 = rho.copy()
        else:
            assert rho.ndim == 3
            rho0 = rho[:, 0].copy()
        # if the xc type concern is LDA, then only extract the density grid (instead of its derivatives)
        if xc_type_code == "LDA":
            rho = rho0
        # evaluate xc grids by original numint object
        exc_code, vxc_code, fxc_code, kxc_code = ni.eval_xc_eff(name_code, rho, deriv=deriv)
        exc_fr, vxc_fr, fxc_fr, kxc_fr = ni.eval_xc_eff(name_fr, rho0, deriv=deriv)
        exc_sr, vxc_sr, fxc_sr, kxc_sr = ni.eval_xc_eff(name_sr, rho0, deriv=deriv, omega=omega)
        # avoid too small denominator (must set grid values to zero on these masks)
        mask = abs(exc_fr) < cutoff
        exc_fr[mask] = cutoff
        rho0[:, mask] = cutoff
        # handle exc, vxc, fxc, kxc
        ratio = exc_sr / exc_fr
        exc = vxc = fxc = kxc = None
        if deriv >= 0:
            exc = exc_code * ratio
            exc[mask] = 0
        if deriv >= 1:
            vxc = vxc_code.copy()
            vxc[:, 0] = (
                + vxc_code[:, 0] * exc_sr / exc_fr
                + exc_code * vxc_sr[:, 0] / exc_fr
                - exc_code * exc_sr * vxc_fr[:, 0] / exc_fr ** 2
            )
            vxc[:, 1:] *= ratio
            vxc[:, :, mask] = 0
        if deriv >= 2:
            # derivative of (c * s / f), c -> code, s -> short-range, f -> full-range
            fxc = fxc_code.copy()
            r = rho0.sum(axis=0)
            c, dc, ddc = exc_code * r, vxc_code[:, 0], fxc_code[:, 0, :, 0]
            pc, dpc = vxc_code[:, 1:], fxc_code[:, 0, :, 1:]
            s, ds, dds = exc_sr * r, vxc_sr[:, 0], fxc_sr[:, 0, :, 0]
            f, df, ddf = exc_fr * r, vxc_fr[:, 0], fxc_fr[:, 0, :, 0]
            fxc *= ratio
            for s1, s2 in itertools.product((0, 1), (0, 1)):
                # handle actual derivatives, iterate by spin
                fxc[s1, 0, s2, 0] = (
                    + (c * dds[s1, s2] + dc[s1] * ds[s2] + dc[s2] * ds[s1] + ddc[s1, s2] * s) / f
                    - (
                        + c * s * ddf[s1, s2]
                        + c * ds[s1] * df[s2] + c * ds[s2] * df[s1]
                        + dc[s1] * s * df[s2] + dc[s2] * s * df[s1]
                    ) / f**2
                    + 2 * c * s * df[s1] * df[s2] / f**3
                )
                fxc[s1, 0, s2, 1:] = fxc[s2, 1:, s1, 0] = (
                    + dpc[s1, s2] * s / f
                    + pc[s2] * ds[s1] / f
                    - pc[s2] * s * df[s1] / f**2
                )
            fxc[:, :, :, :, mask] = 0
        if deriv >= 3:
            # derivative of (c * s / f), c -> code, s -> short-range, f -> full-range
            kxc = kxc_code.copy()
            r = rho0.sum(axis=0)
            c, dc, ddc, dddc = exc_code * r, vxc_code[:, 0], fxc_code[:, 0, :, 0], kxc_code[:, 0, :, 0, :, 0]
            pc, dpc, ddpc = vxc_code[:, 1:], fxc_code[:, 0, :, 1:], kxc_code[:, 0, :, 0, :, 1:]
            ppc, dppc = fxc_code[:, 1:, :, 1:].swapaxes(1, 2), kxc_code[:, 0, :, 1:, :, 1:].swapaxes(2, 3)
            s, ds, dds, ddds = exc_sr * r, vxc_sr[:, 0], fxc_sr[:, 0, :, 0], kxc_sr[:, 0, :, 0, :, 0]
            f, df, ddf, dddf = exc_fr * r, vxc_fr[:, 0], fxc_fr[:, 0, :, 0], kxc_fr[:, 0, :, 0, :, 0]

            kxc *= ratio

            for s1, s2, s3 in itertools.product((0, 1), (0, 1), (0, 1)):
                kxc[s1, 0, s2, 0, s3, 0] = (
                    + (
                        + c * ddds[s1, s2, s3]
                        + dc[s1] * dds[s2, s3] + dc[s2] * dds[s3, s1] + dc[s3] * dds[s1, s2]
                        + ddc[s1, s2] * ds[s3] + ddc[s2, s3] * ds[s1] + ddc[s3, s1] * ds[s2]
                        + dddc[s1, s2, s3] * s) / f
                    - (
                        + c * s * dddf[s1, s2, s3]
                        + c * ds[s1] * ddf[s2, s3] + c * ds[s2] * ddf[s3, s1] + c * ds[s3] * ddf[s1, s2]
                        + c * dds[s1, s2] * df[s3] + c * dds[s2, s3] * df[s1] + c * dds[s3, s1] * df[s2]
                        + dc[s1] * s * ddf[s2, s3] + dc[s2] * s * ddf[s3, s1] + dc[s3] * s * ddf[s1, s2]
                        + dc[s1] * ds[s2] * df[s3] + dc[s1] * ds[s3] * df[s2] + dc[s2] * ds[s1] * df[s3]
                        + dc[s2] * ds[s3] * df[s1] + dc[s3] * ds[s1] * df[s2] + dc[s3] * ds[s2] * df[s1]
                        + ddc[s1, s2] * s * df[s3] + ddc[s2, s3] * s * df[s1] + ddc[s3, s1] * s * df[s2]) / f**2
                    + 2 * (
                        + c * (ds[s1] * df[s2] * df[s3] + ds[s2] * df[s3] * df[s1] + ds[s3] * df[s1] * df[s2])
                        + s * (dc[s1] * df[s2] * df[s3] + dc[s2] * df[s3] * df[s1] + dc[s3] * df[s1] * df[s2])
                        + c * s * (df[s1] * ddf[s2, s3] + df[s2] * ddf[s3, s1] + df[s3] * ddf[s1, s2])) / f**3
                    - 6 * c * s * df[s1] * df[s2] * df[s3] / f**4
                )
                kxc_p3 = (
                    + (pc[s3] * dds[s1, s2] + dpc[s1, s3] * ds[s2] + dpc[s2, s3] * ds[s1] + ddpc[s1, s2, s3] * s) / f
                    - (
                        + pc[s3] * s * ddf[s1, s2]
                        + pc[s3] * ds[s1] * df[s2] + pc[s3] * ds[s2] * df[s1]
                        + dpc[s1, s3] * s * df[s2] + dpc[s2, s3] * s * df[s1]) / f**2
                    + 2 * pc[s3] * s * df[s1] * df[s2] / f**3
                )
                kxc[s1, 0, s2, 0, s3, 1:] = kxc[s2, 0, s1, 0, s3, 1:] = kxc_p3
                kxc[s1, 0, s3, 1:, s2, 0] = kxc[s2, 0, s3, 1:, s1, 0] = kxc_p3
                kxc[s3, 1:, s1, 0, s2, 0] = kxc[s3, 1:, s2, 0, s1, 0] = kxc_p3
                kxc_p2p3 = (
                    + dppc[s1, s2, s3] * s / f
                    + ppc[s2, s3] * ds[s1] / f
                    - ppc[s2, s3] * s * df[s1] / f**2
                )
                kxc_p3p2 = kxc_p2p3.swapaxes(0, 1)
                kxc[s1, 0, s2, 1:, s3, 1:] = kxc[s2, 1:, s1, 0, s3, 1:] = kxc[s2, 1:, s3, 1:, s1, 0] = kxc_p2p3
                kxc[s1, 0, s3, 1:, s2, 1:] = kxc[s3, 1:, s1, 0, s2, 1:] = kxc[s3, 1:, s2, 1:, s1, 0] = kxc_p3p2
            kxc[:, :, :, :, :, :, mask] = 0
        return exc, vxc, fxc, kxc
    return eval_xc_eff


def eval_xc_eff_ext_param_generator(name_code, ext_param):
    """ Evaluate xc_eff grids with external parameters.

    Note that `name_code` should be exactly the same to that defined in LibXC
    (instead of simplificated strings).
    """

    # re-generate external parameter list that could feed to pylibxc
    # the following func_tmp is only temporary object that give external parameters;
    #   to actually evaluate xc, spin is also required.
    func_tmp = LibXCFunctional(name_code, spin=1)
    dict_param = {key: val for key, val in zip(func_tmp.get_ext_param_names(), func_tmp.get_ext_param_default_values())}
    key_mapping = {key.lower(): key for key in dict_param.keys()}
    for key, val in ext_param.items():
        if key.lower() in key_mapping:
            key = key_mapping[key.lower()]
            dict_param[key] = val
        else:
            raise ValueError(f"key {key} seems not in {name_code} external parameter: {dict_param.values()}")
    ext_param_use = [dict_param[key] for key in func_tmp.get_ext_param_names()]
    func_tmp.set_ext_params(ext_param_use)

    # determine xctype
    xctype_id = func_tmp.get_family()
    if xctype_id == pylibxc.flags.XC_FAMILY_LDA:
        xctype_use = "LDA"
    elif xctype_id == pylibxc.flags.XC_FAMILY_GGA:
        xctype_use = "GGA"
    elif xctype_id == pylibxc.flags.XC_FAMILY_MGGA:
        xctype_use = "MGGA"
    else:
        raise ValueError("xc family is not recognized (UNKNOWN, LCA, OEP not accepted).")

    # determine hybrid coefficients
    omega, alpha, beta = 0, 0, 0
    try:
        omega, alpha, beta = func_tmp.get_cam_coef()
    except ValueError:
        try:
            alpha = func_tmp.get_hyb_exx_coef()
        except ValueError:
            pass  # not hybrid or range-separate functional

    # main function of eval_xc_eff
    def eval_xc_eff(numint, xc_code, rho, *args, xctype=None, deriv=1, **kwargs):
        # we do not use xc_code and xctype keywords
        del xc_code, xctype
        xctype = xctype_use

        # for unrestricted methods, input rho can be tuple
        rhop = np.asarray(rho)

        if xctype == 'LDA':
            spin_polarized = rhop.ndim >= 2
        else:
            spin_polarized = rhop.ndim == 3

        if spin_polarized:
            assert rhop.shape[0] == 2
            if rhop.shape[1] == 5:  # MGGA
                ngrids = rhop.shape[2]
                rhop = np.empty((2, 6, ngrids))
                rhop[0,:4] = rho[0][:4]
                rhop[1,:4] = rho[1][:4]
                rhop[:,4] = 0
                rhop[0,5] = rho[0][4]
                rhop[1,5] = rho[1][4]
        else:
            if rhop.shape[0] == 5:  # MGGA
                ngrids = rho.shape[1]
                rhop = np.empty((6, ngrids))
                rhop[:4] = rho[:4]
                rhop[4] = 0
                rhop[5] = rho[4]

        # pylibxc eval_xc
        spin = 2 if spin_polarized else 1  # pylibxc convention
        func = LibXCFunctional(name_code, spin=spin)
        func.set_ext_params(ext_param_use)
        exc, vxc, fxc, kxc = eval_xc(func, rhop, spin_polarized, deriv=deriv, *args, **kwargs)

        # final transformation
        spin = 1 if spin_polarized else 0  # pylibxc convention
        if deriv > 2:
            kxc = xc_deriv.transform_kxc(rhop, fxc, kxc, xctype, spin)
        if deriv > 1:
            fxc = xc_deriv.transform_fxc(rhop, vxc, fxc, xctype, spin)
        if deriv > 0:
            vxc = xc_deriv.transform_vxc(rhop, vxc, xctype, spin)

        return exc, vxc, fxc, kxc

    def get_sigma(nabla_rho):
        return np.einsum("tg, tg -> g", nabla_rho, nabla_rho, optimize=True)

    def eval_xc(func, rho, spin_polarized, deriv=1, *_args, **_kwargs):

        # preparation
        xctype = xctype_use

        if not spin_polarized:
            if xctype == "LDA":
                inp = {"rho": rho}
            elif len(rho) == 4:
                inp = {"rho": rho[0], "sigma": get_sigma(rho[1:4])}
            elif len(rho) == 5:
                inp = {"rho": rho[0], "sigma": get_sigma(rho[1:4]), "tau": rho[4]}
            elif len(rho) == 6:
                inp = {"rho": rho[0], "sigma": get_sigma(rho[1:4]), "tau": rho[5]}
            else:
                assert False
        else:
            if xctype == "LDA":
                inp = {"rho": rho}
            elif len(rho[0]) == 4:
                inp = {"rho": np.array(rho[:, 0], order='C'),
                       "sigma": np.array([get_sigma(rho[0, 1:4]), get_sigma(rho[1, 1:4])])}
            elif len(rho[0]) == 5:
                inp = {"rho": np.array(rho[:, 0], order='C'),
                       "sigma": np.array([get_sigma(rho[0, 1:4]), get_sigma(rho[1, 1:4])]),
                       "tau": np.array(rho[:, 4], order='C')}
            elif len(rho[0]) == 6:
                inp = {"rho": np.array(rho[:, 0], order='C'),
                       "sigma": np.array([get_sigma(rho[0, 1:4]), get_sigma(rho[1, 1:4])]),
                       "tau": np.array(rho[:, 5], order='C')}
            else:
                assert False

        # computation
        comput_kwargs = {
            "do_exc": True,
            "do_vxc": deriv > 0,
            "do_fxc": deriv > 1,
            "do_kxc": deriv > 2,
            "do_lxc": False}
        output = func.compute(inp, **comput_kwargs)

        # result handling
        exc, vxc, fxc, kxc = None, None, None, None
        exc = output["zk"].reshape(-1)
        if xctype == "LDA":
            if deriv > 0:
                vxc = [output.get("vrho")]
            if deriv > 1:
                fxc = [output.get("v2rho2")]
            if deriv > 2:
                kxc = [output.get("v3rho3")]
        elif xctype == "GGA":
            if deriv > 0:
                vxc = [output.get(key) for key in "vrho, vsigma".split(", ")]
            if deriv > 1:
                fxc = [output.get(key) for key in "v2rho2, v2rhosigma, v2sigma2".split(", ")]
            if deriv > 2:
                kxc = [output.get(key) for key in "v3rho3, v3rho2sigma, v3rhosigma2, v3sigma3".split(", ")]
        elif xctype == "MGGA":
            if deriv > 0:
                vxc = [output.get(key) for key in "vrho, vsigma, vlapl, vtau".split(", ")]
            if deriv > 1:
                fxc = [output.get(key) for key in
                       "v2rho2, v2rhosigma, v2sigma2, v2lapl2, v2tau2, "
                       "v2rholapl, v2rhotau, v2lapltau, v2sigmalapl, v2sigmatau".split(", ")]
            if deriv > 2:
                kxc = [output.get(key) for key in
                       "v3rho3, v3rho2sigma, v3rhosigma2, v3sigma3, v3rho2lapl, v3rho2tau, "
                       "v3rhosigmalapl, v3rhosigmatau, v3rholapl2, v3rholapltau, v3rhotau2, "
                       "v3sigma2lapl, v3sigma2tau, v3sigmalapl2, v3sigmalapltau, v3sigmatau2, "
                       "v3lapl3, v3lapl2tau, v3lapltau2, v3tau3".split(", ")]

        return exc, vxc, fxc, kxc

    return eval_xc_eff, (omega, alpha, beta)
