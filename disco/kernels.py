import cupy as cp
from cupyx import jit

from disco.constants import RK45Coeffs


@jit.rawkernel()
def rhs_kernel(
    y,
    t,
    Bx_arr,
    By_arr,
    Bz_arr,
    B_arr,
    Ex_arr,
    Ey_arr,
    Ez_arr,
    x_axis,
    y_axis,
    z_axis,
    t_axis,
    nx,
    ny,
    nz,
    nt,
    r_inner,
    dBdx_arr,
    dBdy_arr,
    dBdz_arr,
    dBxdy_arr,
    dBxdz_arr,
    dBydx_arr,
    dBydz_arr,
    dBzdx_arr,
    dBzdy_arr,
    dydt_arr,
):
    """[CUPY KERNEL] implements RHS of ODE.

    Code adapted from Fortran. Uses gyro-averaged equations of motion
    developed by Brizzard and Chan (Phys. Plasmas 6, 4553, 1999),

    Writes output to dydt[idx, :]
    """
    idx = jit.blockDim.x * jit.blockIdx.x + jit.threadIdx.x
    oob = False

    # Out of bounds check
    if idx < y.shape[0]:
        oob = (
            y[idx, 0] < x_axis[0]
            or y[idx, 0] > x_axis[nx - 1]
            or y[idx, 1] < y_axis[0]
            or y[idx, 1] > y_axis[ny - 1]
            or y[idx, 2] < z_axis[0]
            or y[idx, 2] > z_axis[nz - 1]
            or t[idx] < t_axis[0]
            or t[idx] > t_axis[nt - 1]
            or (y[idx, 0] ** 2 + y[idx, 1] ** 2 + y[idx, 2] ** 2) ** 0.5 < r_inner[idx]
        )

    if idx < y.shape[0] and not oob:
        # Pull variables out of arrays
        ppar = y[idx, 3]
        M = y[idx, 4]

        Bx = Bx_arr[idx]
        By = By_arr[idx]
        Bz = Bz_arr[idx]
        B = B_arr[idx]

        Ex = Ex_arr[idx]
        Ey = Ey_arr[idx]
        Ez = Ez_arr[idx]

        dBdx = dBdx_arr[idx]
        dBdy = dBdy_arr[idx]
        dBdz = dBdz_arr[idx]

        dBxdy = dBxdy_arr[idx]
        dBxdz = dBxdz_arr[idx]
        dBydx = dBydx_arr[idx]
        dBydz = dBydz_arr[idx]
        dBzdx = dBzdx_arr[idx]
        dBzdy = dBzdy_arr[idx]

        # gyro-averaged equations of motion developed by Brizzard and Chan (Phys.
        # Plasmas 6, 4553, 1999),
        gamma = cp.sqrt(1 + 2 * B * M + ppar**2)
        pparl_B = ppar / B
        pparl_B2 = pparl_B / B
        Bxstar = Bx + pparl_B * (dBzdy - dBydz) - pparl_B2 * (Bz * dBdy - By * dBdz)
        Bystar = By + pparl_B * (dBxdz - dBzdx) - pparl_B2 * (Bx * dBdz - Bz * dBdx)
        Bzstar = Bz + pparl_B * (dBydx - dBxdy) - pparl_B2 * (By * dBdx - Bx * dBdy)
        Bsparl = (Bx * Bxstar + By * Bystar + Bz * Bzstar) / B
        gamma_Bsparl = 1 / (gamma * Bsparl)
        pparl_gamma_Bsparl = ppar * gamma_Bsparl
        B_Bsparl = 1 / (B * Bsparl)
        M_gamma_Bsparl = M * gamma_Bsparl
        M_gamma_B_Bsparl = M_gamma_Bsparl / B

        # 	  ...now calculate dynamic quantities...
        dydt_arr[idx, 0] = (
            pparl_gamma_Bsparl * Bxstar  # curv drft + parl
            + M_gamma_B_Bsparl * (By * dBdz - Bz * dBdy)  # gradB drft
            + B_Bsparl * (Ey * Bz - Ez * By)  # ExB drft
        )
        dydt_arr[idx, 1] = (
            pparl_gamma_Bsparl * Bystar  # curv drft + parl
            + M_gamma_B_Bsparl * (Bz * dBdx - Bx * dBdz)  # gradB drft
            + B_Bsparl * (Ez * Bx - Ex * Bz)  # ExB drft
        )
        dydt_arr[idx, 2] = (
            pparl_gamma_Bsparl * Bzstar  # curv drft + parl
            + M_gamma_B_Bsparl * (Bx * dBdy - By * dBdx)  # gradB drft
            + B_Bsparl * (Ex * By - Ey * Bx)  # ExB drft
        )
        dydt_arr[idx, 3] = (
            Bxstar * Ex + Bystar * Ey + Bzstar * Ez
        ) / Bsparl - M_gamma_Bsparl * (  # parl force
            Bxstar * dBdx + Bystar * dBdy + Bzstar * dBdz
        )


@jit.rawkernel()
def multi_interp_kernel(
    nx,
    ny,
    nz,
    nt,
    nxy,
    nxyz,
    nttl,
    ix,
    iy,
    iz,
    it,
    t,
    y,
    b0,
    r_inner,
    t_axis,
    x_axis,
    y_axis,
    z_axis,
    bxvec,
    byvec,
    bzvec,
    exvec,
    eyvec,
    ezvec,
    bx,
    by,
    bz,
    ex,
    ey,
    ez,
    dbxdx,
    dbxdy,
    dbxdz,
    dbydx,
    dbydy,
    dbydz,
    dbzdx,
    dbzdy,
    dbzdz,
    b,
    dbdx,
    dbdy,
    dbdz,
):
    """[CUPY KERNEL] Four dimensional interpolation with finite differencing
    and reused weights.

    Code adapted from fortran.
    """
    idx = jit.blockDim.x * jit.blockIdx.x + jit.threadIdx.x
    oob = False

    # Out of bounds check
    if idx < y.shape[0]:
        oob = (
            y[idx, 0] < x_axis[0]
            or y[idx, 0] > x_axis[nx - 1]
            or y[idx, 1] < y_axis[0]
            or y[idx, 1] > y_axis[ny - 1]
            or y[idx, 2] < z_axis[0]
            or y[idx, 2] > z_axis[nz - 1]
            or t[idx] < t_axis[0]
            or t[idx] > t_axis[nt - 1]
            or (y[idx, 0] ** 2 + y[idx, 1] ** 2 + y[idx, 2] ** 2) ** 0.5 < r_inner[idx]
        )

    # Main body of the kernel
    if idx < y.shape[0] and not oob:
        dx = x_axis[ix[idx] + 1] - x_axis[ix[idx]]
        dy = y_axis[iy[idx] + 1] - y_axis[iy[idx]]
        dz = z_axis[iz[idx] + 1] - z_axis[iz[idx]]
        dt = t_axis[it[idx] + 1] - t_axis[it[idx]]

        # ...determine memory location corresponding to ix,iy,iz,it
        # 	and ix+1,iy+1,iz+1,it+1...
        jjjj = ix[idx] + iy[idx] * nx + iz[idx] * nxy + it[idx] * nxyz + 1
        ijjj = jjjj - 1
        jijj = jjjj - nx
        iijj = jijj - 1
        jjij = jjjj - nxy
        ijij = jjij - 1
        jiij = jijj - nxy
        iiij = jiij - 1
        jjji = jjjj - nxyz
        ijji = ijjj - nxyz
        jiji = jijj - nxyz
        iiji = iijj - nxyz
        jjii = jjij - nxyz
        ijii = ijij - nxyz
        jiii = jiij - nxyz
        iiii = iiij - nxyz

        # ...calculate weighting factors...
        w1 = cp.abs(y[idx, 0] - x_axis[ix[idx]]) / dx
        w1m = 1.0 - w1
        w2 = cp.abs(y[idx, 1] - y_axis[iy[idx]]) / dy
        w2m = 1.0 - w2
        w3 = cp.abs(y[idx, 2] - z_axis[iz[idx]]) / dz
        w3m = 1.0 - w3
        w4 = cp.abs(t[idx] - t_axis[it[idx]]) / dt
        w4m = 1.0 - w4

        w1m2m = w1m * w2m
        w12m = w1 * w2m
        w12 = w1 * w2
        w1m2 = w1m * w2

        w1m2m3m = w1m2m * w3m
        w12m3m = w12m * w3m
        w123m = w12 * w3m
        w1m23m = w1m2 * w3m

        w1m2m3 = w1m2m * w3
        w12m3 = w12m * w3
        w123 = w12 * w3
        w1m23 = w1m2 * w3

        ww01 = w1m2m3m * w4m
        ww02 = w12m3m * w4m
        ww03 = w123m * w4m
        ww04 = w1m23m * w4m
        ww05 = w1m2m3 * w4m
        ww06 = w12m3 * w4m
        ww07 = w123 * w4m
        ww08 = w1m23 * w4m
        ww09 = w1m2m3m * w4
        ww10 = w12m3m * w4
        ww11 = w123m * w4
        ww12 = w1m23m * w4
        ww13 = w1m2m3 * w4
        ww14 = w12m3 * w4
        ww15 = w123 * w4
        ww16 = w1m23 * w4

        # ..define some factors often repeated in the interpolations...
        r = (y[idx, 0] ** 2 + y[idx, 1] ** 2 + y[idx, 2] ** 2) ** (0.5)
        r2 = r * r
        bfac1 = 3.0 * b0[idx] / r2 / r2 / r
        bfac2 = 5.0 * bfac1 / r2

        # ...interpolate field components...
        bx[idx] = (
            bxvec[iiii] * ww01
            + bxvec[jiii] * ww02
            + bxvec[jjii] * ww03
            + bxvec[ijii] * ww04
            + bxvec[iiji] * ww05
            + bxvec[jiji] * ww06
            + bxvec[jjji] * ww07
            + bxvec[ijji] * ww08
            + bxvec[iiij] * ww09
            + bxvec[jiij] * ww10
            + bxvec[jjij] * ww11
            + bxvec[ijij] * ww12
            + bxvec[iijj] * ww13
            + bxvec[jijj] * ww14
            + bxvec[jjjj] * ww15
            + bxvec[ijjj] * ww16
            - bfac1 * y[idx, 0] * y[idx, 2]
        )

        by[idx] = (
            byvec[iiii] * ww01
            + byvec[jiii] * ww02
            + byvec[jjii] * ww03
            + byvec[ijii] * ww04
            + byvec[iiji] * ww05
            + byvec[jiji] * ww06
            + byvec[jjji] * ww07
            + byvec[ijji] * ww08
            + byvec[iiij] * ww09
            + byvec[jiij] * ww10
            + byvec[jjij] * ww11
            + byvec[ijij] * ww12
            + byvec[iijj] * ww13
            + byvec[jijj] * ww14
            + byvec[jjjj] * ww15
            + byvec[ijjj] * ww16
            - bfac1 * y[idx, 1] * y[idx, 2]
        )

        bz[idx] = (
            bzvec[iiii] * ww01
            + bzvec[jiii] * ww02
            + bzvec[jjii] * ww03
            + bzvec[ijii] * ww04
            + bzvec[iiji] * ww05
            + bzvec[jiji] * ww06
            + bzvec[jjji] * ww07
            + bzvec[ijji] * ww08
            + bzvec[iiij] * ww09
            + bzvec[jiij] * ww10
            + bzvec[jjij] * ww11
            + bzvec[ijij] * ww12
            + bzvec[iijj] * ww13
            + bzvec[jijj] * ww14
            + bzvec[jjjj] * ww15
            + bzvec[ijjj] * ww16
            - bfac1 * y[idx, 2] * y[idx, 2]
            + b0[idx] / r2 / r
        )

        ex[idx] = (
            exvec[iiii] * ww01
            + exvec[jiii] * ww02
            + exvec[jjii] * ww03
            + exvec[ijii] * ww04
            + exvec[iiji] * ww05
            + exvec[jiji] * ww06
            + exvec[jjji] * ww07
            + exvec[ijji] * ww08
            + exvec[iiij] * ww09
            + exvec[jiij] * ww10
            + exvec[jjij] * ww11
            + exvec[ijij] * ww12
            + exvec[iijj] * ww13
            + exvec[jijj] * ww14
            + exvec[jjjj] * ww15
            + exvec[ijjj] * ww16
        )

        ey[idx] = (
            eyvec[iiii] * ww01
            + eyvec[jiii] * ww02
            + eyvec[jjii] * ww03
            + eyvec[ijii] * ww04
            + eyvec[iiji] * ww05
            + eyvec[jiji] * ww06
            + eyvec[jjji] * ww07
            + eyvec[ijji] * ww08
            + eyvec[iiij] * ww09
            + eyvec[jiij] * ww10
            + eyvec[jjij] * ww11
            + eyvec[ijij] * ww12
            + eyvec[iijj] * ww13
            + eyvec[jijj] * ww14
            + eyvec[jjjj] * ww15
            + eyvec[ijjj] * ww16
        )

        ez[idx] = (
            ezvec[iiii] * ww01
            + ezvec[jiii] * ww02
            + ezvec[jjii] * ww03
            + ezvec[ijii] * ww04
            + ezvec[iiji] * ww05
            + ezvec[jiji] * ww06
            + ezvec[jjji] * ww07
            + ezvec[ijji] * ww08
            + ezvec[iiij] * ww09
            + ezvec[jiij] * ww10
            + ezvec[jjij] * ww11
            + ezvec[ijij] * ww12
            + ezvec[iijj] * ww13
            + ezvec[jijj] * ww14
            + ezvec[jjjj] * ww15
            + ezvec[ijjj] * ww16
        )

        # ...calculate btot and field derivatives to 1st order...
        # ...first form more intermediate weights...
        w2m3m4m = w2m * w3m * w4m
        w23m4m = w2 * w3m * w4m
        w2m34m = w2m * w3 * w4m
        w234m = w2 * w3 * w4m
        w2m3m4 = w2m * w3m * w4
        w23m4 = w2 * w3m * w4
        w2m34 = w2m * w3 * w4
        w234 = w2 * w3 * w4

        w1m3m4m = w1m * w3m * w4m
        w13m4m = w1 * w3m * w4m
        w1m34m = w1m * w3 * w4m
        w134m = w1 * w3 * w4m
        w1m3m4 = w1m * w3m * w4
        w13m4 = w1 * w3m * w4
        w1m34 = w1m * w3 * w4
        w134 = w1 * w3 * w4

        w1m2m4m = w1m2m * w4m
        w12m4m = w12m * w4m
        w1m24m = w1m2 * w4m
        w124m = w12 * w4m
        w1m2m4 = w1m2m * w4
        w12m4 = w12m * w4
        w1m24 = w1m2 * w4
        w124 = w12 * w4

        # ...calculate component derivatives...
        dbxdx[idx] = (
            (
                (bxvec[jiii] - bxvec[iiii]) * w2m3m4m
                + (bxvec[jjii] - bxvec[ijii]) * w23m4m
                + (bxvec[jiji] - bxvec[iiji]) * w2m34m
                + (bxvec[jjji] - bxvec[ijji]) * w234m
                + (bxvec[jiij] - bxvec[iiij]) * w2m3m4
                + (bxvec[jjij] - bxvec[ijij]) * w23m4
                + (bxvec[jijj] - bxvec[iijj]) * w2m34
                + (bxvec[jjjj] - bxvec[ijjj]) * w234
            )
            / dx
            - bfac1 * y[idx, 2]
            + bfac2 * y[idx, 0] * y[idx, 0] * y[idx, 2]
        )

        dbxdy[idx] = (
            (bxvec[ijii] - bxvec[iiii]) * w1m3m4m
            + (bxvec[jjii] - bxvec[jiii]) * w13m4m
            + (bxvec[ijji] - bxvec[iiji]) * w1m34m
            + (bxvec[jjji] - bxvec[jiji]) * w134m
            + (bxvec[ijij] - bxvec[iiij]) * w1m3m4
            + (bxvec[jjij] - bxvec[jiij]) * w13m4
            + (bxvec[ijjj] - bxvec[iijj]) * w1m34
            + (bxvec[jjjj] - bxvec[jijj]) * w134
        ) / dy + bfac2 * y[idx, 0] * y[idx, 1] * y[idx, 2]

        dbxdz[idx] = (
            (
                (bxvec[iiji] - bxvec[iiii]) * w1m2m4m
                + (bxvec[jiji] - bxvec[jiii]) * w12m4m
                + (bxvec[ijji] - bxvec[ijii]) * w1m24m
                + (bxvec[jjji] - bxvec[jjii]) * w124m
                + (bxvec[iijj] - bxvec[iiij]) * w1m2m4
                + (bxvec[jijj] - bxvec[jiij]) * w12m4
                + (bxvec[ijjj] - bxvec[ijij]) * w1m24
                + (bxvec[jjjj] - bxvec[jjij]) * w124
            )
            / dz
            - bfac1 * y[idx, 0]
            + bfac2 * y[idx, 0] * y[idx, 2] * y[idx, 2]
        )

        dbydx[idx] = (
            (byvec[jiii] - byvec[iiii]) * w2m3m4m
            + (byvec[jjii] - byvec[ijii]) * w23m4m
            + (byvec[jiji] - byvec[iiji]) * w2m34m
            + (byvec[jjji] - byvec[ijji]) * w234m
            + (byvec[jiij] - byvec[iiij]) * w2m3m4
            + (byvec[jjij] - byvec[ijij]) * w23m4
            + (byvec[jijj] - byvec[iijj]) * w2m34
            + (byvec[jjjj] - byvec[ijjj]) * w234
        ) / dx + bfac2 * y[idx, 1] * y[idx, 2] * y[idx, 0]

        dbydy[idx] = (
            (
                (byvec[ijii] - byvec[iiii]) * w1m3m4m
                + (byvec[jjii] - byvec[jiii]) * w13m4m
                + (byvec[ijji] - byvec[iiji]) * w1m34m
                + (byvec[jjji] - byvec[jiji]) * w134m
                + (byvec[ijij] - byvec[iiij]) * w1m3m4
                + (byvec[jjij] - byvec[jiij]) * w13m4
                + (byvec[ijjj] - byvec[iijj]) * w1m34
                + (byvec[jjjj] - byvec[jijj]) * w134
            )
            / dy
            - bfac1 * y[idx, 2]
            + bfac2 * y[idx, 1] * y[idx, 1] * y[idx, 2]
        )

        dbydz[idx] = (
            (
                (byvec[iiji] - byvec[iiii]) * w1m2m4m
                + (byvec[jiji] - byvec[jiii]) * w12m4m
                + (byvec[ijji] - byvec[ijii]) * w1m24m
                + (byvec[jjji] - byvec[jjii]) * w124m
                + (byvec[iijj] - byvec[iiij]) * w1m2m4
                + (byvec[jijj] - byvec[jiij]) * w12m4
                + (byvec[ijjj] - byvec[ijij]) * w1m24
                + (byvec[jjjj] - byvec[jjij]) * w124
            )
            / dz
            - bfac1 * y[idx, 1]
            + bfac2 * y[idx, 2] * y[idx, 2] * y[idx, 1]
        )

        dbzdx[idx] = (
            (
                (bzvec[jiii] - bzvec[iiii]) * w2m3m4m
                + (bzvec[jjii] - bzvec[ijii]) * w23m4m
                + (bzvec[jiji] - bzvec[iiji]) * w2m34m
                + (bzvec[jjji] - bzvec[ijji]) * w234m
                + (bzvec[jiij] - bzvec[iiij]) * w2m3m4
                + (bzvec[jjij] - bzvec[ijij]) * w23m4
                + (bzvec[jijj] - bzvec[iijj]) * w2m34
                + (bzvec[jjjj] - bzvec[ijjj]) * w234
            )
            / dx
            - bfac1 * y[idx, 0]
            + bfac2 * y[idx, 0] * y[idx, 2] * y[idx, 2]
        )

        dbzdy[idx] = (
            (
                (bzvec[ijii] - bzvec[iiii]) * w1m3m4m
                + (bzvec[jjii] - bzvec[jiii]) * w13m4m
                + (bzvec[ijji] - bzvec[iiji]) * w1m34m
                + (bzvec[jjji] - bzvec[jiji]) * w134m
                + (bzvec[ijij] - bzvec[iiij]) * w1m3m4
                + (bzvec[jjij] - bzvec[jiij]) * w13m4
                + (bzvec[ijjj] - bzvec[iijj]) * w1m34
                + (bzvec[jjjj] - bzvec[jijj]) * w134
            )
            / dy
            - bfac1 * y[idx, 1]
            + bfac2 * y[idx, 1] * y[idx, 2] * y[idx, 2]
        )

        dbzdz[idx] = (
            (
                (bzvec[iiji] - bzvec[iiii]) * w1m2m4m
                + (bzvec[jiji] - bzvec[jiii]) * w12m4m
                + (bzvec[ijji] - bzvec[ijii]) * w1m24m
                + (bzvec[jjji] - bzvec[jjii]) * w124m
                + (bzvec[iijj] - bzvec[iiij]) * w1m2m4
                + (bzvec[jijj] - bzvec[jiij]) * w12m4
                + (bzvec[ijjj] - bzvec[ijij]) * w1m24
                + (bzvec[jjjj] - bzvec[jjij]) * w124
            )
            / dz
            - 3.0 * bfac1 * y[idx, 2]
            + bfac2 * y[idx, 2] * y[idx, 2] * y[idx, 2]
        )

        #      ...calculate btot...
        b[idx] = (bx[idx] ** 2.0 + by[idx] ** 2.0 + bz[idx] ** 2.0) ** (0.5)

        # ...calculate derivatives of btot...
        dbdx[idx] = (bx[idx] * dbxdx[idx] + by[idx] * dbydx[idx] + bz[idx] * dbzdx[idx]) / b[idx]
        dbdy[idx] = (bx[idx] * dbxdy[idx] + by[idx] * dbydy[idx] + bz[idx] * dbzdy[idx]) / b[idx]
        dbdz[idx] = (bx[idx] * dbxdz[idx] + by[idx] * dbydz[idx] + bz[idx] * dbzdz[idx]) / b[idx]


@jit.rawkernel()
def do_step_kernel(
    k1,
    k2,
    k3,
    k4,
    k5,
    k6,
    y,
    y_next,
    z_next,
    h,
    t,
    rtol,
    t_final,
    mask,
    stopped,
    x_axis,
    nx,
    y_axis,
    ny,
    z_axis,
    nz,
    t_axis,
    nt,
    r_inner,
    integrate_backwards,
):
    """[CUPY KERNEL] Do a Runge-Kutta Step

    Calculates error, selectively steps, and adjusts step size
    """
    idx = jit.blockDim.x * jit.blockIdx.x + jit.threadIdx.x
    R = RK45Coeffs
    nstate = 5

    if idx < y.shape[0]:
        # Compute thte total error in the position and momentum
        err_total = 0.0
        ynorm = 0.0

        for i in range(nstate):
            y_next[idx, i] = (
                y[idx, i]
                + R.c1 * k1[idx, i]
                + R.c3 * k3[idx, i]
                + R.c4 * k4[idx, i]
                + R.c5 * k5[idx, i]
            )
            z_next[idx, i] = (
                y[idx, i]
                + R.d1 * k1[idx, i]
                + R.d3 * k3[idx, i]
                + R.d4 * k4[idx, i]
                + R.d5 * k5[idx, i]
                + R.d6 * k6[idx, i]
            )
            err_total += (y_next[idx, i] - z_next[idx, i]) ** 2
            ynorm += y[idx, i] ** 2

        err_total = err_total ** (0.5)
        ynorm = ynorm ** (0.5)

        # Compute the error tolerance
        tolerance = rtol[idx] * ynorm
        scale = 0.84 * (tolerance / err_total) ** (1 / 4)

        # Does not exceed target integration
        if integrate_backwards:
            stopped[idx] |= t[idx] + h[idx] < t_final[idx]
        else:
            stopped[idx] |= t[idx] + h[idx] > t_final[idx]

        # Next step is NaNs
        for i in range(nstate):
            stopped[idx] |= cp.isnan(y_next[idx, i])
            stopped[idx] |= cp.isnan(z_next[idx, i])
        
        # Within x,y,z axes bounds
        stopped[idx] |= z_next[idx, 0] < x_axis[0]
        stopped[idx] |= z_next[idx, 1] < y_axis[0]
        stopped[idx] |= z_next[idx, 2] < z_axis[0]

        stopped[idx] |= z_next[idx, 0] > x_axis[nx - 1]
        stopped[idx] |= z_next[idx, 1] > y_axis[ny - 1]
        stopped[idx] |= z_next[idx, 2] > z_axis[nz - 1]

        radius = (z_next[idx, 0] ** 2 + z_next[idx, 1] ** 2 + z_next[idx, 2] ** 2) ** 0.5
        stopped[idx] |= radius < r_inner[idx]

        # WIthin t axes bounds
        stopped[idx] |= t[idx] + h[idx] < t_axis[0]
        stopped[idx] |= t[idx] + h[idx] > t_axis[nt - 1]

        # Mask for iteration
        mask[idx] = (err_total < rtol[idx] * ynorm) & ~stopped[idx]

        # Selectively step particles
        if mask[idx]:
            t[idx] += h[idx]

            for i in range(nstate):
                y[idx, i] = z_next[idx, i]

        if not stopped[idx]:
            h[idx] *= scale
