//! Two-qubit unitary decomposition, translated from `two_qubit_decomposition.py`.
//!
//! Public API:
//!   - [`extract_diagonal`]        — 2-CNOT decomposition that leaves a residual
//!                                   diagonal (phase) gate.
//!   - [`three_cnot_decomposition`] — full 3-CNOT decomposition.
//!
//! Qiskit-specific helpers (`print_circ_unitary`) are intentionally omitted.
//!
//! Gate-tuple convention (matches `main.rs`):
//!   - single-qubit gate:  `Gate { name: "RZ" | "RY" | "RX", ctrl: 0, targ: q, value: angle }`
//!     (ctrl is unused; following the convention from `main.rs`)
//!   - CNOT:                `Gate { name: "CX", ctrl: c, targ: t, value: 0.0 }`

use std::collections::VecDeque;
use std::f64::consts::{E as E_CONST, PI, SQRT_2};

use faer::complex::Complex64;
use faer::traits::ext::ComplexFieldExt;
use faer::{mat, Mat, Scale};
use crate::utils::Gate;

// =====================================================================
//                           Small helpers
// =====================================================================

#[inline]
fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

#[inline]
fn cz() -> Complex64 {
    Complex64::new(0.0, 0.0)
}

#[inline]
fn co() -> Complex64 {
    Complex64::new(1.0, 0.0)
}

/// Principal complex square root.
fn csqrt(z: Complex64) -> Complex64 {
    let r = z.abs().sqrt();
    let half_arg = z.arg() / 2.0;
    Complex64::new(r * half_arg.cos(), r * half_arg.sin())
}

/// `exp(i*theta)` as a Complex64.
fn cis(theta: f64) -> Complex64 {
    Complex64::new(theta.cos(), theta.sin())
}

/// Wrap an angle into `(-pi, pi]`.
fn wrap_pi(a: f64) -> f64 {
    let mut x = (a + PI) % (2.0 * PI);
    if x <= 0.0 {
        x += 2.0 * PI;
    }
    x - PI
}

/// Trace of a 4×4 complex matrix.
fn trace4(m: &Mat<Complex64>) -> Complex64 {
    m[(0, 0)] + m[(1, 1)] + m[(2, 2)] + m[(3, 3)]
}

/// 2×2 complex determinant.
fn det2_c(m: &Mat<Complex64>) -> Complex64 {
    m[(0, 0)] * m[(1, 1)] - m[(0, 1)] * m[(1, 0)]
}

/// 4×4 complex determinant via 2×2-minor (Plücker / Laplace) expansion.
fn det4_c(m: &Mat<Complex64>) -> Complex64 {
    let a = m[(0, 0)] * m[(1, 1)] - m[(0, 1)] * m[(1, 0)];
    let b = m[(0, 0)] * m[(1, 2)] - m[(0, 2)] * m[(1, 0)];
    let cc = m[(0, 0)] * m[(1, 3)] - m[(0, 3)] * m[(1, 0)];
    let d = m[(0, 1)] * m[(1, 2)] - m[(0, 2)] * m[(1, 1)];
    let e = m[(0, 1)] * m[(1, 3)] - m[(0, 3)] * m[(1, 1)];
    let f = m[(0, 2)] * m[(1, 3)] - m[(0, 3)] * m[(1, 2)];
    let g = m[(2, 0)] * m[(3, 1)] - m[(2, 1)] * m[(3, 0)];
    let h = m[(2, 0)] * m[(3, 2)] - m[(2, 2)] * m[(3, 0)];
    let i = m[(2, 0)] * m[(3, 3)] - m[(2, 3)] * m[(3, 0)];
    let j = m[(2, 1)] * m[(3, 2)] - m[(2, 2)] * m[(3, 1)];
    let k = m[(2, 1)] * m[(3, 3)] - m[(2, 3)] * m[(3, 1)];
    let l = m[(2, 2)] * m[(3, 3)] - m[(2, 3)] * m[(3, 2)];
    a * l - b * k + cc * j + d * i - e * h + f * g
}

/// 4×4 real determinant.
fn det4_r(m: &Mat<f64>) -> f64 {
    let a = m[(0, 0)] * m[(1, 1)] - m[(0, 1)] * m[(1, 0)];
    let b = m[(0, 0)] * m[(1, 2)] - m[(0, 2)] * m[(1, 0)];
    let cc = m[(0, 0)] * m[(1, 3)] - m[(0, 3)] * m[(1, 0)];
    let d = m[(0, 1)] * m[(1, 2)] - m[(0, 2)] * m[(1, 1)];
    let e = m[(0, 1)] * m[(1, 3)] - m[(0, 3)] * m[(1, 1)];
    let f = m[(0, 2)] * m[(1, 3)] - m[(0, 3)] * m[(1, 2)];
    let g = m[(2, 0)] * m[(3, 1)] - m[(2, 1)] * m[(3, 0)];
    let h = m[(2, 0)] * m[(3, 2)] - m[(2, 2)] * m[(3, 0)];
    let i = m[(2, 0)] * m[(3, 3)] - m[(2, 3)] * m[(3, 0)];
    let j = m[(2, 1)] * m[(3, 2)] - m[(2, 2)] * m[(3, 1)];
    let k = m[(2, 1)] * m[(3, 3)] - m[(2, 3)] * m[(3, 1)];
    let l = m[(2, 2)] * m[(3, 3)] - m[(2, 3)] * m[(3, 2)];
    a * l - b * k + cc * j + d * i - e * h + f * g
}

/// Real n×n → complex n×n (zero imaginary part).
fn to_c(m: &Mat<f64>) -> Mat<Complex64> {
    let (r, ccols) = (m.nrows(), m.ncols());
    Mat::from_fn(r, ccols, |i, j| Complex64::new(m[(i, j)], 0.0))
}

/// Real part of a complex matrix.
fn re_part(m: &Mat<Complex64>) -> Mat<f64> {
    let (r, ccols) = (m.nrows(), m.ncols());
    Mat::from_fn(r, ccols, |i, j| m[(i, j)].re)
}

/// Imag part of a complex matrix.
fn im_part(m: &Mat<Complex64>) -> Mat<f64> {
    let (r, ccols) = (m.nrows(), m.ncols());
    Mat::from_fn(r, ccols, |i, j| m[(i, j)].im)
}

// =====================================================================
//                       Fixed matrices / constants
// =====================================================================

fn eye2_c() -> Mat<Complex64> {
    Mat::<Complex64>::identity(2, 2)
}

fn sigma_y() -> Mat<Complex64> {
    mat![
        [cz(), c(0.0, -1.0)],
        [c(0.0, 1.0), cz()],
    ]
}

fn sigma_y_kron_2() -> Mat<Complex64> {
    sigma_y().kron(sigma_y())
}

/// `exp(i*pi/4)`.
fn xi() -> Complex64 {
    cis(PI / 4.0)
}

/// CNOT (control 1, target 2) scaled by xi.
fn cnot_1_2_phased() -> Mat<Complex64> {
    let x = xi();
    mat![
        [x, cz(), cz(), cz()],
        [cz(), x, cz(), cz()],
        [cz(), cz(), cz(), x],
        [cz(), cz(), x, cz()],
    ]
}

/// CNOT (control 2, target 1) scaled by xi.
fn cnot_2_1_phased() -> Mat<Complex64> {
    let x = xi();
    mat![
        [x, cz(), cz(), cz()],
        [cz(), cz(), cz(), x],
        [cz(), cz(), x, cz()],
        [cz(), x, cz(), cz()],
    ]
}

/// Magic basis E.
fn magic_e() -> Mat<Complex64> {
    let s = 1.0 / SQRT_2;
    mat![
        [c(s, 0.0), c(0.0, s),  cz(),       cz()      ],
        [cz(),      cz(),       c(0.0, s),  c(s, 0.0) ],
        [cz(),      cz(),       c(0.0, s),  c(-s, 0.0)],
        [c(s, 0.0), c(0.0, -s), cz(),       cz()      ],
    ]
}

fn magic_e_dgr() -> Mat<Complex64> {
    magic_e().conjugate().transpose().to_owned()
}

// =====================================================================
//                Single-qubit rotation gates
// =====================================================================

fn rz(angle: f64) -> Mat<Complex64> {
    let h = angle / 2.0;
    mat![
        [Complex64::new(h.cos(), -h.sin()), cz()],
        [cz(),                              Complex64::new(h.cos(),  h.sin())],
    ]
}

fn rx(angle: f64) -> Mat<Complex64> {
    let (ch, sh) = ((angle / 2.0).cos(), (angle / 2.0).sin());
    mat![
        [Complex64::new(ch, 0.0),  Complex64::new(0.0, -sh)],
        [Complex64::new(0.0, -sh), Complex64::new(ch, 0.0)],
    ]
}

fn ry(angle: f64) -> Mat<Complex64> {
    let (ch, sh) = ((angle / 2.0).cos(), (angle / 2.0).sin());
    mat![
        [Complex64::new(ch, 0.0), Complex64::new(-sh, 0.0)],
        [Complex64::new(sh, 0.0), Complex64::new(ch, 0.0)],
    ]
}

// =====================================================================
//                       ZYZ Euler decomposition
// =====================================================================

/// Decompose a 2×2 unitary into ZYZ Euler angles `(phi, theta, lam)` such that
/// `U = Rz(phi) @ Ry(theta) @ Rz(lam)`.
fn get_zyz_angles(u: &Mat<Complex64>) -> (f64, f64, f64) {
    let u00 = u[(0, 0)];
    let u10 = u[(1, 0)];
    let u11 = u[(1, 1)];

    let theta = 2.0 * u10.abs().atan2(u00.abs());
    const TOL: f64 = 1e-64;

    let (phi, lam) = if u10.abs() < TOL {
        (2.0 * u11.arg(), 0.0)
    } else if u00.abs() < TOL {
        (2.0 * u10.arg(), 0.0)
    } else {
        let sum_phases = 2.0 * u11.arg();
        let diff_phases = 2.0 * u10.arg();
        (
            (sum_phases + diff_phases) / 2.0,
            (sum_phases - diff_phases) / 2.0,
        )
    };
    (phi, theta, lam)
}

// =====================================================================
//             Orthogonal congruence diagonalisation
// =====================================================================

/// Given a complex symmetric matrix S (= S^T), return a real orthogonal Q
/// (det = +1) such that Q^T S Q is approximately diagonal.
///
/// Uses an α-sweep on the real symmetric pencil R + α J (R = Re S, J = Im S).
fn orthogonal_congruence_diagonalize(s: &Mat<Complex64>) -> Mat<f64> {
    let n = s.nrows();
    debug_assert_eq!(n, s.ncols());

    let r = re_part(s);
    let j = im_part(s);

    let alphas: [f64; 21] = [
        0.0, 1.0, -1.0, SQRT_2, -SQRT_2,
        PI, -PI, 0.5, -0.5, E_CONST, -E_CONST,
        3.0f64.sqrt(), -(3.0f64.sqrt()),
        0.1, -0.1, 2.0, -2.0, 1.7, -1.7, 0.3, -0.3,
    ];

    let mut best_q: Option<Mat<f64>> = None;
    let mut best_err = f64::INFINITY;

    for &alpha in alphas.iter() {
        let m: Mat<f64> = Mat::from_fn(n, n, |i, k| r[(i, k)] + alpha * j[(i, k)]);

        // Symmetric (Hermitian) eigendecomposition for a real symmetric matrix.
        let eig = m
            .self_adjoint_eigen(faer::Side::Lower)
            .expect("self_adjoint_eigen failed in orthogonal_congruence_diagonalize");
        let q: Mat<f64> = eig.U().to_owned();

        // D = Q^T @ S @ Q (use complex form of Q to multiply with complex S).
        let q_c = to_c(&q);
        let d = q_c.transpose() * s * &q_c;

        // Max off-diagonal magnitude.
        let mut err: f64 = 0.0;
        for i in 0..n {
            for k in 0..n {
                if i != k {
                    let a = d[(i, k)].abs();
                    if a > err {
                        err = a;
                    }
                }
            }
        }

        if err < best_err {
            best_err = err;
            best_q = Some(q);
        }
        if err < 1e-12 {
            break;
        }
    }

    let mut q = best_q.expect("orthogonal_congruence_diagonalize: no candidate");

    if det4_r(&q) < 0.0 {
        for i in 0..n {
            q[(i, 0)] = -q[(i, 0)];
        }
    }
    q
}

// =====================================================================
//                     Auxiliary numerical pieces
// =====================================================================

/// Project a 4×4 unitary onto SU(4) by dividing out the global phase,
/// choosing the fourth-root branch with smallest absolute angle.
/// Returns `(U / phase, phase)`.
fn project_to_su4(u: &Mat<Complex64>) -> (Mat<Complex64>, Complex64) {
    let mut det_u = det4_c(u);
    if det_u.abs() < 1e-15 {
        det_u = Complex64::new(1e-15, 0.0);
    }
    let det_angle = det_u.arg();
    let det_mag = det_u.abs();
    let mag4 = det_mag.powf(0.25);

    let mut best_angle = 0.0_f64;
    let mut best_dist = f64::INFINITY;
    for k in 0..4 {
        let a = wrap_pi((det_angle + 2.0 * PI * k as f64) / 4.0);
        if a.abs() < best_dist {
            best_dist = a.abs();
            best_angle = a;
        }
    }
    let phase = Complex64::new(mag4, 0.0) * cis(best_angle);
    let inv_phase = Complex64::new(1.0, 0.0) / phase;
    let u_norm = u * Scale(inv_phase);
    (u_norm, phase)
}

/// Gamma map: `U -> U (σ_y ⊗ σ_y) U^T (σ_y ⊗ σ_y)`. (Plain transpose, not adjoint.)
fn gamma_map(u: &Mat<Complex64>) -> Mat<Complex64> {
    let syy = sigma_y_kron_2();
    u * &syy * u.transpose() * &syy
}

/// Sort eigenvalue angles, placing the branch cut inside the largest gap so
/// numerically-adjacent eigenvalues stay adjacent after sorting.
fn robust_angle_sort(eigvals: &[Complex64]) -> Vec<f64> {
    let n = eigvals.len();
    let angles: Vec<f64> = eigvals.iter().map(|z| z.arg()).collect();
    if n == 0 {
        return angles;
    }

    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| angles[a].partial_cmp(&angles[b]).unwrap());
    let sa: Vec<f64> = idx.iter().map(|&i| angles[i]).collect();

    let mut gaps = vec![0.0f64; n];
    for i in 0..n - 1 {
        gaps[i] = sa[i + 1] - sa[i];
    }
    gaps[n - 1] = sa[0] + 2.0 * PI - sa[n - 1];

    let mut gi = 0usize;
    for i in 1..n {
        if gaps[i] > gaps[gi] {
            gi = i;
        }
    }
    let cut = if gi < n - 1 {
        sa[gi] + gaps[gi] / 2.0
    } else {
        sa[n - 1] + gaps[n - 1] / 2.0
    };
    let shift = PI - cut;

    let shifted: Vec<f64> = eigvals
        .iter()
        .map(|z| (*z * cis(shift)).arg())
        .collect();

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| shifted[a].partial_cmp(&shifted[b]).unwrap());
    order.iter().map(|&i| angles[i]).collect()
}

/// Pair eigenvalues that should be conjugates and return `(a, b)` with
/// `0 <= a <= b`.
fn pair_conjugate_angles(eigvals: &[Complex64]) -> (f64, f64) {
    let n = eigvals.len();
    let mut used = vec![false; n];
    let mut pair_angles: Vec<f64> = Vec::new();

    for i in 0..n {
        if used[i] {
            continue;
        }
        let mut best_j: i32 = -1;
        let mut best_d = f64::INFINITY;
        for j in (i + 1)..n {
            if used[j] {
                continue;
            }
            // For a conjugate pair on the unit circle: λ_i * λ_j = 1.
            let d = (eigvals[i] * eigvals[j] - co()).abs();
            if d < best_d {
                best_d = d;
                best_j = j as i32;
            }
        }
        used[i] = true;
        if best_j >= 0 {
            used[best_j as usize] = true;
            let cos_a = (eigvals[i].re + eigvals[best_j as usize].re) / 2.0;
            let cos_clamped = cos_a.clamp(-1.0, 1.0);
            pair_angles.push(cos_clamped.acos());
        }
    }
    pair_angles.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (pair_angles[0], pair_angles[1])
}

/// Eigenvalues of a 4×4 complex matrix as a `Vec<Complex64>`.
fn eigvals4(m: &Mat<Complex64>) -> Vec<Complex64> {
    let eig = m.eigen().expect("eigen() failed");
    let s = eig.S();
    // `S()` returns eigenvalues as an n×1 matrix; pull them out one by one.
    (0..m.nrows()).map(|i| s[i]).collect()
}

/// Brute-force 4×4 Hungarian (min total cost permutation). Plenty fast for n=4.
fn hungarian_4x4(cost: &[[f64; 4]; 4]) -> [usize; 4] {
    fn recurse(
        a: &mut [usize; 4],
        k: usize,
        best: &mut ([usize; 4], f64),
        cost: &[[f64; 4]; 4],
    ) {
        if k == 4 {
            let total: f64 = (0..4).map(|i| cost[i][a[i]]).sum();
            if total < best.1 {
                *best = (*a, total);
            }
            return;
        }
        for i in k..4 {
            a.swap(k, i);
            recurse(a, k + 1, best, cost);
            a.swap(k, i);
        }
    }
    let mut a = [0, 1, 2, 3];
    let mut best = ([0, 1, 2, 3], f64::INFINITY);
    recurse(&mut a, 0, &mut best, cost);
    best.0
}

/// Reshape a 4×4 as (2,2,2,2), transpose (0,2,1,3), reshape back to 4×4.
/// Equivalent to: `M_flat[2a+c, 2b+d] = M[2a+b, 2c+d]`.
fn reshape_swap_axes(m: &Mat<Complex64>) -> Mat<Complex64> {
    Mat::from_fn(4, 4, |p, q| {
        let a = p / 2;
        let cc = p % 2;
        let b = q / 2;
        let d = q % 2;
        m[(2 * a + b, 2 * cc + d)]
    })
}

// =====================================================================
//                  Tensor / KAK pieces
// =====================================================================

/// Given a 4×4 matrix that is (approximately) `kron(a, b)`, recover `a` and `b`.
fn extract_tensor_factors(m: &Mat<Complex64>) -> (Mat<Complex64>, Mat<Complex64>) {
    let m_flat = reshape_swap_axes(m);
    let svd = m_flat.svd().expect("SVD failed in extract_tensor_factors");
    let u = svd.U();
    let s = svd.S();
    let v = svd.V(); // faer: x = U Σ V^†, so vh[0, :] in numpy = conj(V[:, 0]).

    // u[:, 0].reshape(2, 2)
    let mut a = Mat::from_fn(2, 2, |i, j| u[(2 * i + j, 0)]);
    // vh[0, :].reshape(2, 2)  ==  conj(V[:, 0]).reshape(2, 2)
    let mut b = Mat::from_fn(2, 2, |i, j| v[(2 * i + j, 0)].conj());

    // First singular value (real; stored as Complex64 with zero imaginary part).
    let s0 = s[0].re;
    let sqrt_s0 = Complex64::new(s0.sqrt(), 0.0);
    a = a * Scale(sqrt_s0);
    b = b * Scale(sqrt_s0);

    let det_a = det2_c(&a);
    let det_b = det2_c(&b);
    let inv = |z: Complex64| co() / z;

    if det_a.abs1() > 1e-30 {
        a = a * Scale(inv(csqrt(det_a)));
    } else {
        a = a * Scale(Complex64::new(1e15, 0.0));
    }
    if det_b.abs() > 1e-30 {
        b = b * Scale(inv(csqrt(det_b)));
    } else {
        b = b * Scale(Complex64::new(1e15, 0.0));
    }

    (a, b)
}

/// Solve for the four single-qubit unitaries `(a, b, c, d)` such that
/// `U_E ≈ kron(a, b) @ kernel_E @ kron(c, d)` (after the magic-basis map).
fn get_single_qubit_unitaries(
    u_e: &Mat<Complex64>,
    k_e: &Mat<Complex64>,
) -> (
    Mat<Complex64>,
    Mat<Complex64>,
    Mat<Complex64>,
    Mat<Complex64>,
) {
    // S_U = U_E @ U_E^T  (PLAIN transpose, not adjoint).
    let s_u: Mat<Complex64> = u_e * u_e.transpose();
    let s_k: Mat<Complex64> = k_e * k_e.transpose();

    let a_u: Mat<f64> = orthogonal_congruence_diagonalize(&s_u);
    let b_k0: Mat<f64> = orthogonal_congruence_diagonalize(&s_k);

    let a_u_c = to_c(&a_u);
    let b_k0_c = to_c(&b_k0);

    // D_U and D_k (diagonal entries).
    let d_u_mat: Mat<Complex64> = a_u_c.transpose() * &s_u * &a_u_c;
    let d_k_mat: Mat<Complex64> = b_k0_c.transpose() * &s_k * &b_k0_c;

    let mut ang_u = [0.0f64; 4];
    let mut ang_k = [0.0f64; 4];
    for i in 0..4 {
        ang_u[i] = d_u_mat[(i, i)].arg();
        ang_k[i] = d_k_mat[(i, i)].arg();
    }

    // Hungarian assignment on circular angle-distance.
    let mut cost = [[0.0f64; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            let diff = (ang_u[i] - ang_k[j]).abs();
            cost[i][j] = diff.min(2.0 * PI - diff);
        }
    }
    let perm = hungarian_4x4(&cost);

    // Permute columns of B_k.
    let b_k_perm: Mat<f64> = Mat::from_fn(4, 4, |i, j| b_k0[(i, perm[j])]);

    // Precompute pieces that don't change across sign trials.
    let e_mat = magic_e();
    let e_dgr = magic_e_dgr();
    let k_e_adj = k_e.conjugate().transpose().to_owned();

    // Try all 16 sign combinations on columns of B_k; keep the candidate making
    // C_tilde most rank-1 (i.e. closest to a tensor product).
    let mut best_b = b_k_perm.cloned();
    let mut best_rank1 = f64::INFINITY;
    let mut b_k = b_k_perm.cloned();

    for sign_bits in 0..16u32 {
        let mut signs = [1.0f64; 4];
        for i in 0..4 {
            if (sign_bits >> i) & 1 == 1 {
                signs[i] = -1.0;
            }
        }
        // NOTE: B_cand = current B_k with column signs flipped.  The Python
        // source updates B_k = best_B inside the loop body, so we mirror that.
        let b_cand: Mat<f64> = Mat::from_fn(4, 4, |i, j| b_k[(i, j)] * signs[j]);
        let b_cand_c = to_c(&b_cand);

        // Need det(A_U @ B_cand^T) > 0 (real-valued).
        let mab = &a_u_c * b_cand_c.transpose();
        let mab_r = re_part(&mab);
        if det4_r(&mab_r) < 0.0 {
            // Mirror Python: B_k = best_B at end of iteration even when skipped.
            b_k = best_b.cloned();
            continue;
        }

        let c_cand = &k_e_adj * &b_cand_c * a_u_c.transpose() * u_e;
        let c_tilde = &e_mat * &c_cand * &e_dgr;

        let m_flat = reshape_swap_axes(&c_tilde);
        let svd = m_flat
            .svd()
            .expect("SVD failed in get_single_qubit_unitaries");
        let s_v = svd.S();
        let s0 = s_v[0].re;
        let s1 = s_v[1].re;
        let ratio = if s0 > 1e-15 { s1 / s0 } else { 0.0 };
        if ratio < best_rank1 {
            best_rank1 = ratio;
            best_b = b_cand.cloned();
        }

        // Match Python's in-loop `B_k = best_B`.
        b_k = best_b.cloned();
    }

    let b_k_final = best_b;
    let mut a_u_final = a_u.cloned();

    // Final det check: flip a column of A_U if needed.
    let a_u_final_c = to_c(&a_u_final);
    let b_k_final_c = to_c(&b_k_final);
    let mab = &a_u_final_c * b_k_final_c.transpose();
    let mab_r = re_part(&mab);
    if det4_r(&mab_r) < 0.0 {
        for i in 0..4 {
            a_u_final[(i, 0)] = -a_u_final[(i, 0)];
        }
    }
    let a_u_final_c = to_c(&a_u_final);

    let big_c = &k_e_adj * &b_k_final_c * a_u_final_c.transpose() * u_e;
    let a_tilde = &e_mat * &a_u_final_c * b_k_final_c.transpose() * &e_dgr;
    let c_tilde = &e_mat * &big_c * &e_dgr;

    let (a, b) = extract_tensor_factors(&a_tilde);
    let (cc, d) = extract_tensor_factors(&c_tilde);
    (a, b, cc, d)
}

/// If the reconstruction and `U` differ by at most a unit-modulus scalar,
/// absorb that scalar into `a`.
fn fix_global_phase(
    recon: &Mat<Complex64>,
    u: &Mat<Complex64>,
    a: Mat<Complex64>,
) -> Mat<Complex64> {
    let phase_diff =
        trace4(&(recon.conjugate().transpose() * u)) / Complex64::new(4.0, 0.0);
    if phase_diff.abs() > 1e-12 {
        let correction = phase_diff.conj() / Complex64::new(phase_diff.abs(), 0.0);
        a * Scale(correction)
    } else {
        a
    }
}

// =====================================================================
//                       Gate-emission helpers
// =====================================================================

#[inline]
fn g1(name: &'static str, angle: f64, qubit: usize) -> Gate {
    Gate {
        name,
        ctrl: 0,
        targ: qubit,
        value: angle,
    }
}

#[inline]
fn cx(ctrl: usize, targ: usize) -> Gate {
    Gate {
        name: "CX",
        ctrl,
        targ,
        value: 0.0,
    }
}

// =====================================================================
//                       Public decompositions
// =====================================================================

/// 2-CNOT decomposition that leaves a residual diagonal "phase" gate.
/// Returns `(diag_u * phase, gate_queue)`.
pub fn extract_diagonal(
    u: &Mat<Complex64>,
    source: usize,
) -> (Mat<Complex64>, VecDeque<Gate>) {
    let (big_u, phase) = project_to_su4(u);
    // M = gamma_map(U^T)^T
    let big_u_t = big_u.transpose().to_owned();
    let m_mat = gamma_map(&big_u_t).transpose().to_owned();

    let t1 = m_mat[(0, 0)];
    let t2 = m_mat[(1, 1)];
    let t3 = m_mat[(2, 2)];
    let t4 = m_mat[(3, 3)];

    let num = (t1 + t2 + t3 + t4).im;
    let den = (t1 + t4 - t3 - t2).re;
    let psi = if num.abs() < 1e-12 && den.abs() < 1e-12 {
        0.0
    } else {
        num.atan2(den)
    };

    let cnot12 = cnot_1_2_phased();
    let i2 = eye2_c();

    let delta = &cnot12 * i2.kron(rz(psi)) * &cnot12;
    let gamma_u_delta = gamma_map(&(&big_u * &delta));
    let eigvals = eigvals4(&gamma_u_delta);

    let (a_angle, b_angle) = pair_conjugate_angles(&eigvals);
    let theta = (a_angle - b_angle) / 2.0;
    let phi = -(a_angle + b_angle) / 2.0;

    let e_mat = magic_e();
    let e_dgr = magic_e_dgr();

    let u_e = &e_dgr * &big_u * &delta * &e_mat;
    let kernel = &cnot12 * rx(theta + PI).kron(rz(phi)) * &cnot12;
    let k_e = &e_dgr * &kernel * &e_mat;

    let (a, b, cc, d) = get_single_qubit_unitaries(&u_e, &k_e);

    let diag_u = &cnot12 * i2.kron(rz(-psi)) * &cnot12;
    let recon = a.kron(b.clone()) * &kernel * cc.kron(d.clone()) * &diag_u;

    let a = fix_global_phase(&recon, &big_u, a);

    let (a1, a2, a3) = get_zyz_angles(&a);
    let (b1, b2, b3) = get_zyz_angles(&b);
    let (c1, c2, c3) = get_zyz_angles(&cc);
    let (d1, d2, d3) = get_zyz_angles(&d);

    let mut gates: VecDeque<Gate> = VecDeque::new();
    gates.push_back(g1("RZ", c3, 1));
    gates.push_back(g1("RY", c2, 1));
    gates.push_back(g1("RZ", c1, 1));
    gates.push_back(g1("RZ", d3, source));
    gates.push_back(g1("RY", d2, source));
    gates.push_back(g1("RZ", d1, source));
    gates.push_back(cx(1, source));
    gates.push_back(g1("RZ", phi, source));
    gates.push_back(g1("RX", theta + PI, 1));
    gates.push_back(cx(1, source));
    gates.push_back(g1("RZ", a3, 1));
    gates.push_back(g1("RY", a2, 1));
    gates.push_back(g1("RZ", a1, 1));
    gates.push_back(g1("RZ", b3, source));
    gates.push_back(g1("RY", b2, source));
    gates.push_back(g1("RZ", b1, source));

    (diag_u * Scale(phase), gates)
}

/// Full 3-CNOT decomposition for a generic 2-qubit unitary.
pub fn three_cnot_decomposition(u: &Mat<Complex64>, source: usize) -> VecDeque<Gate> {
    let (big_u, _phase) = project_to_su4(u);
    let gamma_u = gamma_map(&big_u);
    let eigvals = eigvals4(&gamma_u);
    let angles = robust_angle_sort(&eigvals);

    let alpha = -(angles[0] + angles[1]) / 2.0 - PI / 2.0;
    let beta = (angles[0] + angles[2]) / 2.0 + PI / 2.0;
    let delta = -(angles[1] + angles[2]) / 2.0 - PI / 2.0;

    let cnot12 = cnot_1_2_phased();
    let cnot21 = cnot_2_1_phased();
    let i2 = eye2_c();
    let e_mat = magic_e();
    let e_dgr = magic_e_dgr();

    let kernel = &cnot21
        * i2.kron(ry(alpha))
        * &cnot12
        * rz(delta).kron(ry(beta))
        * &cnot21;

    let u_e = &e_dgr * &big_u * &e_mat;
    let k_e = &e_dgr * &kernel * &e_mat;

    let (a, b, cc, d) = get_single_qubit_unitaries(&u_e, &k_e);

    let recon = a.kron(b.clone()) * &kernel * cc.kron(d.clone());
    let a = fix_global_phase(&recon, &big_u, a);

    let (a1, a2, a3) = get_zyz_angles(&a);
    let (b1, b2, b3) = get_zyz_angles(&b);
    let (c1, c2, c3) = get_zyz_angles(&cc);
    let (d1, d2, d3) = get_zyz_angles(&d);

    let mut gates: VecDeque<Gate> = VecDeque::new();
    gates.push_back(g1("RZ", c3, 1));
    gates.push_back(g1("RY", c2, 1));
    gates.push_back(g1("RZ", c1, 1));
    gates.push_back(g1("RZ", d3, source));
    gates.push_back(g1("RY", d2, source));
    gates.push_back(g1("RZ", d1, source));
    gates.push_back(cx(source, 1));
    gates.push_back(g1("RZ", delta, 1));
    gates.push_back(g1("RY", beta, source));
    gates.push_back(cx(1, source));
    gates.push_back(g1("RY", alpha, source));
    gates.push_back(cx(source, 1));
    gates.push_back(g1("RZ", a3, 1));
    gates.push_back(g1("RY", a2, 1));
    gates.push_back(g1("RZ", a1, 1));
    gates.push_back(g1("RZ", b3, source));
    gates.push_back(g1("RY", b2, source));
    gates.push_back(g1("RZ", b1, source));

    gates
}