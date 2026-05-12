use faer::{Mat, complex::Complex64, traits::ext::ComplexFieldExt};
use rand::RngExt;
use std::f64::consts::PI;
use pathfinding::prelude::dijkstra;
use std::{collections::{HashMap, HashSet, VecDeque}};

#[derive(Copy, Clone, Debug)]
pub struct Gate {
    pub name: &'static str,
    pub ctrl: usize,
    pub targ: usize,
    pub value: f64
}

impl PartialEq for Gate {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.ctrl == other.ctrl && self.targ == other.ctrl && self.value == other.value
    }
}

pub fn generate_u(num_qubits: usize) -> Mat<Complex64> {
    let dim = 1usize << num_qubits; // 2^num_qubits
    let mut rng = rand::rng();

    // Generate random complex entries on the unit circle: e^{i(2πx - π)}
    let mut entries: Vec<Complex64> = (0..dim * dim)
        .map(|_| {
            let x: f64 = rng.random();
            let theta = 2.0 * PI * x - PI;
            Complex64::new(theta.cos(), theta.sin())
        })
        .collect();

    // Normalize the flat array
    let norm = entries
        .iter()
        .map(|z| z.re * z.re + z.im * z.im)
        .sum::<f64>()
        .sqrt();
    for z in &mut entries {
        *z = Complex64::new(z.re / norm, z.im / norm);
    }

    // Reshape into dim × dim matrix (row-major indexing like NumPy)
    let mat_a = Mat::from_fn(dim, dim, |i, j| entries[i * dim + j]);

    let qr = mat_a.qr();
    let q = qr.compute_thin_Q();

    q
}

pub fn angles_from_diag(diag: Mat<Complex64>) -> Vec<f64> {
    let mut angles: Vec<f64> = Vec::from([]);
    let half = diag.nrows() / 2;
    for i in 0..half  {
        angles.push(diag[(half + i, half + i)].arg());
    }

    angles
}

pub fn get_subset_of_neighbors(neighbors: &HashMap<i64, HashSet<i64>>, subset_nodes: &HashSet<i64>) -> HashMap<i64, HashSet<i64>> {
    let mut subset = neighbors.clone();
    for key in neighbors.clone().keys() {
        if !subset_nodes.contains(key) { subset.remove(key); }
        else {
            let intersection: HashSet<i64> = subset.get(key).unwrap().intersection(subset_nodes).copied().collect();
            subset.insert(*key, intersection);
        }
    }

    subset
}

pub fn get_path(neighbors: &HashMap<i64, HashSet<i64>>, subset_nodes: &HashSet<i64>, source: i64, target: i64) -> Vec<i64> {
    let new_neighbors = get_subset_of_neighbors(neighbors, subset_nodes);
    let result = dijkstra(
    &source,
    |&n| new_neighbors.get(&n).into_iter().flatten().map(|&neighbor| (neighbor, 1u32)),
    |&n| n == target,
);

    result.unwrap().0
}

pub fn mottonen_transformation(multiplexer_angles: &Vec<f64>, gray_code: Option<&Vec<i64>>) -> Vec<f64> {
    let n = multiplexer_angles.len();
    let mut transformed_angles: Vec<f64> = vec![0.0; n];
    let num_controls = (n as f64).log2() as usize;

    let power = 2.0f64.powi(-(num_controls as i32));

    for i in 0..n {
        let mut temp: f64 = 0.0;
        let g_m = if gray_code.clone() == None { i ^ (i >> 1)} else {gray_code.clone().unwrap()[i] as usize};

        for j in 0..n {
            let dot_product = (g_m & j).count_ones() % 2;
            temp += if dot_product == 1 { -multiplexer_angles[j] } else { multiplexer_angles[j] };
        }
        transformed_angles[i] = power * temp * 2.0;
    }
    transformed_angles
}


use std::f64::consts::SQRT_2;

use faer::{mat};
// =====================================================================
//                            Public API
// =====================================================================

/// Build the 2^n × 2^n unitary realised by `gates`.
///
/// `gates` is in time order: the front of the queue is applied first.
pub fn gates_to_unitary(gates: &VecDeque<Gate>, num_qubits: usize) -> Mat<Complex64> {
    let dim = 1usize << num_qubits;
    let mut u = Mat::<Complex64>::identity(dim, dim);
    for g in gates {
        let m = gate_matrix(g, num_qubits);
        u = &u * &m;
    }
    u
}

/// Same as [`gates_to_unitary`] but infers `num_qubits` from the largest qubit
/// index that appears in `gates`. Returns the 1×1 identity if `gates` is empty.
pub fn gates_to_unitary_auto(gates: &VecDeque<Gate>) -> Mat<Complex64> {
    let max_q = gates
        .iter()
        .flat_map(|g| [g.ctrl, g.targ])
        .max()
        .unwrap_or(0);
    gates_to_unitary(gates, max_q + 1)
}

/// Compare two unitaries up to a global phase.
///
/// Returns `Some(phase)` if `built ≈ phase · target` (Frobenius distance < `tol`),
/// where `|phase| = 1`. Returns `None` otherwise.
///
/// The phase is computed as `tr(target^† · built) / n`, normalised to unit
/// modulus. If the matrices really do differ only by a scalar, this gives that
/// scalar exactly; otherwise the Frobenius check rejects them.
pub fn equal_up_to_global_phase(
    built: &Mat<Complex64>,
    target: &Mat<Complex64>,
    tol: f64,
) -> Option<Complex64> {
    let n = target.nrows();
    if built.nrows() != n || built.ncols() != n || target.ncols() != n {
        return None;
    }

    let inner = target.conjugate().transpose() * built;
    let mut tr = Complex64::new(0.0, 0.0);
    for i in 0..n {
        tr = tr + inner[(i, i)];
    }
    let avg = tr / Complex64::new(n as f64, 0.0);
    let mag = avg.abs();
    if mag < 1e-12 {
        // Matrices are essentially orthogonal — definitely not phase-equivalent.
        return None;
    }
    let phase = avg / Complex64::new(mag, 0.0);

    let mut sq = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let d = built[(i, j)] - target[(i, j)] * phase;
            sq += d.abs() * d.abs();
        }
    }
    if sq.sqrt() < tol {
        Some(phase)
    } else {
        None
    }
}

// =====================================================================
//                       Internal: gate dispatch
// =====================================================================

fn gate_matrix(g: &Gate, n: usize) -> Mat<Complex64> {
    match g.name {
        "RZ"   => embed_single_qubit(&rz(g.value), g.targ, n),
        "RY"   => embed_single_qubit(&ry(g.value), g.targ, n),
        "RX"   => embed_single_qubit(&rx(g.value), g.targ, n),
        "H"    => embed_single_qubit(&hadamard(), g.targ, n),
        "CX"   => cnot_full(g.ctrl, g.targ, n),
        "SWAP" => swap_full(g.ctrl, g.targ, n),
        other  => panic!("gates_to_unitary: unknown gate '{other}'"),
    }
}

// =====================================================================
//                    Internal: matrix construction
// =====================================================================

/// Embed a 2×2 gate on qubit `q` into the full 2^n × 2^n Hilbert space as
/// `I_{n-1} ⊗ ... ⊗ U_q ⊗ ... ⊗ I_0`.
fn embed_single_qubit(g: &Mat<Complex64>, q: usize, n: usize) -> Mat<Complex64> {
    let i2 = Mat::<Complex64>::identity(2, 2);
    let mut acc: Option<Mat<Complex64>> = None;
    for k in (0..n).rev() {
        let factor = if k == q { g.clone() } else { i2.clone() };
        acc = Some(match acc {
            None => factor,
            Some(a) => a.kron(factor),
        });
    }
    acc.unwrap_or_else(|| Mat::<Complex64>::identity(1, 1))
}

/// Canonical (unphased) CNOT on `(ctrl, targ)`, embedded in 2^n × 2^n.
fn cnot_full(ctrl: usize, targ: usize, n: usize) -> Mat<Complex64> {
    let dim = 1usize << n;
    Mat::from_fn(dim, dim, |i, j| {
        let mapped = if (j >> ctrl) & 1 == 1 { j ^ (1 << targ) } else { j };
        if i == mapped {
            Complex64::new(1.0, 0.0)
        } else {
            Complex64::new(0.0, 0.0)
        }
    })
}

/// SWAP on `(a, b)`, embedded in 2^n × 2^n.
fn swap_full(a: usize, b: usize, n: usize) -> Mat<Complex64> {
    let dim = 1usize << n;
    Mat::from_fn(dim, dim, |i, j| {
        let ba = (j >> a) & 1;
        let bb = (j >> b) & 1;
        let mapped = if ba != bb { j ^ ((1 << a) | (1 << b)) } else { j };
        if i == mapped {
            Complex64::new(1.0, 0.0)
        } else {
            Complex64::new(0.0, 0.0)
        }
    })
}

// =====================================================================
//                    Internal: single-qubit gates
// =====================================================================

fn rz(angle: f64) -> Mat<Complex64> {
    let h = angle / 2.0;
    mat![
        [Complex64::new(h.cos(), -h.sin()), Complex64::new(0.0, 0.0)],
        [Complex64::new(0.0, 0.0),          Complex64::new(h.cos(),  h.sin())],
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

fn hadamard() -> Mat<Complex64> {
    let s = 1.0 / SQRT_2;
    mat![
        [Complex64::new(s, 0.0), Complex64::new(s, 0.0)],
        [Complex64::new(s, 0.0), Complex64::new(-s, 0.0)],
    ]
}