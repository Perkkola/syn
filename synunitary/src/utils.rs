use faer::{Mat, complex::Complex64};
use rand::RngExt;
use std::f64::consts::PI;
use pathfinding::prelude::dijkstra;
use std::{collections::{HashMap, HashSet, VecDeque}};

#[derive(Copy, Clone)]
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