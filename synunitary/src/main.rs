use std::collections::{VecDeque, HashMap};
use std::{env};
use faer::traits::ext::ComplexFieldExt;
use faer::{Mat, complex::Complex64, Scale, mat};
use itertools::Itertools;
use synunitary::utils::{generate_u, angles_from_diag, Gate, mottonen_transformation};
use synunitary::architecture_aware_routing::RoutedMultiplexer;
use std::io;

pub struct BlockZXZ {
    coupling_map: Vec<[i64; 2]>,
    gate_queue: VecDeque<Gate>,
    diag: Mat<Complex64>,
    original_multiplexer: RoutedMultiplexer,
    routed_multiplexers: HashMap<i64, RoutedMultiplexer>,
    swaps_per_level: HashMap<i64, Option<Vec<i64>>>,
    swap_maps: HashMap<i64, Option<HashMap<usize, usize>>>,
}

impl BlockZXZ {
    pub fn new(coupling_map: Vec<[i64; 2]>) -> Self {
        BlockZXZ { 
            coupling_map, 
            gate_queue: VecDeque::new(), 
            diag: Mat::<Complex64>::identity(1, 1), 
            original_multiplexer: RoutedMultiplexer::new(Vec::new(), 0),
            routed_multiplexers: HashMap::new(), 
            swaps_per_level: HashMap::new(), 
            swap_maps: HashMap::new() }
    }

    fn swap_to(&mut self, path: Vec<i64>, reverse: bool) {
        let indices: Vec<usize> = if reverse { (0..path.len() - 1).rev().collect() } else { (0..path.len() - 1).collect() };
        for i in indices {
            let q_1 = self.original_multiplexer.arch_to_grey_map[&path[i]];
            let q_2 = self.original_multiplexer.arch_to_grey_map[&path[i + 1]];

            self.gate_queue.push_back( Gate {
                name: "SWAP",
                ctrl: q_1,
                targ: q_2,
                value: 0.0
            });
        }
    }

    fn get_cnot_unitary(&self, num_qubits: i64, cnot: &Gate) -> Mat<Complex64> {
        let mut cnot_base: Mat<Complex64> = Mat::identity(2i64.pow(num_qubits as u32) as usize, 2i64.pow(num_qubits as u32) as usize);
        let target_mask = 1 << cnot.targ;

        for i in 0..2i64.pow(num_qubits as u32) {
            if (i >> cnot.ctrl) & 1 == 1 {
                cnot_base[(i as usize, i as usize)] = Complex64::new(0.0,0.0);
                cnot_base[(i as usize, (i ^ target_mask) as usize)] = Complex64::new(1.0,0.0);

            }
        }

        cnot_base
    }

    fn block(&self, tl: &Mat<Complex64>, tr: &Mat<Complex64>, bl: &Mat<Complex64>, br: &Mat<Complex64>) -> Mat<Complex64> {
        let n = tl.nrows();

        Mat::from_fn(2 * n, 2 * n, |i, j| {
            if i < n && j < n {
                tl[(i, j)]
            }
            else if i >= n && j < n {
                tr[(i % n, j % n)]
            }
            else if i < n && j >= n {
                bl[(i % n, j % n)]
            } else {
                br[(i % n, j % n)]
            }
        })
    }

    fn reunitarize(&self, w: Mat<Complex64>) -> Mat<Complex64> {
        let mut x = w.cloned();
        let n = x.nrows();
        for _ in 0..50 {
            let xhx = x.conjugate().transpose() * &x;
            let eye = Mat::<Complex64>::identity(n, n);
            let err = (&xhx - &eye).map(|val| val.abs()).max().unwrap_or(0.0);
            if err < 1.0e-14 {
                break;
            }
            x = x * (Scale(Complex64::new(3.0, 0.0)) * &eye - &xhx) / Scale(Complex64::new(2.0, 0.0));
        }
        x
    }

    pub fn demultiplex(&self, u_1: Mat<Complex64>, u_2: Mat<Complex64>) -> (Mat<Complex64>, Mat<Complex64>, Mat<Complex64>) {
        let block_len = u_1.nrows();
        let zeros = Mat::<Complex64>::zeros(block_len, block_len);

        let u_1_u_2_dgr = u_1 * u_2.conjugate().transpose();

        let eigen_decomp = u_1_u_2_dgr.eigen().expect("Something went wrong!");
        let eigvals = eigen_decomp.S();
        let eigvecs = eigen_decomp.U();

        let sqrt_eigval = eigvals.map(|eigval| (Complex64::i() * eigval.arg() / 2.0).exp());

        let mut w = &sqrt_eigval * &eigvecs.conjugate().transpose() * &u_2;
        w = self.reunitarize(w);

        let diag_as_mat = Mat::from_fn(block_len, block_len, |i, j| {
            if i == j { sqrt_eigval[i] } else { Complex64::new(0.0, 0.0) }
        });

        let block_diag = self.block(&diag_as_mat, &zeros, &zeros, &diag_as_mat.conjugate().transpose().to_owned());

        (eigvecs.to_owned(), block_diag, w)
    }

    fn decompose_two_qubit_unitary(&mut self, mut u: Mat<Complex64>, rightmost_unitary: bool, leftmost_unitary: bool) {
        if !rightmost_unitary { u = self.diag.clone() * u; }
        if !leftmost_unitary {
            // let (diag_u, gates) = extract_diagonal(u, 0);
            // self.diag = diag_u
            // self.gate_queue.append(&mut gates);
        } else {
            // let gates = three_cnot_decomposition(u, 0);
            // self.gate_queue.append(&mut gates);
        }
    }

    fn initialize_multiplexers(&mut self, num_qubits: i64) {
        let mut multiplexer = RoutedMultiplexer::new(self.coupling_map.clone(), num_qubits);
        multiplexer.map_grey_qubits_to_arch_unitary_synth();
        multiplexer.get_optimal_gray_code();

        self.original_multiplexer = multiplexer.clone();
        let mut recursion_level = 0;

        for i in (3..num_qubits + 1).rev() {
            let target = multiplexer.grey_to_arch_map[&((i - 1) as usize)];
            let mut path_to_root = &multiplexer.optimal_neighborhood[&target];

            let mut best_cost = multiplexer.clone().get_optimal_gray_code();
            let mut best_arch_to_grey = multiplexer.arch_to_grey_map.clone();
            let mut best_swap_count = 0i64;
            let mut best_multiplexer = multiplexer.clone();

            let mut arch_to_grey_copy = multiplexer.arch_to_grey_map.clone();

            for node in path_to_root {
                if !arch_to_grey_copy.keys().contains(node) {
                    multiplexer.clone().recompute_optimal_neighborhood();
                    path_to_root = &multiplexer.optimal_neighborhood[&target];
                }
            }

            let mut swap_count = 0i64;

            for j in (1..path_to_root.len()).rev() {
                let mut multiplexer_cp = multiplexer.clone();
                let temp = arch_to_grey_copy[&path_to_root[j]];
                arch_to_grey_copy.insert(path_to_root[j], arch_to_grey_copy[&path_to_root[j - 1]]);
                arch_to_grey_copy.insert(path_to_root[j - 1], temp);
                
                multiplexer_cp.arch_to_grey_map = arch_to_grey_copy.clone();
                multiplexer_cp.grey_to_arch_map = arch_to_grey_copy.clone().values().into_iter().copied().zip(arch_to_grey_copy.clone().keys().copied()).collect();
                multiplexer_cp.root = path_to_root[j - 1];

                swap_count += 1;
                let mut current_cost = multiplexer_cp.get_optimal_gray_code();
                current_cost += swap_count * 3 * 2;

                if current_cost < best_cost {
                    best_cost = current_cost;
                    best_arch_to_grey = arch_to_grey_copy.clone();
                    best_swap_count = swap_count;
                    best_multiplexer = multiplexer_cp.clone();
                }
            }

            let mut current_level_multiplexer = best_multiplexer.clone();
            current_level_multiplexer.arch_to_grey_map = best_arch_to_grey.clone();
            current_level_multiplexer.grey_to_arch_map = best_arch_to_grey.clone().values().into_iter().copied().zip(best_arch_to_grey.clone().keys().copied()).collect();
            current_level_multiplexer.root = current_level_multiplexer.grey_to_arch_map[&((i - 1) as usize)];
            current_level_multiplexer.furthest_node = current_level_multiplexer.grey_to_arch_map[&0];
            current_level_multiplexer.recompute_optimal_neighborhood();

            if best_swap_count == 0 {
                self.swaps_per_level.insert(recursion_level, None);
                self.swap_maps.insert(recursion_level, None);
            } else {
                self.swaps_per_level.insert(recursion_level, Some(path_to_root[(path_to_root.len() - 1 - best_swap_count as usize)..].into()));
                self.swap_maps.insert(recursion_level, Some(current_level_multiplexer.arch_to_grey_map.clone().values().into_iter().copied().zip(self.original_multiplexer.arch_to_grey_map.clone().values().copied()).collect()));
            }
            self.routed_multiplexers.insert(recursion_level, current_level_multiplexer);

            multiplexer.num_qubits -= 1;
            multiplexer.num_control -= 1;
            let value = multiplexer.grey_to_arch_map.remove(&((i - 1) as usize)).unwrap();
            multiplexer.arch_to_grey_map.remove(&value);
            multiplexer.arch_qubits.remove(&value);
            multiplexer.root = multiplexer.grey_to_arch_map[&((i - 2) as usize)];
            multiplexer.optimal_neighborhood.remove(&value);

            recursion_level += 1;
        }
    }

    pub fn compute_decomposition(&mut self, 
        u: Mat<Complex64>, 
        init: bool, 
        rightmost_unitary: bool, 
        leftmost_unitary: bool, 
        recursion_level: i64) {
        
        let n = u.nrows();
        let num_qubits = (n as f64).log2().ceil() as i64;
        let target_qubit = num_qubits - 1;

        if init { 
            self.initialize_multiplexers(num_qubits);
            print!("{:?}", self.routed_multiplexers);
         }
        if num_qubits == 2 {
            self.decompose_two_qubit_unitary(u.cloned(), rightmost_unitary, leftmost_unitary);
            return;
        }

        let block_len = n / 2;

        let x = u.get(0..block_len, 0..block_len);
        let y = u.get(0..block_len, block_len..);
        let u_21 = u.get(block_len.., 0..block_len);
        let u_22 = u.get(block_len.., block_len..);

        let svd_x = x.svd().expect("Something went wrong!");
        let v_x = svd_x.U();
        let sigma_x = svd_x.S();
        let w_x_dgr = svd_x.V();
        
        let s_x = v_x * sigma_x * v_x.conjugate().transpose();
        let u_x = v_x * w_x_dgr;

        let svd_y = y.svd().expect("Something went wrong!");
        let v_y = svd_y.U();
        let sigma_y = svd_y.S();
        let w_y_dgr = svd_y.V();

        let s_y = v_y * sigma_y * v_y.conjugate().transpose();
        let u_y = v_y * w_y_dgr;

        let c = Scale(-Complex64::i()) * &u_x.conjugate().transpose() * &u_y;
        let a_1 = (&s_x + Scale(Complex64::i()) * &s_y) * &u_x;
        let a_2 = &u_21 + &u_22 * (Scale(Complex64::i()) * &u_y.conjugate().transpose() * &u_x);

        let eye = Mat::<Complex64>::identity(block_len, block_len);
        let zeros = Mat::<Complex64>::zeros(block_len, block_len);

        let b = Scale(Complex64::new(2.0, 0.0)) * &a_1.conjugate().transpose() * &x - &eye;

        let (v_a, block_diag_a, w_a) = self.demultiplex(a_1, a_2);
        let (v_c, block_diag_c, w_c) = self.demultiplex(eye.clone(), c);
        
        let mut b_tilde = self.block(&(&w_a * &v_c), &zeros, &zeros, &(&w_a * &b * &v_c));


        let h = mat![[Complex64::new(1.0 / f64::sqrt(2.0), 0.0), Complex64::new(1.0 / f64::sqrt(2.0), 0.0)],
                                    [Complex64::new(1.0 / f64::sqrt(2.0), 0.0), Complex64::new(-1.0 / f64::sqrt(2.0), 0.0)]];

        let mut routed_multiplexer = self.routed_multiplexers[&recursion_level].clone();
        let angles_c = angles_from_diag(block_diag_c);
        let transformed_angles_c = mottonen_transformation(&angles_c, Some(&routed_multiplexer.gray_code));

        let angles_a = angles_from_diag(block_diag_a);
        let transformed_angles_a = mottonen_transformation(&angles_a, Some(&routed_multiplexer.gray_code));

        // print!("{transformed_angles_c:#?}");
        // let mut input = String::new();
        // io::stdin().read_line(&mut input);
        routed_multiplexer.multiplexer_angles = transformed_angles_c;

        let (_, mut gates_c) = routed_multiplexer.execute_gates();
        let mut gates_a = routed_multiplexer.replace_mapped_angles(&transformed_angles_a, true);

        while !gates_c.is_empty() {
            let mut popped_gate = gates_c.pop_back().unwrap();

            if popped_gate.name == "RZ" {
                let next_cnot = gates_c.pop_back().unwrap();

                if next_cnot.ctrl == 0 && next_cnot.targ == 1 {
                    gates_c.push_back(popped_gate.clone());
                    popped_gate = next_cnot;

                    let rz_a = gates_a.pop_front().unwrap();
                    gates_a.pop_front();
                    gates_a.push_front(rz_a);
                }
                else {
                    gates_c.push_back(next_cnot);
                    gates_c.push_back(popped_gate);
                    break;
                }
            } else if popped_gate.ctrl >= target_qubit as usize + 1 || popped_gate.targ >= target_qubit as usize + 1 {
                gates_c.push_back(popped_gate);
                break;
            }
            else {
                gates_a.pop_front().unwrap();
            }

            let mut unitary = self.get_cnot_unitary(num_qubits, &popped_gate);
            unitary = h.kron(eye.clone()) * unitary * h.kron(eye.clone());
            b_tilde = unitary.clone() * b_tilde * unitary;
        }

        let b_11 = b_tilde.get(0..block_len, 0..block_len).to_owned();
        let b_22 = b_tilde.get(block_len.., block_len..).to_owned();

        let (v_b, block_diag_b, w_b) = self.demultiplex(b_11, b_22);
        let angles_b = angles_from_diag(block_diag_b);
        let transformed_angles_b = mottonen_transformation(&angles_b, Some(&routed_multiplexer.gray_code));
        let mut gates_b = routed_multiplexer.replace_mapped_angles(&transformed_angles_b, false);

        if self.swap_maps[&recursion_level] != None {
            let swap_map = self.swap_maps[&recursion_level].clone().unwrap();

            for gate in &mut gates_a {
                gate.ctrl = swap_map[&gate.ctrl];
                gate.targ = swap_map[&gate.targ];
            }
            for gate in &mut gates_b {
                gate.ctrl = swap_map[&gate.ctrl];
                gate.targ = swap_map[&gate.targ];
            }
            for gate in &mut gates_c {
                gate.ctrl = swap_map[&gate.ctrl];
                gate.targ = swap_map[&gate.targ];
            }
        }

        self.compute_decomposition(v_a, false, rightmost_unitary, false, recursion_level + 1);
        
        if self.swap_maps[&recursion_level] != None { self.swap_to(self.swaps_per_level[&recursion_level].clone().unwrap(), true);}
        self.gate_queue.append(&mut gates_a);
        if self.swap_maps[&recursion_level] != None { self.swap_to(self.swaps_per_level[&recursion_level].clone().unwrap(), false);}
        
        self.compute_decomposition(v_b, false, false, false, recursion_level + 1);
        
        self.gate_queue.push_back(Gate {name: "H", ctrl: 0, targ: target_qubit as usize, value: 0.0 });
        if self.swap_maps[&recursion_level] != None { self.swap_to(self.swaps_per_level[&recursion_level].clone().unwrap(), true);}
        self.gate_queue.append(&mut gates_b);
        if self.swap_maps[&recursion_level] != None { self.swap_to(self.swaps_per_level[&recursion_level].clone().unwrap(), false);}
        self.gate_queue.push_back(Gate {name: "H", ctrl: 0, targ: target_qubit as usize, value: 0.0 });
        
        self.compute_decomposition(w_b, false, false, false, recursion_level + 1);
        
        if self.swap_maps[&recursion_level] != None { self.swap_to(self.swaps_per_level[&recursion_level].clone().unwrap(), true);}
        self.gate_queue.append(&mut gates_c);
        if self.swap_maps[&recursion_level] != None { self.swap_to(self.swaps_per_level[&recursion_level].clone().unwrap(), false);}
        
        self.compute_decomposition(w_c, false, false, leftmost_unitary, recursion_level + 1);
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let default = "3".to_string();
    let num_qubits = args.get(1).unwrap_or(&default).parse::<usize>().expect("Expected a number");
    let coupling_map: Vec<[i64; 2]> = Vec::from([[1, 2], [1, 4], [2, 5], [3, 4], [8, 3], [4, 5], [9, 4], [5, 6], [10, 5], [6, 7], [11, 6], [12, 7], [8, 9], [8, 13], [9, 10], [9, 14], [10, 11], [10, 15], [11, 12], [16, 11], [17, 12], [13, 14], [14, 15], [18, 14], [16, 15], [19, 15], [16, 17], [16, 20], [18, 19], [19, 20]]);
    
    
    let mut zxz = BlockZXZ::new(coupling_map);
    let u = generate_u(num_qubits);


    zxz.compute_decomposition(u, true, true, true, 0);
    
}
