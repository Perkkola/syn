use std::collections::{VecDeque, HashMap};
use std::env;
use faer::traits::ext::ComplexFieldExt;
use faer::traits::math_utils::sqrt;
use faer::{Mat, complex::Complex64, Scale, mat};
use synunitary::utils::{generate_u, angles_from_diag};
pub struct BlockZXZ {
    coupling_map: Vec<[i64; 2]>,
    gate_queue: VecDeque<(&'static str, i32, i32, f64)>,
    diag: Mat<Complex64>,
    routed_multiplexers: Vec<i64>,
    swaps_per_level: Vec<i64>,
    swap_maps: Vec<HashMap<i32, i32>>,
}

impl BlockZXZ {
    pub fn new(coupling_map: Vec<[i64; 2]>) -> Self {
        BlockZXZ { 
            coupling_map, 
            gate_queue: VecDeque::new(), 
            diag: Mat::<Complex64>::identity(1, 1), 
            routed_multiplexers: Vec::new(), 
            swaps_per_level: Vec::new(), 
            swap_maps: Vec::new() }
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

    pub fn compute_decomposition(mut self, 
        u: Mat<Complex64>, 
        init: Option<bool>, 
        rightmost_unitary: Option<bool>, 
        leftmost_unitary: Option<bool>, 
        recursion_level: Option<i32>) -> Self {
        
        let init = init.unwrap_or(false);
        let rightmost_unitary = rightmost_unitary.unwrap_or(false);
        let leftmost_unitary = leftmost_unitary.unwrap_or(false);
        let recursion_level = recursion_level.unwrap_or(0);

        let n = u.nrows();
        let num_qubits = (n as f64).log2().ceil() as i32;
        let target_qubit = num_qubits - 1;
        let block_len = n / 2;

        let x = u.get(0..block_len, 0..block_len);
        let y = u.get(0..block_len, block_len..);
        let u_21 = u.get(block_len.., 0..block_len);
        let u_22 = u.get(block_len.., block_len..);

        let svd_x = x.svd().expect("Something went wrong!");
        let V_x = svd_x.U();
        let sigma_x = svd_x.S();
        let W_x_dgr = svd_x.V();
        
        let S_x = V_x * sigma_x * V_x.conjugate().transpose();
        let U_x = V_x * W_x_dgr;

        let svd_y = y.svd().expect("Something went wrong!");
        let V_y = svd_y.U();
        let sigma_y = svd_y.S();
        let W_y_dgr = svd_y.V();

        let S_y = V_y * sigma_y * V_y.conjugate().transpose();
        let U_y = V_y * W_y_dgr;

        let c = Scale(-Complex64::i()) * &U_x.conjugate().transpose() * &U_y;
        let A_1 = (&S_x + Scale(Complex64::i()) * &S_y) * &U_x;
        let A_2 = &u_21 + &u_22 * (Scale(Complex64::i()) * &U_y.conjugate().transpose() * &U_x);

        let eye = Mat::<Complex64>::identity(block_len, block_len);
        let zeros = Mat::<Complex64>::zeros(block_len, block_len);

        let b = Scale(Complex64::new(2.0, 0.0)) * &A_1.conjugate().transpose() * &x - &eye;

        let (V_A, block_diag_A, W_A) = self.demultiplex(A_1, A_2);
        let (V_C, block_diag_C, W_C) = self.demultiplex(eye, c);
        
        let b_tilde = self.block(&(&W_A * &V_C), &zeros, &zeros, &(&W_A * &b * &V_C));


        let h = mat![[Complex64::new(1.0 / f64::sqrt(2.0), 0.0), Complex64::new(1.0 / f64::sqrt(2.0), 0.0)],
                                    [Complex64::new(1.0 / f64::sqrt(2.0), 0.0), Complex64::new(-1.0 / f64::sqrt(2.0), 0.0)]];

        print!("{block_diag_C:#?}\n");
        let angles_C = angles_from_diag(block_diag_C);
        // print!("{block_diag_A:#?}\n");
        print!("{angles_C:#?}\n");

        self
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let default = "3".to_string();
    let num_qubits = args.get(1).unwrap_or(&default).parse::<usize>().expect("Expected a number");
    let coupling_map: Vec<[i64; 2]> = Vec::from([[1, 2], [1, 4], [2, 5], [3, 4], [8, 3], [4, 5], [9, 4], [5, 6], [10, 5], [6, 7], [11, 6], [12, 7], [8, 9], [8, 13], [9, 10], [9, 14], [10, 11], [10, 15], [11, 12], [16, 11], [17, 12], [13, 14], [14, 15], [18, 14], [16, 15], [19, 15], [16, 17], [16, 20], [18, 19], [19, 20]]);
    
    
    let mut zxz = BlockZXZ::new(coupling_map);
    let u = generate_u(num_qubits);

    zxz.compute_decomposition(u, Some(false), Some(false), Some(false), Some(0));
   
}
