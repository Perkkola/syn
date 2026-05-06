use std::fmt::{Display, Formatter, Error};
use std::{collections::{HashMap, HashSet, VecDeque}};
use crate::utils::Gate;
use crate::cluster_finding::{order_cluster, all_pairs_distances};

pub struct RoutedMultiplexer {
    multiplexer_angles: Vec<f64>,
    coupling_map: Vec<[i64; 2]>,
    num_qubits: i64,
    num_control: i64,
    neighbors: HashMap<i64, HashSet<i64>>,
    vertices: Vec<i64>,
    root: i64,
    furthest_node: i64,
    arch_to_grey_map: HashMap<i64, i64>,
    grey_to_arch_map: HashMap<i64, i64>,
    optimal_neighborhood: HashMap<i64, Vec<i64>>,
    pairwise_dists: HashMap<i64, HashMap<i64, i64>>,
    gray_code: Vec<i64>,
    gray_gate_queue: VecDeque<Gate>,
    gray_state_queue: VecDeque<i64>
}

impl Display for RoutedMultiplexer {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        write!(f, "Num_qubits: {}, Root: {}, Gray_to_arch: {:#?}, Arch_to_gray: {:#?}, Optimal_ optimal_neighborhood: {:#?}", self.num_qubits, self.root, self.grey_to_arch_map, self.arch_to_grey_map, self.optimal_neighborhood)
    }
}

impl RoutedMultiplexer {
    pub fn new(coupling_map: Vec<[i64; 2]>, num_qubits: i64) -> Self {
        let mut neighbors: HashMap<i64, HashSet<i64>> = HashMap::new();

        for edge in &coupling_map {
            if neighbors.get(&edge[0]) == None {
                neighbors.insert(edge[0], HashSet::new());
            }
            if neighbors.get(&edge[1]) == None {
                neighbors.insert(edge[1], HashSet::new());
            }

            let mut set_a = neighbors.get(&edge[0]).unwrap().to_owned();
            set_a.insert(edge[1]);
            neighbors.insert(edge[0], set_a);

            let mut set_a = neighbors.get(&edge[1]).unwrap().to_owned();
            set_a.insert(edge[0]);
            neighbors.insert(edge[1], set_a);
        }

        let dists = all_pairs_distances(&neighbors);
        let cluster = Vec::from([1, 2, 4, 5, 3]);
        let ordered = order_cluster(&cluster, &dists);
        print!("{ordered:#?}");
        RoutedMultiplexer { 
            multiplexer_angles: Vec::new(), 
            coupling_map, 
            num_qubits, 
            num_control: num_qubits - 1, 
            neighbors: neighbors, 
            vertices: Vec::new(), 
            root: 0, 
            furthest_node: 0, 
            arch_to_grey_map: HashMap::new(), 
            grey_to_arch_map: HashMap::new(), 
            optimal_neighborhood: HashMap::new(), 
            pairwise_dists: HashMap::new(), 
            gray_code: Vec::new(), 
            gray_gate_queue: VecDeque::new(), 
            gray_state_queue: VecDeque::new() 
        }
    }

    // pub fn find_optimal_neighborhood_closest_cluster(&self) -> (HashMap<i64, Vec<i64>, HashMap<i64, i64>>) {
    //     let (cluster_e, dists) = find_closest_cluster(self.num_qubits, &self.coupling_map, &self.neighbors, "auto".to_string());
    //     (HashMap::new(), HashMap::new())
    // }

}