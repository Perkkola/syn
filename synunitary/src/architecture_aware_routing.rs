use std::fmt::{Display, Formatter, Error};
use std::{collections::{HashMap, HashSet, VecDeque}};
use itertools::Itertools;

use crate::utils::{Gate, get_path};
use crate::cluster_finding::{find_closest_cluster};

#[derive(Clone)]
pub struct RoutedMultiplexer {
    multiplexer_angles: Vec<f64>,
    num_qubits: i64,
    num_control: i64,
    neighbors: HashMap<i64, HashSet<i64>>,
    root: i64,
    furthest_node: i64,
    arch_to_grey_map: HashMap<i64, usize>,
    grey_to_arch_map: HashMap<usize, i64>,
    optimal_neighborhood: HashMap<i64, Vec<i64>>,
    pairwise_dists: HashMap<i64, HashMap<i64, i64>>,
    gray_code: Vec<i64>,
    gray_gate_queue: VecDeque<Gate>,
    gray_state_queue: VecDeque<i64>,
    arch_qubits: HashSet<i64>,
    gate_queue: VecDeque<Gate>,
    discovered_pp_terms: HashSet<i64>,
    state: HashMap<i64, i64>,
    state_to_angle_dict: HashMap<i64, f64>
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

        RoutedMultiplexer { 
            multiplexer_angles: Vec::new(), 
            num_qubits, 
            num_control: num_qubits - 1, 
            neighbors: neighbors, 
            root: 0, 
            furthest_node: 0, 
            arch_to_grey_map: HashMap::new(), 
            grey_to_arch_map: HashMap::new(), 
            optimal_neighborhood: HashMap::new(), 
            pairwise_dists: HashMap::new(), 
            gray_code: Vec::new(), 
            gray_gate_queue: VecDeque::new(), 
            gray_state_queue: VecDeque::new(),
            arch_qubits: HashSet::new(),
            gate_queue: VecDeque::new(),
            discovered_pp_terms: HashSet::new(),
            state: HashMap::new(),
            state_to_angle_dict: HashMap::new()

        }
    }

    pub fn find_optimal_neighborhood_closest_cluster(&mut self) -> (HashMap<i64, Vec<i64>>, HashMap<usize, i64>) {
        let (cluster_e, dists) = find_closest_cluster(self.num_qubits, &self.neighbors, "auto");
        self.pairwise_dists = dists;
        let root = cluster_e[0];

        let mut paths: HashMap<i64, Vec<i64>> = HashMap::from([(root, Vec::from([root]))]);

        for node in cluster_e.clone() {
            if node != root {
                let cluster_e_set: HashSet<i64> = cluster_e.clone().into_iter().collect();
                let path = get_path(&self.neighbors, &cluster_e_set, root, node);
                paths.insert(node, path);

            }
        }
        let grey_to_arch: HashMap<usize, i64> = cluster_e.iter().copied().enumerate().collect();

        (paths, grey_to_arch)
    }

    pub fn recompute_optimal_neighborhood(&mut self) {
        let mut new_optimal_neighborhood: HashMap<i64, Vec<i64>> = HashMap::new();
        for node in self.optimal_neighborhood.keys() {
            let path = get_path(&self.neighbors, &self.arch_qubits, self.root, *node);
            new_optimal_neighborhood.insert(*node, path);
        }
        self.optimal_neighborhood = new_optimal_neighborhood;
    }

    pub fn map_grey_qubits_to_arch_unitary_synth(&mut self) {
        let (optimal_neighborhood, grey_to_arch_map) = self.find_optimal_neighborhood_closest_cluster();
        self.optimal_neighborhood = optimal_neighborhood;
        self.grey_to_arch_map = grey_to_arch_map;

        for (key, value) in self.grey_to_arch_map.clone().iter() {
            self.arch_to_grey_map.insert(*value, *key);
        }

        let arch_qubits: HashSet<i64> = self.grey_to_arch_map.clone().values().copied().into_iter().collect();

        self.arch_qubits = arch_qubits;
        self.root = *self.grey_to_arch_map.get(&(self.num_control as usize)).unwrap();
        self.furthest_node = *self.grey_to_arch_map.get(&0).unwrap();
    }

    pub fn long_range_cnot_cost(&mut self, dist: i64) -> i64 {
        if dist > 1 { 4 * dist - 4 }
        else { 1 }
    }

    pub fn get_optimal_gray_code(&mut self) -> i64 {
        let mut dists: HashMap<i64, i64> = HashMap::new();

        for node in self.arch_to_grey_map.clone().keys() {
            if *node == self.root { continue; }
            dists.insert(*node, *self.pairwise_dists.clone().get(&self.root).unwrap().get(node).unwrap());
        }

        let mut dists_list: Vec<(i64, i64)> = dists.keys().copied().zip(dists.values().copied()).collect();
        dists_list.sort_by_key(|x| x.1);

        let mut arch_cnots: HashMap<usize, (i64, i64)> = HashMap::new();

        for (index, node) in dists_list.iter().enumerate() {
            arch_cnots.insert(index, *node);
        }

        let upper_bound: i64 = (2i64).pow(self.num_control as u32) + 1;

        let mut grey_state: HashMap<i64, i64> = (0..self.num_qubits).map(|x| (x, 1 << x as i64)).into_iter().collect();

        for i in 1..upper_bound {
            let mut highest_index_diff: i64 = 0;

            for j in 0..self.num_control {
                if (i >> j) & 1 != ((i - 1) >> j) & 1 { highest_index_diff = j + 1; }

            }
            let cnot = arch_cnots.get(&((highest_index_diff - 1) as usize)).unwrap();
            self.gray_gate_queue.push_back(
                Gate{
                    name: "CX", 
                    ctrl: *self.arch_to_grey_map.get(&cnot.0).unwrap(), 
                    targ: *self.arch_to_grey_map.get(&self.root).unwrap(), 
                    value: 0.0}
            );
            self.gray_state_queue.push_back(*grey_state.get(&self.num_control).unwrap());
            let applied_cnot = grey_state.get(&(*self.arch_to_grey_map.get(&cnot.0).unwrap() as i64)).unwrap();
            let current_entry = grey_state.get(&self.num_control).unwrap();
            grey_state.insert(self.num_control, *applied_cnot ^ current_entry);
        }

        let grey_code: Vec<i64> = self.gray_state_queue.clone().into_iter().map(|x| x ^ (1 << self.num_control)).collect();
        self.gray_code = grey_code;

        let mut cnots: HashMap<i64, i64> = HashMap::from([(self.num_control - 1, 2)]);

        for i in 0..self.num_qubits - 2 {
            cnots.insert(self.num_control -2 - i, 2);
            for j in 0..i {
                cnots.entry(self.num_control - 1 - j).and_modify(|x| *x *= 2);
            }
        }

        let total_cost = dists_list.clone().into_iter().map(|x| x.1).zip(cnots.values().copied()).fold(0, |x, y| x + y.1 * self.long_range_cnot_cost(y.0));
        total_cost
    }

    pub fn cancel_or_append(&mut self, cnot: Gate, ignore: bool) {
        let prev_gate = self.gate_queue.pop_back().unwrap();
        let mut cancelled = true;

        if prev_gate != cnot {
            self.gate_queue.push_back(prev_gate);
            self.gate_queue.push_back(cnot);
            cancelled = false;
        }

        if cnot.targ == self.num_control as usize {
            if !self.discovered_pp_terms.contains(self.state.get(&self.num_control).unwrap()) {
                self.discovered_pp_terms.insert(*self.state.get(&self.num_control).unwrap());
                self.gate_queue.push_back(Gate {
                    name: "RZ",
                    ctrl: 0,
                    targ: self.num_control as usize,
                    value: *self.state_to_angle_dict.get(&self.state.get(&self.num_control).unwrap()).unwrap()
                });
            }
            else if ignore == true && self.discovered_pp_terms.contains(self.state.get(&self.num_control).unwrap()) && !cancelled {
                self.gate_queue.pop_back();
                let state_control = self.state.get(&(cnot.ctrl as i64)).unwrap();
                let state_targ = self.state.get(&(cnot.targ as i64)).unwrap();
                self.state.insert(cnot.targ as i64, *state_control ^ *state_targ);
            }
        }
    }

    pub fn long_range_cnot(&mut self, arch_path: Vec<i64>, ignore: bool) {
        let grey_path: Vec<usize> = arch_path.into_iter().map(|x| self.arch_to_grey_map.get(&x).unwrap()).copied().rev().collect();
        let grey_path_dist = grey_path.len();

        for i in 0..grey_path_dist - 1 {
            let state_ctrl = self.state.get(&(grey_path[i] as i64)).unwrap();
            let state_targ = self.state.get(&(grey_path[i + 1] as i64)).unwrap();
            
            self.state.insert((i + 1) as i64, *state_ctrl ^ *state_targ);
            self.cancel_or_append(Gate {
                name: "CX",
                ctrl: grey_path[i],
                targ: grey_path[i + 1],
                value: 0.0
            }, ignore);
    }

        for j in (1..grey_path_dist - 3).rev() {
            let state_ctrl = self.state.get(&(grey_path[j - 1] as i64)).unwrap();
            let state_targ = self.state.get(&(grey_path[j] as i64)).unwrap();
            
            self.state.insert((j) as i64, *state_ctrl ^ *state_targ);
            self.cancel_or_append(Gate {
                name: "CX",
                ctrl: grey_path[j - 1],
                targ: grey_path[j],
                value: 0.0
            }, ignore);
    }

        for k in 1..grey_path_dist - 1 {
            let state_ctrl = self.state.get(&(grey_path[k] as i64)).unwrap();
            let state_targ = self.state.get(&(grey_path[k + 1] as i64)).unwrap();
            
            self.state.insert((k + 1) as i64, *state_ctrl ^ *state_targ);
            self.cancel_or_append(Gate {
                name: "CX",
                ctrl: grey_path[k],
                targ: grey_path[k + 1],
                value: 0.0
            }, ignore);
    }

        for l in (2..grey_path_dist - 3).rev() {
                let state_ctrl = self.state.get(&(grey_path[l - 1] as i64)).unwrap();
                let state_targ = self.state.get(&(grey_path[l] as i64)).unwrap();
                
                self.state.insert((l) as i64, *state_ctrl ^ *state_targ);
                self.cancel_or_append(Gate {
                    name: "CX",
                    ctrl: grey_path[l - 1],
                    targ: grey_path[l],
                    value: 0.0
                }, ignore);
        }
}
    pub fn reset_state(&mut self) {
        for i in 0..self.num_qubits {
            if (self.state.get(&self.num_control).unwrap() >> i) & 1 == 1 {
                let arch_qubit = self.grey_to_arch_map.get(&(i as usize)).unwrap();
                let arch_path = self.optimal_neighborhood.get(arch_qubit).unwrap();
                self.long_range_cnot(arch_path.clone(), false);
            } 
        }
    }

    pub fn find_missing_term(&mut self, unfound_terms: HashSet<i64>) {}

    pub fn execute_gates(&mut self) -> (usize, VecDeque<Gate>) {
        let pp_terms: HashSet<i64> = (((2i64).pow(self.num_control as u32))..((2i64).pow(self.num_qubits as u32))).into_iter().collect();

        self.discovered_pp_terms = HashSet::from([(2i64).pow(self.num_control as u32)]);
        self.state = (0..self.num_qubits).map(|x| (x, 1 << x)).collect();
        self.state_to_angle_dict = self.gray_state_queue.clone().iter().copied().zip(self.multiplexer_angles.clone()).collect();

        let init_state = self.state.clone();

        self.gate_queue.push_back(Gate { 
            name: "RZ",
            ctrl: 0, 
            targ: self.num_control as usize, 
            value: self.multiplexer_angles[0] 
        });

        for gate in self.gray_gate_queue.clone() {
            let ctrl_qubit = gate.ctrl;
            let arch_qubit = self.grey_to_arch_map.get(&ctrl_qubit).unwrap();
            let arch_path = self.optimal_neighborhood.get(arch_qubit).unwrap();

            let dist = arch_path.len() - 1;
            let ignore = dist <= 1;
            self.long_range_cnot(arch_path.clone(), ignore);
        }

        if init_state != self.state {
            self.reset_state();
        }

        let unfound_terms: HashSet<i64> = pp_terms.difference(&self.discovered_pp_terms).copied().collect();

        if unfound_terms.len() > 0 {
            self.reset_state();
            self.find_missing_term(unfound_terms);
        }

        let circuit_length  = self.gate_queue.clone().into_iter().filter(|x| x.name != "RZ").try_len().unwrap_or(0);

        if self.discovered_pp_terms.len() != pp_terms.len() {
            print!("Found {}/{} phase polynomial terms.", self.discovered_pp_terms.len(), pp_terms.len());
        }

        if self.state != init_state {
            print!("State was bit reset cirrectly!");
        }

        (circuit_length, self.gate_queue.clone())
    }

    pub fn replace_mapped_angles(&mut self) {}
}