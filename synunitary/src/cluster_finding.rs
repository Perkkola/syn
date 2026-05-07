use std::collections::{HashMap, HashSet, VecDeque};

use itertools::Itertools;
use num_integer::binomial;

pub fn bfs_distances(source: i64, neighbors: &HashMap<i64, HashSet<i64>>) -> HashMap<i64, i64> {
    let mut dist = HashMap::from([(source, 0 as i64)]);
    let mut queue = VecDeque::from([source]);

    while !queue.is_empty() {
        let node = queue.pop_front().unwrap(); // popleft()
        for nb in neighbors.get(&node).unwrap_or(&HashSet::new()) {
            if !dist.contains_key(nb) {

                dist.insert(*nb, dist.get(&node).unwrap() + 1);
                queue.push_back(*nb);
            }
        }
    }
    dist
}

pub fn all_pairs_distances(neighbors: &HashMap<i64, HashSet<i64>>) -> HashMap<i64, HashMap<i64, i64>> {
    let mut dists: HashMap<i64, HashMap<i64, i64>> = HashMap::new();
    for node in neighbors.keys() {
        dists.insert(*node, bfs_distances(*node, neighbors));
    }
    dists
}

pub fn cluster_cost(nodes: &Vec<i64>, dist: &HashMap<i64, HashMap<i64, i64>>) -> f64 {
    let mut total = 0.0;
    for i in 0..nodes.len() {
        for j in i+1..nodes.len() {
            let def: HashMap<i64, i64> = HashMap::new();
            let d = dist.get(&nodes[i]).unwrap_or(&def).get(&nodes[j]).unwrap_or(&i64::MAX);

            if *d == i64::MAX { return f64::MAX }
            total += *d as f64;
        }
    }
    total
}

pub fn order_cluster(cluster: &Vec<i64>, dist: &HashMap<i64, HashMap<i64, i64>>) -> Vec<i64> {
    let mut remaining: HashSet<i64>= cluster.into_iter().copied().collect();

    let total_dist = |a|  {
        let sum: i64 = cluster.into_iter().filter(|x| **x != a).map(|y| *dist.get(&a).unwrap().get(y).unwrap_or(&i64::MAX)).sum();
        sum
    };

    let center = remaining.clone().into_iter().min_by_key(|x| (total_dist(*x), *x)).unwrap();

    let mut ordered = Vec::from([center]);
    remaining.remove(&center);

    while !remaining.is_empty() {
        let cost_to_ordered = |b: i64| {
            let sum: i64 = ordered.clone().into_iter().map(|x| dist.get(&b).unwrap().get(&x).unwrap_or(&i64::MAX)).sum();
            sum
        };
        let nxt = remaining.clone().into_iter().min_by_key(|x| (cost_to_ordered(*x), *x)).unwrap();
        ordered.push(nxt);
        remaining.remove(&nxt);
    }
   

    ordered
}

pub fn find_closest_cluster_exact(n: i64, neighbors: &HashMap<i64, HashSet<i64>>) -> (Vec<i64>, HashMap<i64, HashMap<i64, i64>>) {
    let dists = all_pairs_distances(neighbors);
    let all_nodes: Vec<&i64> = neighbors.keys().collect();

    let mut best_cluster: Vec<i64> = Vec::new();
    let mut best_cost = i64::MAX;
    for combo in all_nodes.iter().combinations(n as usize) {
        let mapped: Vec<i64> = combo.iter().map(|x| ***x).collect();
        let cost = cluster_cost(&mapped, &dists);

        if cost < best_cost as f64 {
            best_cost = cost as i64;
            best_cluster = mapped.to_owned();
        }
}

    (order_cluster(&best_cluster, &dists), dists)
}

pub fn find_closest_cluster_greedy(n: i64, neighbors: &HashMap<i64, HashSet<i64>>) -> (Vec<i64>, HashMap<i64, HashMap<i64, i64>>) {
    let dists = all_pairs_distances(neighbors);
    let all_nodes: Vec<&i64> = neighbors.keys().collect();

    let mut best_pair: Vec<i64> = Vec::new();
    let mut best_pair_score = i64::MAX;

    for u in all_nodes.clone() {
        for v in neighbors.get(u).unwrap() {
            if u >= v { continue; }

            let score: i64 = all_nodes.clone().iter().map(| w | 
                dists.get(u).unwrap().get(*w).unwrap_or(&i64::MAX) + 
                dists.get(v).unwrap().get(*w).unwrap_or(&i64::MAX)).sum();

            if score < best_pair_score {
                best_pair_score = score;
                best_pair = Vec::from([*u, *v]);
            }
        }
    }

    let mut cluster = best_pair.clone();
    let mut in_cluster: HashSet<i64>= cluster.clone().into_iter().collect();

    let mut candidate_cost: HashMap<i64, i64> = HashMap::new();

    for node in all_nodes.clone() {
        if !in_cluster.contains(node) {
            candidate_cost.insert(*node, 
            cluster.clone().iter().map(|c| dists.get(node).unwrap().get(c).unwrap_or(&i64::MAX)).sum()
            );
        }
    }

    while cluster.len() < n as usize {
        let candidate_cost_clone = candidate_cost.clone();
        let best_node = candidate_cost_clone.iter().min_by_key(|x| x.1).unwrap().0;
        cluster.push(*best_node);
        in_cluster.insert(*best_node);

        candidate_cost.remove(best_node);

        for node in candidate_cost.clone().keys() {
            candidate_cost.entry(*node).and_modify(|x| *x += dists.get(node).unwrap().get(best_node).unwrap_or(&i64::MAX));
        }
    }

    (order_cluster(&cluster, &dists), dists)
}

pub fn find_closest_cluster(n: i64, neighbors: &HashMap<i64, HashSet<i64>>, method: &str) -> (Vec<i64>, HashMap<i64, HashMap<i64, i64>>) {
    let num_nodes = neighbors.keys().len();
    let mut use_method = "exact";

    if method == "auto" {
        if binomial(num_nodes, n as usize) <= 50000 {
            use_method = "exact";
        } else {
            use_method = "greedy";
        }
    }

    if use_method == "exact" { return find_closest_cluster_exact(n, neighbors) }
    else { return find_closest_cluster_greedy(n, neighbors)}
}