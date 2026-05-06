use std::{collections::{HashMap, HashSet, VecDeque}, hash::Hash};

use faer::utils::bound::Array;

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
    let mut remaining: HashSet<i64>= HashSet::new();
    for node in cluster { remaining.insert(*node); }

    let total_dist = |a|  {
        let sum: i64 = cluster.into_iter().filter(|x| **x != a).map(|y| *dist.get(&a).unwrap().get(y).unwrap_or(&i64::MAX)).sum();
        sum
    };

    let center = remaining.clone().into_iter().min_by_key(|x| total_dist(*x)).unwrap();

    let mut ordered = Vec::from([center]);
    remaining.remove(&center);

    while !remaining.is_empty() {
        let cost_to_ordered = |b: i64| {
            let sum: i64 = ordered.clone().into_iter().map(|x| dist.get(&b).unwrap().get(&x).unwrap_or(&i64::MAX)).sum();
            sum
        };
        let nxt = remaining.clone().into_iter().min_by_key(|x| cost_to_ordered(*x)).unwrap();
        ordered.push(nxt);
        remaining.remove(&nxt);
    }
   

    ordered
}