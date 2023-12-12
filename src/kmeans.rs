use rand::Rng;
use rand::prelude::SliceRandom;
use crate::graph;



// Euclidean distance between two points
pub fn euclidean_distance(point1: &[f64], point2: &[f64]) -> f64 {
    point1.iter().zip(point2.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt()
}

// Extract features from the records for clustering
pub fn extract_features(records: &[graph::Record]) -> Vec<Vec<f64>> {
    records
        .iter()
        .map(|record| record.data.iter().skip(5).take(3).cloned().collect())
        .collect()
}

// Split data into training and test sets
pub fn split_data(records: Vec<graph::Record>, ratio: f64) -> (Vec<graph::Record>, Vec<graph::Record>) {
    let num_samples = records.len();
    let num_train = (num_samples as f64 * ratio).round() as usize;

    let mut indices: Vec<usize> = (0..num_samples).collect();
    indices.shuffle(&mut rand::thread_rng());

    let train_set: Vec<_> = indices.iter().take(num_train).map(|&i| records[i].clone()).collect();
    let test_set: Vec<_> = indices.iter().skip(num_train).map(|&i| records[i].clone()).collect();

    (train_set, test_set)
}

// Run K-Means clustering
pub fn run_kmeans(
    features: Vec<Vec<f64>>,
    num_clusters: usize,
) -> (Vec<usize>, Vec<Vec<f64>>) {
    // Ensure there are at least as many data points as clusters
    assert!(
        features.len() >= num_clusters,
        "Not enough data points for the given number of clusters"
    );

    // Initialize centroids randomly
    let mut rng = rand::thread_rng();
    let mut centroids: Vec<Vec<f64>> = Vec::new();
    for _ in 0..num_clusters {
        let random_index = rng.gen_range(0..features.len());
        centroids.push(features[random_index].clone());
    }

    // Run K-Means iterations
    let mut assignments: Vec<usize> = vec![0; features.len()];
    for _ in 0..10 {
        // Assign each point to the nearest centroid
        for (i, feature) in features.iter().enumerate() {
            let mut min_distance = f64::INFINITY;
            let mut cluster = 0;

            for (j, centroid) in centroids.iter().enumerate() {
                let distance = euclidean_distance(feature, centroid);
                if distance < min_distance {
                    min_distance = distance;
                    cluster = j;
                }
            }

            assignments[i] = cluster;
        }

        // Update centroids based on the mean of assigned points
        for j in 0..num_clusters {
            let cluster_points: Vec<&Vec<f64>> = features
                .iter()
                .zip(assignments.iter())
                .filter(|&(_, &cluster)| cluster == j)
                .map(|(point, _)| point)
                .collect();
        
            if !cluster_points.is_empty() {
                let mean: Vec<f64> = cluster_points[0]
                    .iter()
                    .enumerate()
                    .map(|(i, _)| {
                        cluster_points.iter().map(|v| v[i]).sum::<f64>() / cluster_points.len() as f64
                    })
                    .collect();
                centroids[j] = mean;
            }
        }
    }

    (assignments, centroids)
}


pub fn compute_mean_squared_error(
    data_points: &[Vec<f64>],
    assignments: &[usize],
    centroids: &[Vec<f64>],
) -> f64 {

    // Check if the lengths match
    if data_points.len() != assignments.len() {
        panic!("Lengths of data points and assignments must be the same.");
    }

    let mut squared_error_sum = 0.0;

    for (data_point, &cluster) in data_points.iter().zip(assignments.iter()) {
        let centroid = &centroids[cluster];
        squared_error_sum += euclidean_distance(data_point, centroid).powi(2);
    }

    squared_error_sum / data_points.len() as f64
}

pub fn calculate_silhouette_score(
    features: &[Vec<f64>],
    assignments: &[usize],
    centroids: &[Vec<f64>],
) -> f64 {
    let num_samples = features.len();

    let mut silhouette_sum = 0.0;

    for i in 0..num_samples {
        let a_i = calculate_average_distance_to_other_points_in_cluster(i, features, assignments);

        let b_i = calculate_average_distance_to_nearest_cluster(i, features, assignments, centroids);

        let s_i = if a_i < b_i {
            1.0 - (a_i / b_i)
        } else if a_i > b_i {
            (b_i / a_i) - 1.0
        } else {
            0.0
        };

        silhouette_sum += s_i;
    }

    silhouette_sum / num_samples as f64
}

// Helper function to calculate average distance to other points in the same cluster
fn calculate_average_distance_to_other_points_in_cluster(
    index: usize,
    features: &[Vec<f64>],
    assignments: &[usize],
) -> f64 {
    let cluster = assignments[index];
    let mut distance_sum = 0.0;
    let mut count = 0;

    for (i, &assignment) in assignments.iter().enumerate() {
        if assignment == cluster && i != index {
            distance_sum += euclidean_distance(&features[index], &features[i]);
            count += 1;
        }
    }

    if count > 0 {
        distance_sum / count as f64
    } else {
        0.0
    }
}

// Helper function to calculate average distance to nearest cluster
fn calculate_average_distance_to_nearest_cluster(
    index: usize,
    features: &[Vec<f64>],
    assignments: &[usize],
    centroids: &[Vec<f64>],
) -> f64 {
    let cluster = assignments[index];
    let mut min_distance = f64::INFINITY;

    for (i, _) in centroids.iter().enumerate() {
        if assignments[i] != cluster {
            let distance = euclidean_distance(&features[index], &centroids[i]);
            min_distance = min_distance.min(distance);
        }
    }

    min_distance
}
// Function to normalize features
pub fn min_max_scaling(features: &mut Vec<Vec<f64>>) {
    for i in 0..features[0].len() {
        let min_val = features.iter().map(|x| x[i]).fold(f64::INFINITY, |a, b| a.min(b));
        let max_val = features.iter().map(|x| x[i]).fold(f64::NEG_INFINITY, |a, b| a.max(b));

        for j in 0..features.len() {
            features[j][i] = (features[j][i] - min_val) / (max_val - min_val);
        }
    }
}
