mod graph;
mod kmeans;
mod plot;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    if let Err(err) = run() {
        eprintln!("Error: {}", err);
    }
    Ok(())
}

fn run() -> Result<(), Box<dyn Error>> {
    let file_path = r#"C:\Users\hyper\OneDrive\Desktop\College\DS 210\Final Project\final_code\john snow.csv"#;
    //ADAPT THIS TO FIT YOUR FILE LOCATION

    let graph_data = graph::Graph::read_graph_from_file(file_path)?;

    let num_clusters = 9; //CHANGE THIS AS NEEDED TO TEST FUNCITONALITY
    let num_iterations = 10; //CHANGE THIS AS NEEDED TO TEST FUNCITONALITY
    let train_test_ratio = 0.75; //CHANGE THIS AS NEEDED TO TEST FUNCITONALITY


    let mut test_deaths = 0;

    //variables to store copies of other outputs so that the program doesn't erase the values
    let mut real_test_set: Vec<_> = Vec::new();
    let mut sil_test_set:Vec<_> = Vec::new();

    
    //these are storage variables for the outputs of the training iterations with the best silhouette score
    let mut best_iteration = 0;
    let mut best_silhouette_score = f64::NEG_INFINITY;
    let mut best_cluster_to_death = Vec::new();
    let mut best_mse = 0.0;
    let mut best_data_by_cluster = vec![Vec::new(); num_clusters];
    let mut best_centroids: Vec<Vec<f64>> = vec![Vec::new()];

    // Initialize variables
    let mut mse_results = Vec::new();

    //=================================================================================================================
    // KMEANS ITERATIONS

    for iteration in 0..num_iterations {
        // Splitting data and extracting main features
        let (train_set, test_set) = kmeans::split_data(graph_data.data.clone(), train_test_ratio);

        //this will extract the features we want to use for clustering, which are:
            // 1. Distance to Pestfield
            // 2. Distance to Sewer
            // 3. Distance to Broad St. Pump
        //afterwards, we scale this data using the min_max_scaling function in order to properly work with the data
        let mut train_extracted_features = kmeans::extract_features(&train_set);
        let mut test_extracted_features = kmeans::extract_features(&test_set);
        kmeans::min_max_scaling(&mut train_extracted_features);
        kmeans::min_max_scaling(&mut test_extracted_features);

        real_test_set = test_extracted_features;
        sil_test_set = test_set.clone(); 
        //we need to store these results later for processing

        //=================================================================================================================
        //PROCESSING TRAINING

        // Running kmeans clustering on the training set to get cluster assignments
        let (train_assignments, train_centroids) =
            kmeans::run_kmeans(train_extracted_features.clone(), num_clusters);

        // Calculating average deaths associated with each cluster in training set
        //it will store them in a vector that looks like this [(0,.576), (1, 0.677)...] which makes indexing into the cluster easy
        let mut sorted_cluster_to_death: Vec<_> = train_set
            .iter()
            .zip(train_assignments.iter())
            .fold(vec![None; num_clusters], |mut acc, (record, &cluster)| {
                let current_value = acc[cluster].get_or_insert((0.0, 0)).clone();
                let new_value = (current_value.0 + record.data[1], current_value.1 + 1);
                acc[cluster] = Some(new_value);
                acc
            })
            .into_iter()
            .enumerate()
            .filter_map(|(cluster, value)| {
                value.map(|(total_deaths, count)| (cluster, total_deaths / count as f64))
            })
            .collect();
    
            sorted_cluster_to_death.sort_by(|a, b| a.0.cmp(&b.0));
            //this sorts it properly by the first value inside of the pairs

        test_deaths = test_set.iter().map(|record| record.data[1] as i64).sum();
        //this will collect all the deaths from the test data set for comparison later

        //this computes the mean squared error for the iteration, and it's score is going to be quite low because of the scale of our data 
        let mse = kmeans::compute_mean_squared_error(
            &train_extracted_features,
            &train_assignments,
            &train_centroids,
        );

        mse_results.push(mse); //pushes results into mse_results that we will use later for calculcating the average

        println!("Clusters for Training Iteration {}:", iteration);

        //this for loop will iterate over the training set and count the number of nodes within each cluster, and print them out cluster by cluster
        for cluster_idx in 0..num_clusters {
            let node_count = train_set
                .iter()
                .zip(train_assignments.iter())
                .filter(|(_, &assignment)| assignment == cluster_idx)
                .count();
            println!("Cluster: {}, Nodes: {}", cluster_idx, node_count);
        }

        //since our mse is so small, we need another way of seeing how accurate our clustering is
        //the silhouette_score is an index from (-1..1) that reflects how well a group of data points is clustered compared to how far its distance is to other clusters' centroids
        //A score closer to one means that our points are well assigned, while a score closer to negative one means points are not properly assigned

        let silhouette_score = kmeans::calculate_silhouette_score(
            &kmeans::extract_features(&train_set),
            &train_assignments,
            &train_centroids,
        );

        //since we want the characteristics of the iteraiton with the best clustering, we need to save it's information in these variables for later use
        if silhouette_score > best_silhouette_score {
            best_silhouette_score = silhouette_score;
            best_iteration = iteration;
            best_cluster_to_death = sorted_cluster_to_death.clone();
            best_mse = mse;
            best_centroids = train_centroids.clone();

            best_data_by_cluster = train_extracted_features
                .iter()
                .zip(train_assignments.iter())
                .fold(vec![Vec::new(); num_clusters], |mut acc, (scaled_feature, &cluster)| {
                    while acc.len() <= cluster {
                        acc.push(Vec::new());
                    }
                    acc[cluster].push((scaled_feature[0], scaled_feature[1], scaled_feature[2]));
                    acc
                });
        }

        println!("Silhouette Score for Iteration {}: {:4}", iteration, silhouette_score);
        println!("MSE this Iteration: {:4}", mse);

        println!();
    }

    let average_mse: f64 = mse_results.iter().sum::<f64>() / num_iterations as f64;

    //=================================================================================================================
    //PRINTING THE TRAINING RESULTS

    println!("===============================================");
    println!("TRAINING RESULTS:");
    println!();
    println!(
        "BEST ITERATION: {}\n Silhouette Score of {:.4}\n MSE of {:.4} (scaled)",
        best_iteration, best_silhouette_score, best_mse
    );

    println!("Average deaths associated with each cluster: {:?}", best_cluster_to_death);
    println!("Average MSE: {}", average_mse);

    println!("Check Project Folder for TrainingResults.png of the best iteration");

    //this will call RUST PLOTTERS to create a 3d representation of the clustering iteration with the best results
    plot::train_plotting(&best_data_by_cluster, num_clusters, &[best_centroids], best_iteration)?;
    println!();

    //=================================================================================================================
    //TESTING THE ACCURACY OF TRAINING

    println!("===============================================");
    println!("TEST RESULTS");

    println!("Death Map used:");
    println!("{:?}", best_cluster_to_death); //this contains the average deaths in the clusters that we will then multiply our future cluster nodes by
    println!();
    
    //these variables will store the total difference across iterations, which iteration had the lowest difference, and set up a variable to store the silhouette score  
    let mut total_difference = 0; 
    let mut best_death_iteration = 0;
    let mut lowest_deaths = 100.0;
    let mut test_sil = 0.0;

    //these will store our values that will then be used for graphing the best iteration of our tests
    let mut graph_test_data_by_cluster = vec![Vec::new(); num_clusters];
    let mut graph_test_centroids: Vec<Vec<f64>> = vec![Vec::new()];

    //BEGINING OF TEST ITERATIONS
    for i in 0..num_iterations {
        let mut test_predicted_deaths = 0;
        println!("Test Iteration {}", i);

        //for each iteration we will generate new centroids and new test assignments for the test split of our data
        let (test_assignments, test_centroids) = kmeans::run_kmeans(real_test_set.clone(), num_clusters);

        for cluster_idx in 0..num_clusters {
            let node_count = real_test_set
                .iter()
                .zip(test_assignments.iter())
                .filter(|(_, &assignment)| assignment == cluster_idx)
                .count();

            if let Some((_, avg_deaths)) = best_cluster_to_death.get(cluster_idx) {
                let predicted_deaths = node_count as f64 * avg_deaths;
                test_predicted_deaths += predicted_deaths as i32;
                //this will store the predicted deaths for the iteration
            } else {
                eprintln!("Warning: No data for cluster index {}", cluster_idx);
                //this will print a warning if there is no data for the cluster we are indexing
            }
        }

        //this stores the record data for each cluster, and the one corresponding to the best iteration gets used for graphing
        let test_data_by_cluster = real_test_set
            .iter()
            .zip(test_assignments.iter())
            .fold(vec![Vec::new(); num_clusters], |mut acc, (scaled_feature, &cluster)| {
                while acc.len() <= cluster {
                    acc.push(Vec::new());
                }
                acc[cluster].push((scaled_feature[0], scaled_feature[1], scaled_feature[2]));
                acc
            });

       
        //if we find an iteration with the lowest difference between our predicted deaths and actual deaths, its characteristics will be stored as the best of the bunch
        let it_difference = (test_deaths as i32 - test_predicted_deaths).abs();
        if it_difference <= lowest_deaths as i32 {
            lowest_deaths = it_difference as f64;
            best_death_iteration = i;
            graph_test_data_by_cluster = test_data_by_cluster;
            graph_test_centroids = test_centroids.clone();
            test_sil = kmeans::calculate_silhouette_score(
                &kmeans::extract_features(&sil_test_set),
                &test_assignments,
                &test_centroids,
            );
        } else {
            lowest_deaths = lowest_deaths;
            best_death_iteration = best_death_iteration;
            //otherwise, there are no changes to current values
        }

        //these print statements print the results of each test iteration 
        println!("Predicted Deaths based on Training Clustering: {}", test_predicted_deaths as i32);
        println!("Actual Test Deaths: {}", test_deaths);
        println!("Difference between actual and predicted: {}", (test_deaths - test_predicted_deaths as i64).abs());
        println!();

        total_difference += (test_deaths - test_predicted_deaths as i64).abs();
        plot::test_plotting(&graph_test_data_by_cluster, num_clusters, &[graph_test_centroids.clone()], best_death_iteration)?;
        //the iteration with the best values get graphed!
    }
    //these print statements reveal the best iteration, what the average difference is between iterations, and what other factors like the silhouette score are
    println!("Average Difference between Actual and Predicted deaths:");
    println!("{} Deaths", (total_difference / num_iterations as i64).abs());
    println!();
    println!("Iteration with the Lowest Difference: \n Iteration: {} \n Difference: {} \n Silhouette Score: {:.4} ", best_death_iteration, lowest_deaths, test_sil);
    println!("Check Project Folder for TestResults.png of this iteration ^");

    Ok(())
}



//=================================================================================================================
//TESTS FOR FINAL PROJECT

#[test]
fn test_3d_graphing() {
    //this tests if we get a OK result from the graph, letting me know it gets built
    let data_by_cluster = vec![vec![(1.0, 2.0, 3.0)], vec![(4.0, 5.0, 6.0)]];
    let num_clusters = 2;
    let centroids = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
    let iteration = 0;

    let result = plot::train_plotting(&data_by_cluster, num_clusters, &[centroids], iteration);

    assert!(result.is_ok());
}

#[test]
fn test_kmeans_run() {
    //this tests if we get a proper number of centroids and assignments from the kmeans clustering algorithm
    let data: Vec<Vec<f64>> = vec![
        vec![1.6, 2.5],
        vec![4.5, 6.7],
        vec![6.7, 8.9],
    ];

    let data_length = data.len();
    let num_clusters = 3;

    let (assignments, centroids) = kmeans::run_kmeans(data, num_clusters);

    assert_eq!(assignments.len(), data_length);
    assert_eq!(centroids.len(), num_clusters);
}

#[test]
fn test_split_data() {
    //this test checks if our function to split data is working 
    let data = vec![
        graph::Record { data: vec![1.0, 2.0] },
        graph::Record { data: vec![3.0, 4.0] },
        graph::Record { data: vec![5.0, 6.0] },
        graph::Record { data: vec![7.0, 8.0] },
        graph::Record { data: vec![9.0, 10.0] },
        graph::Record { data: vec![1.0, 2.0] },
        graph::Record { data: vec![3.0, 4.0] },
        graph::Record { data: vec![5.0, 6.0] },
        graph::Record { data: vec![7.0, 8.0] },
        graph::Record { data: vec![9.0, 10.0] },
    ];

    let ratio = 0.7;
    let (train_set, test_set) = kmeans::split_data(data.clone(), ratio);

    assert_eq!(train_set.len() + test_set.len(), data.len());
    assert_eq!(train_set.len(), 7);
    assert_eq!(test_set.len(), 3);
}
