use plotters::prelude::*;
use plotters::style::RGBColor;
use rand::Rng;

//THIS MODULE WILL CONSTRUCT THE 3D GRAPHS USING RUST PLOTTERS

//to plot our training results, we will employ train_plotting
pub fn train_plotting(
    data_by_cluster: &[Vec<(f64, f64, f64)>], //this is the data associated with each cluster for the best iteration
    num_clusters: usize, 
    centroids: &[Vec<Vec<f64>>],
    best_iteration: usize 
) -> Result<(), Box<dyn std::error::Error>> {
    // Create a bitmap backend to store our png file in the project folder
    let root = BitMapBackend::new(
        r#"C:\Users\hyper\OneDrive\Desktop\College\DS 210\Final Project\final_code\TrainResults.png"#,
        //ADAPT THIS TO FIT YOUR MACHINE
        (1920, 1080), //the size of the image is 1920x1080 so that the image is readable
    )
    .into_drawing_area();
    root.fill(&WHITE)?;

    // Create a 3D scatter plot chart
    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .caption(
            format!( //title
                "TRAINING RESULTS: 3D Scatter Plot of Clustering Algorithm for John Snow Cholera Data Set:\n Iteration {}",
                best_iteration
            )
            .as_str(),
            ("sans-serif", 50),
        )
        .build_cartesian_3d(0.0..1.0, 0.0..1.0, 0.0..1.0)?;

    chart.configure_axes().draw()?;

    let mut rng = rand::thread_rng();

    for i in 0..num_clusters {
        // Generate a random color for each cluster
        let color = RGBColor(
            rng.gen_range(0..=255),
            rng.gen_range(0..=255),
            rng.gen_range(0..=255),
        );
        //this will generate a num_cluster amount of colors to color its corresponding data, ex. cluster 0 is green, cluster 1 is blue...

        chart.draw_series(
            data_by_cluster[i] //reads through each cluster and plots their data points
                .iter()
                .map(|point| Circle::new(*point, 5, &color)),
        )?
        .label(format!("Cluster {} Data Points ", i))
        .legend(move |(x, y)| {
            PathElement::new(vec![(x, y), (x + 20, y)], &color)
        });

        chart.draw_series(
            centroids // reads through each centroid and plots it along with its associated data
                .iter()
                .enumerate()
                .map(|(_j, centroid)| TriangleMarker::new(
                    (centroid[i][0], centroid[i][1], centroid[i][2]),
                    10,
                    &color,
                )),
        )?;
    }

    chart
        .configure_series_labels()
        .label_font(("sans-serif", 25).into_font()) // Set the font family and size
        .border_style(&BLACK) // Set the border color for the legend
        .background_style(&WHITE.mix(0.8)) // Set the background color and transparency
        .draw()?;

    Ok(())
}

//this is a very similar function to train_plotting, with the only difference being the title of the image produced. 
pub fn test_plotting(
    data_by_cluster: &[Vec<(f64, f64, f64)>],
    num_clusters: usize,
    centroids: &[Vec<Vec<f64>>],
    best_iteration: usize ) -> Result<(), Box<dyn std::error::Error>> {
    // Create a bitmap backend
    let root = BitMapBackend::new(
        r#"C:\Users\hyper\OneDrive\Desktop\College\DS 210\Final Project\final_code\TestResults.png"#,
        //ADAPT THIS TO FIT YOUR MACHINE
        (1920, 1080),
    )
    .into_drawing_area();
    root.fill(&WHITE)?;

    // Create a 3D scatter plot chart
    let mut chart = ChartBuilder::on(&root)
        .margin(20)
        .caption(
            format!(
                "TEST RESULTS: 3D Scatter Plot of Clustering Algorithm for John Snow Cholera Data Set: Iteration {}", best_iteration
            )
            .as_str(),
            ("sans-serif", 50),
        )
        .build_cartesian_3d(0.0..1.0, 0.0..1.0, 0.0..1.0)?;

    chart.configure_axes().draw()?;

    // Extract data for cluster 0
    let mut rng = rand::thread_rng();

    for i in 0..num_clusters {
        // Generate a random color for each cluster
        let color = RGBColor(
            rng.gen_range(0..=255),
            rng.gen_range(0..=255),
            rng.gen_range(0..=255),
        );

        chart.draw_series(
            data_by_cluster[i]
                .iter()
                .map(|point| Circle::new(*point, 5, &color)),
        )?
        .label(format!("Cluster {} Data Points (Triangle is the centroid)", i))
        .legend(move |(x, y)| {
            PathElement::new(vec![(x, y), (x + 20, y)], &color)
        });

        chart.draw_series(
            centroids
                .iter()
                .enumerate()
                .map(|(_j, centroid)| TriangleMarker::new(
                    (centroid[i][0], centroid[i][1], centroid[i][2]),
                    10,
                    &color,
                )),
        )?;
    }

    chart
        .configure_series_labels()
        .label_font(("sans-serif", 25).into_font()) // Set the font family and size
        .border_style(&BLACK) // Set the border color
        .background_style(&WHITE.mix(0.8)) // Set the background color and transparency
        .draw()?;

    Ok(())
}

