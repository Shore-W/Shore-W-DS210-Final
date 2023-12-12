use csv::ReaderBuilder;
use serde::Deserialize;
use std::error::Error;
use std::fs::File;

#[derive(Debug, Deserialize, Clone)]
pub struct Record {
    pub data: Vec<f64>,
}

#[derive(Debug)]
pub struct Graph {
    pub data: Vec<Record>,
}

impl Graph {
    pub fn new() -> Self {
        Graph { data: Vec::new() }
    }

    pub fn read_graph_from_file(file_path: &str) -> Result<Graph, Box<dyn Error>> {
        let file = File::open(file_path)?;
        let mut graph = Graph::new();

        let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);
        for result in rdr.deserialize() {
            let record: Vec<f64> = result?;
            graph.data.push(Record { data: record });
        }

        Ok(graph)
    }

}
