use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use qdrant_client::qdrant::{OptimizersConfigDiffBuilder, UpdateCollectionBuilder};
use qdrant_client::Qdrant;

mod base;
mod plm;

/// Struct to hold CLI arguments for uploading to a Qdrant collection.
#[derive(Parser, Debug)]
#[command(name = "Qdrant Uploader")]
#[command(author = "Your Name <your.email@example.com>")]
#[command(version = "1.0")]
#[command(about = "Uploads embeddings to a Qdrant collection")]
struct Args {
    /// Name of the new Qdrant collection
    #[arg()]
    collection_name: String,

    /// Path to the embeddings .h5 file
    #[arg(short = 'f', long, value_name = "FILE", required = true)]
    embeddings_file: String,

    /// Qdrant server URL
    #[arg(short = 'u', long, default_value = "http://localhost:6334")]
    qdrant_url: String,

    /// Dimension of the embeddings vectors
    #[arg(short = 'd', long, default_value_t = 512)]
    embedding_dim: usize,

    /// Only upload the first n embeddings
    #[arg(short = 'n', long)]
    n_embeddings: Option<usize>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let mut client_builder = Qdrant::from_url(&args.qdrant_url);
    
    // Only add API key if it's set in the environment
    if let Ok(api_key) = std::env::var("API_KEY") {
        client_builder = client_builder.api_key(api_key);
    }

    let client = client_builder
        .build()
        .context("Failed to build Qdrant Client")?;

    crate::base::create_collection(&client, &args.collection_name, args.embedding_dim)
        .await
        .context("Failed to create collection")?;
    crate::base::upload(
        &client,
        &args.collection_name,
        &args.embeddings_file,
        args.n_embeddings,
    )
    .await?;

    reenable_indexing(&client, &args.collection_name)
        .await
        .context("Failed to re-enable indexing in database")?;

    Ok(())
}

async fn reenable_indexing(client: &Qdrant, name: &str) -> Result<()> {
    client
        .update_collection(
            UpdateCollectionBuilder::new(name).optimizers_config(
                OptimizersConfigDiffBuilder::default().indexing_threshold(20000),
            ),
        )
        .await?;
    Ok(())
}
