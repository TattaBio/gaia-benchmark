use anyhow::{Context, Result};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use itertools::Itertools;
use qdrant_client::qdrant::{
    CreateCollectionBuilder, CreateFieldIndexCollectionBuilder, Datatype, Distance, FieldType,
    HnswConfigDiffBuilder, KeywordIndexParamsBuilder, PointStruct, UpsertPointsBuilder,
    VectorParamsBuilder,
};
use qdrant_client::qdrant::{OptimizersConfigDiffBuilder, UpdateCollectionBuilder};
use qdrant_client::Qdrant;
use uuid::Uuid;

mod glm;

const UPLOAD_CHUNK_SIZE: usize = 10000;

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

    create_collection(&client, &args.collection_name, args.embedding_dim)
        .await
        .context("Failed to create collection")?;
    upload(
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

/// Create a new Qdrant collection with the specified name and embedding dimension.
pub async fn create_collection(client: &Qdrant, name: &str, embedding_dim: usize) -> Result<()> {
    if client.collection_exists(name).await? {
        println!("collection exists, deleting...");
        client.delete_collection(name).await?;
    }

    client
        .create_collection(
            CreateCollectionBuilder::new(name)
                .vectors_config(
                    VectorParamsBuilder::new(embedding_dim as u64, Distance::Cosine)
                        .datatype(Datatype::Float16)
                        .on_disk(true),
                )
                .on_disk_payload(true)
                .hnsw_config(HnswConfigDiffBuilder::default().on_disk(true))
                // Qdrant defaults to something higher, but we set the indexing threshold to 0 to disable indexing
                .optimizers_config(OptimizersConfigDiffBuilder::default().indexing_threshold(0))
                .shard_number(2),
        )
        .await?;

    // In an online discussion, Qdrant founders recommended creating payload indices when the collection is created.
    // This is in contrast to vector indices, which are created after the upload is complete.
    client
        .create_field_index(
            CreateFieldIndexCollectionBuilder::new(name, "cds_id", FieldType::Keyword)
                .field_index_params(KeywordIndexParamsBuilder::default().on_disk(true)),
        )
        .await?;

    Ok(())
}

/// Upload embeddings to the Qdrant collection. The embeddings are read from an HDF5 file, which must contain two datasets:
/// `ids` and `embeddings`. Both datasets must have the same number of rows. If they do not, behavior is undefined.
///
/// Only the first `upload_limit` embeddings are uploaded. If `upload_limit` is `None`, all embeddings are uploaded.
pub async fn upload(
    client: &Qdrant,
    collection_name: &str,
    embeddings_file: &str,
    upload_limit: Option<usize>,
) -> Result<()> {
    let file = hdf5::File::open(embeddings_file)?; // open for reading
    let data = crate::glm::GlmData {
        ids: &file.dataset("ids")?,
        embeddings: &file.dataset("embeddings")?,
    };
    let n = upload_limit.unwrap_or(data.ids.shape()[0]);

    let point_structs = data
        .iter()
        .take(n)
        .map(|entry| {
            PointStruct::new(
                Uuid::new_v4().to_string(),
                entry.embedding,
                [("cds_id", entry.id.into())],
            )
        })
        .chunks(UPLOAD_CHUNK_SIZE);

    let pb = ProgressBar::new(n as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] ({pos}/{len}, ETA {eta})",
        )
        .unwrap(),
    );
    for chunk in &point_structs {
        let chunk = chunk.collect::<Vec<_>>();
        let len = chunk.len();
        client
            .upsert_points(UpsertPointsBuilder::new(collection_name, chunk))
            .await?;
        pb.inc(len as u64);
    }

    Ok(())
}

/// Re-enable vector indexing in the specified Qdrant collection.
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
