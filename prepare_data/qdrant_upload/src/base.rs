use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use itertools::Itertools;
use qdrant_client::{
    qdrant::{
        CreateCollectionBuilder, CreateFieldIndexCollectionBuilder, Datatype, Distance, FieldType,
        HnswConfigDiffBuilder, KeywordIndexParamsBuilder, OptimizersConfigDiffBuilder, PointStruct,
        UpsertPointsBuilder, VectorParamsBuilder,
    },
    Qdrant,
};
use uuid::Uuid;

const CHUNK_SIZE: usize = 10000;

pub async fn upload(
    client: &Qdrant,
    collection_name: &str,
    embeddings_file: &str,
    n: Option<usize>,
) -> Result<()> {
    let file = hdf5::File::open(embeddings_file)?; // open for reading
    let plm = crate::plm::PlmData {
        ids: &file.dataset("ids")?,
        embeddings: &file.dataset("embeddings")?,
    };
    let n = n.unwrap_or(plm.ids.shape()[0]);

    let point_structs = plm
        .iter()
        .take(n)
        .map(|entry| {
            PointStruct::new(
                Uuid::new_v4().to_string(),
                entry.embedding,
                [("cds_id", entry.id.into())],
            )
        })
        .chunks(CHUNK_SIZE);

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
                .optimizers_config(OptimizersConfigDiffBuilder::default().indexing_threshold(0))
                .shard_number(2),
        )
        .await?;

    // Qdrant founders recommend creating payload indices when the collection is created,
    // unlike vector indices, which we disable until the upload is complete.
    client
        .create_field_index(
            CreateFieldIndexCollectionBuilder::new(name, "cds_id", FieldType::Keyword)
                .field_index_params(KeywordIndexParamsBuilder::default().on_disk(true)),
        )
        .await?;

    Ok(())
}
