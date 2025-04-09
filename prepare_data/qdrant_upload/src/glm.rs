use anyhow::Context;
use hdf5::{types::VarLenUnicode, Dataset};
use ndarray::s;

pub struct GlmData<'a> {
    pub ids: &'a Dataset,
    pub embeddings: &'a Dataset,
}

pub struct GlmEntry {
    pub id: String,
    pub embedding: Vec<f32>,
}

pub struct ChunkedGlmDataIterator<'a> {
    data: &'a GlmData<'a>,
    cur_index: usize,
    chunk_size: usize,
}

impl GlmData<'_> {
    pub fn iter(&self) -> impl Iterator<Item = GlmEntry> + '_ {
        ChunkedGlmDataIterator {
            data: self,
            cur_index: 0,
            chunk_size: 1000, // Read from the HDF5 file in chunks of 1000
        }
        .flatten()
    }
}

impl<'a> Iterator for ChunkedGlmDataIterator<'a> {
    type Item = Vec<GlmEntry>;

    /// Read the next chunk of data (ids and embeddings) from the HDF5 file.
    fn next(&mut self) -> Option<Self::Item> {
        let min_length = *[self.data.ids.shape()[0], self.data.embeddings.shape()[0]]
            .iter()
            .min()?;
        let end_index = std::cmp::min(self.cur_index + self.chunk_size, min_length);

        let ids = self
            .data
            .ids
            .read_slice_1d::<VarLenUnicode, _>(s![self.cur_index..end_index])
            .context("Failed to read IDs")
            .ok()?;
        let ids = ids.iter().map(|id| id.to_string());

        let embeddings = self
            .data
            .embeddings
            .read_slice::<f32, _, ndarray::Dim<[usize; 3]>>(s![self.cur_index..end_index, .., ..])
            .context("Failed to read embeddings")
            .ok()?;
        let embeddings = embeddings
            .axis_iter(ndarray::Axis(0))
            .map(|x| x.as_slice().expect("Can convert Array to slice").to_vec());

        self.cur_index += self.chunk_size;
        Some(
            ids.zip(embeddings)
                .map(|(id, embedding)| GlmEntry { id, embedding })
                .collect::<Vec<_>>(),
        )
    }
}
