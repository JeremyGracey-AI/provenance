import numpy as np
import pytest

from provenance.backends.page_index import IndexedPage, PageIndex


def _pages(n: int) -> list[IndexedPage]:
    return [IndexedPage(doc_id="d", page_number=i + 1, image_file=f"d_p{i + 1}.png") for i in range(n)]


def test_save_load_roundtrip(tmp_path):
    embeddings = np.random.rand(3, 8).astype("float32")
    index = PageIndex(_pages(3), embeddings)
    emb_path, man_path = tmp_path / "index.npy", tmp_path / "manifest.json"
    index.save(emb_path, man_path)

    loaded = PageIndex.load(emb_path, man_path)
    assert [p.id for p in loaded.pages] == ["d#p1", "d#p2", "d#p3"]
    assert loaded.dim == 8
    np.testing.assert_allclose(loaded.embeddings, embeddings)


def test_shape_mismatch_rejected():
    with pytest.raises(AssertionError):
        PageIndex(_pages(3), np.zeros((2, 8), dtype="float32"))
