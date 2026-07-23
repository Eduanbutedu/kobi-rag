from rag.embedding import EMBEDDING_DIM, embed_texts


def test_empty_input_returns_empty():
    assert embed_texts([]) == []

def test_vectors_have_expected_dimension():
    vectors = embed_texts(["staj yönergesi"])
    assert len(vectors) == 1
    assert len(vectors[0]) == EMBEDDING_DIM


def test_similar_texts_are_closer_than_unrelated():
    def dot(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b, strict=True))

    v_izin, v_tatil, v_futbol = embed_texts(
        ["Yıllık izin başvurusu nasıl yapılır?",
         "Tatil günlerimi nasıl talep edebilirim?",
         "Futbol maçı kaç dakika sürer?"]
    )
    assert dot(v_izin, v_tatil) > dot(v_izin, v_futbol)