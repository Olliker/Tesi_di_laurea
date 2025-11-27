from chromadb.api import ClientAPI

class ChromaRepository:
    def __init__(self, client: ClientAPI):
        self._client = client

    def upsert_documents(self, collection: str, ids: list[str], docs: list[str]) -> None:
        coll = self._client.get_or_create_collection(name=collection)
        coll.upsert(ids=ids, documents=docs)

    def search(self, collection: str, text: str, n_results: int = 5):
        coll = self._client.get_collection(name=collection)
        if coll is None:
            return None
        return coll.query(query_texts=[text], n_results=n_results, include=["documents", "metadatas"])

    ...