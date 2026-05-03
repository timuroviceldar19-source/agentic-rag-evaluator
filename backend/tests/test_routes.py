from fastapi.testclient import TestClient

from app.main import app


def test_root_redirects_to_docs() -> None:
    client = TestClient(app)
    response = client.get("/", follow_redirects=False)

    assert response.status_code in {307, 308}
    assert response.headers["location"] == "/docs"
