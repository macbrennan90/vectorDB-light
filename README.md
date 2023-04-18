# Vector DB

A Python module to create and query a simple vector database using document embeddings.

## Features

- Store documents with associated metadata
- Embed documents using a customizable embedding model
- Query the database for the most similar documents given a text input
- Load documents in bulk from a JSON file

## Installation

1. Clone this repository:

```
git clone https://github.com/macbrennan90/vectorDB-light.git
```

2. Navigate to the project directory:

```
cd vectorDB-light
```

3. Install the package and its dependencies:

```
pip install -r requirements.txt
```

## Usage

To use the `VectorDB` class in your project, simply import it:

```
from vector_db import VectorDB
```

### Example

```
from vector_db import VectorDB
import your_embedding_model

vector_db = VectorDB(your_embedding_model)

# Add a document to the database
vector_db.add_document(0, "Document 1 text", {"metadata1": "value1"})

# Query the database for the most similar documents
query_text = "query text"
top_k_results = vector_db.query(query_text, top_k=3)

# Load documents from a JSON file
vector_db.load_documents_from_json("data/initial_documents.json")
```

## Tests

To run the tests, first install the `pytest` package:

```
pip install pytest
```

Then, run the tests using the `pytest` command:

```
pytest
```

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas or report bugs.

## License

MIT License
