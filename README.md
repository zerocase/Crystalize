# Crystalize

  

Crystalize is an open-source Embedding & Clustering 3D visualization tool. It provides a GUI that makes it easy to parse and clean up jsonlines formatted logs for visual exploration, allowing users to see the effectiveness of embeddings and clustering.

## Features


- Import and process jsonlines formatted logs

- Support for both single file and folder import

- Selective field choosing for embedding and clustering

- Pre-processing of data

- Customizable embedding models (default: AllMiniLM-v2)

- Automatic embedding generation

- Interactive 3D visualization

- Rotation and zoom functionality

- Cluster highlighting

- Tree widget for cluster and log exploration

- Hierarchical view of clusters and their logs

- Ability to toggle cluster visibility
  

## Installation


1. Clone the repository:

```bash

git clone https://github.com/your-username/crystalize.git

```

```bash

cd crystalize

```


Create a virtual environment:

```bash

python -m venv venv
```

On Windows use `venv\Scripts\activate`
```bash

source venv/bin/activate

```

Install requirements:

```bash

pip install -r requirements.txt

```


Run the main.py file:

```bash

python main.py

```

## Usage

1. Click "Open File" or "Open Folder" to import your jsonlines formatted log file(s).

2. Select relevant fields for embedding and clustering in the "Common Fields" section.

3. Click "Pre-process Data" to prepare the data.

4. Choose an embedding model from the dropdown (default or custom from Hugging Face).

5. Click "Generate Embeddings" to create embeddings and an initial visualization.

6. Click "Perform Clustering" to group logs and enhance the visualization.

7. Explore the data:

- Use the mouse to rotate and zoom in the 3D visualization.

- Select clusters or individual logs in the tree widget to highlight them in the 3D view.

- Toggle cluster visibility using checkboxes in the tree widget.

- Expand cluster nodes to see individual logs and their details.

## Notes

- Only jsonlines format is currently supported for log files.

- For importing new data, it's recommended to clear the database first using the "Clear Database" button.

- There is currently no project system, but backing up and switching databases is possible.

- CUDA-compatible GPUs significantly speed up the embedding process.

## Requirements

Crystalize requires Python 3.12.6 and the following main dependencies:

- PyQt6

- NumPy

- scikit-learn

- sentence-transformers

- PyOpenGL

- torch

For a full list of dependencies, see `requirements.txt`.

## License

This project is open-source and available under the [GNU General Public License v2.0](LICENSE).
