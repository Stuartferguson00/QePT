import tinydb
import numpy as np
from tinydb.middlewares import CachingMiddleware
from tinydb.storages import JSONStorage


class SimulationDB:
    """
    A handler for the TinyDB database that stores simulation results.
    """
    def __init__(self, db_path='simulation_results.json'):
        """
        Initializes the database connection.

        Args:
            db_path (str): The path to the database file.
        """
        # Define a default function for JSON serialization to handle NumPy types
        def _numpy_default(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.integer, np.floating)):
                return o.item()
            raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

        # Using CachingMiddleware for performance and custom serializer for numpy types
        storage = CachingMiddleware(JSONStorage)
        self._db = tinydb.TinyDB(db_path, storage=storage, default=_numpy_default)

    def insert(self, parameters: dict, results: dict):
        """
        Inserts or updates a simulation result in the database based on its parameters.

        Args:
            parameters (dict): A dictionary of the simulation hyperparameters.
            results (dict): A dictionary of the simulation results.
        """
        # Combine parameters and results into a single document
        document = {**parameters, **results}
        
        # Use the parameters to check for an existing document
        # The Query().fragment(parameters) checks if a document contains
        # all the key-value pairs from the `parameters` dictionary.
        self._db.upsert(document, tinydb.Query().fragment(parameters))

    def search(self, query):
        """
        Searches the database for records matching the query.
        Converts numpy-related list data back to numpy arrays upon retrieval.

        Args:
            query: A TinyDB Query object.

        Returns:
            A list of matching records, with relevant fields converted back
            to numpy arrays.
        """
        results = self._db.search(query)
        for res in results:
            if 'data_save' in res and isinstance(res['data_save'], list):
                res['data_save'] = np.array(res['data_save'])
        return results

    def all(self):
        """
        Retrieves all records from the database.

        Returns:
            A list of all records.
        """
        return self.search(tinydb.Query().noop()) # a query that matches everything

    def close(self):
        """
        Closes the database connection.
        """
        self._db.close()

    def clear(self):
        """
        Clears all data from the database.
        """
        self._db.truncate()

    def delete_by_params(self, params: dict):
        """
        Deletes records from the database that match all specified parameters.

        Args:
            params (dict): A dictionary of key-value pairs. All documents
                           containing these exact key-value pairs will be deleted.
        """
        SimQuery = tinydb.Query()
        self._db.remove(SimQuery.fragment(params))

if __name__ == '__main__':
    # Example usage and test
    db = SimulationDB('test_db.json')
    db.clear()

    # Example data
    params1 = {'n_spins': 4, 'm_replicas': 2, 'm_quantum_replicas': 0}
    results1 = {
        'mean_optimal_efforts': 123.4,
        'data_save': np.random.rand(10, 4)
    }

    params2 = {'n_spins': 8, 'm_replicas': 4, 'm_quantum_replicas': 1}
    results2 = {
        'mean_optimal_efforts': 567.8,
        'data_save': np.random.rand(20, 4)
    }
    
    params3 = {'n_spins': 4, 'm_replicas': 4, 'm_quantum_replicas': 1}
    results3 = {
        'mean_optimal_efforts': 910.1,
        'data_save': np.random.rand(5, 4)
    }

    # Insert data
    db.insert(params1, results1)
    db.insert(params2, results2)
    db.insert(params3, results3)

    print("All records:")
    all_recs = db.all()
    print(f"Found {len(all_recs)} records.")
    assert isinstance(all_recs[0]['data_save'], np.ndarray)
    print("Numpy conversion on load confirmed.")

    # Search for data
    print("Searching for n_spins = 4:")
    q = tinydb.Query()
    search_results = db.search(q.n_spins == 4)
    print(f"Found {len(search_results)} records.")
    for r in search_results:
        print(r)
    
    # Verify numpy array type after search
    assert len(search_results) == 2
    assert isinstance(search_results[0]['data_save'], np.ndarray)

    print("Database handler test passed.")
    db.close()
    
    # Clean up the test db file
    import os
    os.remove('test_db.json')
