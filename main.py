import tables


def store_raw_file_in_hdf5(csv_file_path, hdf5_file_path, hdf5_node_path):
    # Open or create the HDF5 file
    with tables.open_file(hdf5_file_path, mode='a') as h5file:
        # Read the CSV file as binary data
        with open(csv_file_path, 'rb') as f:
            csv_binary_data = f.read()

        # Correctly identify the node path for checking existence and removal
        # Ensure hdf5_node_path does not refer to the root node and is correctly formatted
        if hdf5_node_path.startswith('/'):
            full_path = hdf5_node_path
        else:
            full_path = '/' + hdf5_node_path  # Ensure the path is absolute

        # Check if the node exists
        if h5file.__contains__(full_path):
            # Remove the node if it exists
            h5file.remove_node(full_path, recursive=True)

        # Now, safely proceed to recreate and store data in the node
        atom = tables.UInt8Atom()
        filters = tables.Filters(complevel=5, complib='blosc')
        array_c = h5file.create_carray(h5file.root, hdf5_node_path.strip('/'), atom, shape=(len(csv_binary_data),),
                                       filters=filters)
        array_c[:] = bytearray(csv_binary_data)


# Example usage
csv_file_path = 'noah_data.csv'
hdf5_file_path = 'your_output_file.h5'
hdf5_node_path = 'raw_csv_storage'  # Ensure this is not root or improperly formatted

store_raw_file_in_hdf5(csv_file_path, hdf5_file_path, hdf5_node_path)
