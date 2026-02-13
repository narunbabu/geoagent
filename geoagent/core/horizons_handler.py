import os
import pickle
import numpy as np
try:
    from zmapio import ZMAPGrid
    HAS_ZMAPIO = True
except ImportError:
    HAS_ZMAPIO = False

default_projectfiles = {
    'horizons': 'horizons.pkl'
}

class HorizonsHandler:
    def __init__(self, project_folder):
        self.project_folder = project_folder
        self.loaded_data = {}
        self.project_files = default_projectfiles

    def load_project(self, project_folder):
        self.project_folder = project_folder
        for data_type, file_name in self.project_files.items():
            file_path = os.path.join(project_folder, file_name)
            
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    self.loaded_data[data_type] = pickle.load(f)
            # print(file_path,self.loaded_data[data_type])
    def save_project(self, project_folder):
        for data_type, file_name in self.project_files.items():
            data = self.loaded_data.get(data_type, {})
            if data:  # Check if data is non-empty
                pickle_path = os.path.join(project_folder, file_name)
                with open(pickle_path, 'wb') as f:
                    pickle.dump(data, f)
        self.project_folder = project_folder

    def get_data(self, data_type):
        return self.loaded_data.get(data_type)

    def get_available_datatypes(self):
        return [data_type for data_type in self.project_files if data_type in self.loaded_data]

    def read_horizons(self, file_paths):
        hors_db_path = os.path.join(self.project_folder, self.project_files['horizons'])
        
        # Load existing horizons if any
        if 'horizons' in self.loaded_data:
            horizons = self.loaded_data['horizons']
        else:
            horizons = {}

        for file_path in file_paths:
            if file_path.endswith('.zmap'):
                horizon_name = os.path.basename(file_path).replace('.zmap', '')
                # horizon_name = base_horizon_name
                # horizon_name = filename.replace('.zmap', '')
                # print(f'horizon_name {horizon_name}')
                # file_path = os.path.join(folder, filename)
                if not HAS_ZMAPIO:
                    raise ImportError("zmapio is required for ZMAP import. Install with: pip install zmapio")
                zmap = ZMAPGrid(file_path)
                horizons[horizon_name] = {
                    'X': zmap.x_values,
                    'Y': zmap.y_values,
                    'Z': zmap.z_values.T
                }

        # Update the loaded_data dictionary
        self.loaded_data['horizons'] = horizons

        # Save the updated horizons to the pickle file
        # self.save_pickle(horizons, hors_db_path)

        print(f'Horizons after import: {horizons.keys()}')
        return horizons

    def find_nearest_indices(self, x_values, y_values, well_x, well_y):
        """Find nearest indices handling both 1D and 2D coordinate arrays"""
        try:
            # Handle different array formats
            if x_values.ndim == 1:
                # 1D array - use directly
                x_axis = x_values
            elif x_values.ndim == 2:
                # 2D meshgrid - extract 1D axis from first row
                x_axis = x_values[0, :]
            else:
                raise ValueError(f"Unsupported X array dimensions: {x_values.ndim}")
            
            if y_values.ndim == 1:
                # 1D array - use directly
                y_axis = y_values
            elif y_values.ndim == 2:
                # 2D meshgrid - extract 1D axis from first column
                y_axis = y_values[:, 0]
            else:
                raise ValueError(f"Unsupported Y array dimensions: {y_values.ndim}")
            
            # Find nearest indices
            x_index = np.argmin(np.abs(x_axis - well_x))
            y_index = np.argmin(np.abs(y_axis - well_y))
            
            return y_index, x_index
            
        except Exception as e:
            print(f"HorizonsHandler: Error in find_nearest_indices: {e}")
            print(f"X shape: {x_values.shape}, Y shape: {y_values.shape}")
            raise

    def get_nearest_value_at_well(self, horizon_name, well_x, well_y, use_nearest=False, num_points=6):
        horizons = self.loaded_data['horizons']
        horizon_data = horizons[horizon_name]
        
        if use_nearest:
            # Get the nearest single point value
            y_index, x_index = self.find_nearest_indices(horizon_data['X'], horizon_data['Y'], well_x, well_y)
            nearest_value = horizon_data['Z'][y_index, x_index]
            return nearest_value
        else:
            # Use the default method with multiple points (weighted average)
            return self.get_horizon_value_at_well(horizon_name, well_x, well_y, num_points)

    def extract_z_values(self, x, y, num_points=6, use_nearest=False):
        horizons = self.loaded_data['horizons']
        z_values = {}
        for horizon_name, horizon_data in horizons.items():
            z_values[horizon_name] = []
            for xi, yi in zip(x, y):
                z_values[horizon_name].append(
                    self.get_nearest_value_at_well(horizon_name, xi, yi, use_nearest, num_points)
                )
        return z_values    

    # def extract_z_values_givenHorizons(self, xi, yi,horizon_names, num_points=6, use_nearest=False):
    #     horizons = self.loaded_data['horizons']
    #     z_values = {}
    #     for horizon_name, horizon_data in horizons.items():
    #         z_values[horizon_name] = []

    #         z_values[horizon_name].append(
    #             self.get_nearest_value_at_well(horizon_name, xi, yi, use_nearest, num_points)
    #         )
    #     return z_values 


    def get_horizon_time_at_well(self, horizon_name, well_x, well_y, num_points=6):
        """Get horizon time at well using inverse distance weighting"""
        try:
            horizons = self.loaded_data['horizons']
            horizon_data = horizons[horizon_name]
            center_y, center_x = self.find_nearest_indices(horizon_data['X'], horizon_data['Y'], well_x, well_y)
            
            # Get surrounding points
            y_indices = np.clip(np.arange(center_y - num_points//2, center_y + num_points//2 + 1), 0, horizon_data['Z'].shape[0] - 1)
            x_indices = np.clip(np.arange(center_x - num_points//2, center_x + num_points//2 + 1), 0, horizon_data['Z'].shape[1] - 1)
            
            # Create meshgrid of indices
            y_mesh, x_mesh = np.meshgrid(y_indices, x_indices)
            
            # Handle coordinate array access - ensure we can get X,Y values at mesh indices
            X_values = horizon_data['X']
            Y_values = horizon_data['Y']
            Z_values = horizon_data['Z']
            
            if X_values.ndim == 1 and Y_values.ndim == 1:
                # 1D coordinate arrays - create meshgrid for distance calculation
                X_mesh, Y_mesh = np.meshgrid(X_values[x_indices], Y_values[y_indices])
                # For 1D coordinates, use meshgrid transpose to match Z indexing
                X_mesh = X_mesh.T
                Y_mesh = Y_mesh.T
            elif X_values.ndim == 2 and Y_values.ndim == 2:
                # 2D coordinate arrays - use direct indexing
                X_mesh = X_values[y_mesh, x_mesh]
                Y_mesh = Y_values[y_mesh, x_mesh]
            else:
                raise ValueError(f"Inconsistent coordinate array dimensions: X={X_values.ndim}D, Y={Y_values.ndim}D")
            
            # Calculate distances
            distances = np.sqrt((X_mesh - well_x)**2 + (Y_mesh - well_y)**2)
            
            # Avoid division by zero
            distances = np.where(distances == 0, 1e-12, distances)
            
            # Compute weights based on inverse distance
            weights = 1 / distances
            weights /= np.sum(weights)  # Normalize weights
            
            # Compute weighted average of Z values
            interpolated_value = np.sum(weights * Z_values[y_mesh, x_mesh])
            
            return interpolated_value
            
        except Exception as e:
            print(f"HorizonsHandler: Error in get_horizon_time_at_well for {horizon_name}: {e}")
            print(f"X shape: {horizon_data['X'].shape}, Y shape: {horizon_data['Y'].shape}, Z shape: {horizon_data['Z'].shape}")
            raise
    def get_all_horizon_name_time_at_well(self, well_x, well_y):
        horizon_data = self.loaded_data['horizons']
        hors = []
        for hname in horizon_data:
            hvalue=self.get_horizon_time_at_well( hname, well_x, well_y, num_points=6)
            hors.append((hname,hvalue))
        return hors

    def save_new_horizon(self, horizon_name, x_values, y_values, z_values):
        """
        Save a new horizon to the project's horizons data
        
        Parameters:
        - horizon_name (str): Name of the new horizon
        - x_values (np.ndarray): X coordinate values
        - y_values (np.ndarray): Y coordinate values  
        - z_values (np.ndarray): Z/time values
        """
        try:
            # Initialize horizons data if not exists
            if 'horizons' not in self.loaded_data:
                self.loaded_data['horizons'] = {}
            
            # Add new horizon data
            self.loaded_data['horizons'][horizon_name] = {
                'X': x_values,
                'Y': y_values,
                'Z': z_values
            }
            
            # Save to project file
            self.save_project(self.project_folder)
            
            print(f"Saved new horizon '{horizon_name}' to project")
            return True
            
        except Exception as e:
            print(f"Error saving horizon '{horizon_name}': {str(e)}")
            return False

    def save_pickle(self, data, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(data, f)

    def load_pickle(self, file_name):
        with open(file_name, 'rb') as f:
            data = pickle.load(f)
        return data
