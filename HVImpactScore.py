# Copyright (c) 2024 Stephen Marinsek
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.""

# HVImpactScore Python Library
# Author: Stephen Marinsek, 2024

# Import dependencies
import copy
import cv2
import math
from multiprocessing import Pool
import numpy as np
import random
from scipy.optimize import minimize

# Class to store a list of SPH data nodes. Typically the full set of imported SPH data
# from a single simulation
class SPHData:
    """
    Class that loads, contains, and manipulates SPH data from a solver.
    """
    
    # Class constructor
    def __init__(self):

        self.sph_nodes = []
        self.node_dict = {}

    # Private function to import SPH data from a single axis text file
    def _read_node_data(self,
                        filepath,
                        header_lines_trim):

        try:
            file = open(filepath, "r")
            raw_data = file.read()
            data = raw_data.splitlines()[header_lines_trim:]
            file.close()
            return data
        
        except:
            print("Error in SPHData._read_node_data: Unable to read " + filepath)

    # Function to import SPH data from text files
    def load_data(self,
                  x_fpath,
                  y_fpath,
                  z_fpath,
                  header_lines_trim,
                  node_diameter,
                  node_density):
        """
        This function loads a set of SPH node data from text files. The data should be in text format with each
        row of data being [node number] [variable value].
        Inputs:
            x_fpath: String, Filepath to the node x position data
            y_fpath: String, Filepath to the node y position data
            z_fpath: String, Filepath to the node z position data
            header_lines_trim: Integer, Number of lines of header information in data files to ignore
            node_diameter: Float, SPH particle diameter
            node_density: Float, Nominal material density
        Outputs:
            None
        """
         
        # Read in data from the files provided.
        xpos_data = self._read_node_data(x_fpath, header_lines_trim)
        ypos_data = self._read_node_data(y_fpath, header_lines_trim)
        zpos_data = self._read_node_data(z_fpath, header_lines_trim)

        # Iterate through list and create new node objects for each item and append to list
        for i in range(0, len(xpos_data)):

            xpos_line = xpos_data[i].split()
            ypos_line = ypos_data[i].split()
            zpos_line = zpos_data[i].split()

            node_number = int(xpos_line[0])
            node_x_loc = float(xpos_line[1])
            node_y_loc = float(ypos_line[1])
            node_z_loc = float(zpos_line[1])

            new_sph_node = SPHNode(node_number,
                               node_x_loc,
                               node_y_loc,
                               node_z_loc,
                               node_diameter,
                               node_density)
            
            self.sph_nodes.append(new_sph_node)

    # Transform the imported node data via rotation about the z-axis
    def transform_data_zrot(self, z_rot_rads):
        """
        This function rotates the data about the z axis by a specified amount.
        Inputs:
            z_rot_rads: Float, Rotation magnitude about the z axis, in Radians
        Outputs:
            None
        """
            
        for i in range(0, len(self.sph_nodes)):
             
            node = self.sph_nodes[i]

            # Transform position w/ rotation about z axis
            rot_x_pos = node.pos[0]*math.cos(z_rot_rads) - node.pos[1]*math.sin(z_rot_rads)
            rot_y_pos = node.pos[0]*math.sin(z_rot_rads) + node.pos[1]*math.cos(z_rot_rads)
            rot_z_pos = node.pos[2]

            node.pos = np.array([rot_x_pos, rot_y_pos, rot_z_pos])

    # Function to radially duplicate and distribute nodes about the z-axis
    def dist_data_zrot(self,
                       rot_qty,
                       angle_min_rads,
                       angle_max_rads):
        """
        This function duplicates and distributes sph data randomly about the z axis. This is
        useful to reduce noise when debris cloud is axisymmetric.
        Input:
            rot_qty: Integer, Number of additional data sets to randomly rotate and add.
            angle_min_rads: Float, Lower bound of random rotation range in Radians
            angle_max_rads: Float, Upper bound of random rotation range in Radians
        Output:
            None
        """

        rotation_list = []
        for i in range(0, rot_qty):
            rotation_list.append(random.uniform(angle_min_rads, angle_max_rads))
        
        SPHData.transform_data_zrot(self, rotation_list[0])
        SPHData._add_data_zrot(self, rotation_list[1:])
        
    # Private function used for SPH node radial distribution
    def _add_data_zrot(self,
                       z_rotation_list):

        new_node_list = []
        for rotations in z_rotation_list:
            for i in range(0, len(self.sph_nodes)):
                
                node = self.sph_nodes[i]
                new_sph_node = SPHNode(node.node_number,
                                       node.pos[0],
                                       node.pos[1],
                                       node.pos[2],
                                       node.diameter,
                                       node.density)
                
                # Perform rotation of new node
                rot_x_pos = node.pos[0]*math.cos(rotations) - node.pos[1]*math.sin(rotations)
                rot_y_pos = node.pos[0]*math.sin(rotations) + node.pos[1]*math.cos(rotations)
                rot_z_pos = node.pos[2]

                new_sph_node.pos = np.array([rot_x_pos, rot_y_pos, rot_z_pos])
                new_node_list.append(new_sph_node)

        self.sph_nodes = np.append(self.sph_nodes, new_node_list)

    # Function that rotates the data about the x axis
    def transform_data_xrot(self, x_rot_rads):
        """
        This function rotates the data about the x axis by a specified amount.
        Inputs:
            x_rot_rads: Float, Rotation magnitude about the x axis, in Radians
        Outputs:
            None
        """

        for node in self.sph_nodes:

            # Transform position w/ rotation about x axis
            rot_x_pos = node.pos[0]
            rot_y_pos = node.pos[1]*math.cos(x_rot_rads) - node.pos[2]*math.sin(x_rot_rads)
            rot_z_pos = node.pos[1]*math.sin(x_rot_rads) + node.pos[2]*math.cos(x_rot_rads)
            
            node.pos = np.array([rot_x_pos, rot_y_pos, rot_z_pos])
    
    # Function that flips the data about the z plane (or x & y axes)
    def flip_data_zplane(self):
        """
        This function flips the data about the z plane.
        Inputs:
            None
        Outputs:
            None
        """
          
        # Iterate through the SPH data and flip about the z axis
        for node in self.sph_nodes:
              
            # Flip position about z plane
            node.pos = np.array([node.pos[0], node.pos[1], -1.0 * node.pos[2]])
           
    # Function that expands axisymmetric simulation data about the x plane
    def expand_sym_about_x_plane(self):
        """
        This function flips the data about the x plane.
        Inputs:
            None
        Outputs:
            None
        """
        
        # Create duplicate node list
        new_node_list = copy.deepcopy(self.sph_nodes)

        # Perform x plane expansion by flipping over the x plane for new nodes
        for node in new_node_list:
            node.pos[0] = -1.0 * node.pos[0]
            
        self.sph_nodes = np.append(self.sph_nodes, new_node_list)
            
    # Function that expands axisymmetric simulation data about the y plane
    def expand_sym_about_y_plane(self):
        """
        This function expands the data by mirroring about the y plane. This is used when
        simulation leverages planes of symmetry.
        Inputs:
            None
        Outputs:
            None
        """
    
        # Create duplicate node list
        new_node_list = copy.deepcopy(self.sph_nodes)

        # Perform x plane expansion by flipping over the x plane for new nodes
        for node in new_node_list:
            node.pos[1] = -1.0 * node.pos[1]
            
        self.sph_nodes = np.append(self.sph_nodes, new_node_list)

# Class to store the information from a single SPH node    
class SPHNode:

    # Class constructor
    def __init__(self,
                 node_number,
                 x,
                 y,
                 z,
                 diameter,
                 density):
        
        # Initialize state
        self.node_number = node_number
        self.pos = np.array([x, y, z])
        self.diameter = diameter
        self.density = density
        self.mass = self.density*(4/3)*math.pi*math.pow(self.diameter / 2.0, 3)
        self.group_hash = None

    # Function to assign the hash key for node during image generation
    def set_group_hash(self, hash_value):

        self.group_hash = hash_value

class ImageProcessing:
    """
    This class loads experimental data images and preforms image processing tasks.
    """

    @staticmethod
    def load_exp_image(filepath):
        """
        This function loads an image of a debris cloud, typically obtained via experiment.
        Inputs:
            filepath: String, Filepath to the image
        Outputs:
            image: 2D Numpy Array: Loaded image
        """

        image = cv2.imread(filepath)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_array = np.array(image_gray)
        return image_array.astype(np.float64)

    @staticmethod
    def normalize_image(image):
        """
        This function normalizes an image so the pixel intensity range is [0.0, 255.0]
        Input:
            image: 2D Numpy Array: Image to normalize
        Output:
            image: 2D Numpy Array: Normalized image
        """

        normalized_image = np.zeros(np.shape(image))

        darkest_val = image.min()
        lightest_val = image.max()

        for row in range(0, np.shape(image)[0]):
            for col in range(0, np.shape(image)[1]):
                normalized_image[row, col] = 255.0 * (image[row, col] - darkest_val) / (lightest_val - darkest_val)

        return normalized_image

    @staticmethod
    def threshold_image(image, threshold_value):
        """
        This function removes background noise by setting everything below the threshold value to white.
        Background removal should generally be followed by normalization.
        Input:
            image: 2D Numpy Array: Image to remove background
        Output:
            image: 2D Numpy Array: Image with background removed
        """

        thresh_image = np.zeros(np.shape(image))

        for row in range(0, np.shape(image)[0]):
            for col in range(0, np.shape(image)[1]):

                if image[row, col] > threshold_value:
                    thresh_image[row, col] = threshold_value
                else:
                    thresh_image[row, col] = image[row, col]

                thresh_image[row, col] = 255.0 * thresh_image[row, col] / threshold_value

        return thresh_image

    @staticmethod
    def threshold_image_idw(image, 
                            threshold_margin, 
                            sample_area_array):
        """
        This function removes the background of an image by subtracting intensity via inverse distance
        interpolation between background sample areas with an additional margin to account for background
        noise. Background removal should generally be followed by normalization.
        Inputs:
            image: 2D Numpy Array, Image to remove background
            threshold_margin: Float, Additional background removal margin to account for noise
            sample_area_array: Float [[[Y Upper Left, X Upper Left], [Y Lower Right, X Lower Right]], ...Additional areas...]
        Outputs:
            image: 2D Numpy Array, Image with background removed
        """
           
        def _inverse_distance_weighting(distances, values, power):
            
            for i in range(0, len(distances)):
                if distances[i] <= 0.0:
                    distances[i] = 0.00001

            numerator = np.sum(values / (distances ** power))
            weights = np.sum(1 / (distances ** power))
            interpolated_value = numerator / weights
            
            return interpolated_value
        
        # Create new, blank image
        thresh_image = np.zeros(np.shape(image))

        # Average each sample area to get background intensity
        bg_int_list = []
        bg_loc_list = []
        
        # Create location and intensity list for inverse distance weighted interpolation
        for i in range(0, len(sample_area_array)):
            sample_area = sample_area_array[i]
            loc_y_center = (sample_area[0][0] + sample_area[1][0]) / 2.0
            loc_x_center = (sample_area[0][1] + sample_area[1][1]) / 2.0
            bg_loc_list.append([loc_y_center, loc_x_center])
            
            area_intensity = np.average(image[sample_area[0][0]:sample_area[1][0],
                                              sample_area[0][1]:sample_area[1][1]])
            bg_int_list.append(area_intensity)

        # Subtract interpolated intensity from each pixel
        img_height = np.shape(image)[0]
        img_width = np.shape(image)[1]
        
        for row in range(0, img_height):
            for col in range(0, img_width):
                
                distances = []
                for point in range(0, len(bg_loc_list)):
                    distances.append(math.sqrt(math.pow((bg_loc_list[point][0] - row), 2) + math.pow((bg_loc_list[point][1] - col), 2)))

                sub_mag = (255.0 - _inverse_distance_weighting(np.array(distances), np.array(bg_int_list), 1)) + threshold_margin

                if image[row, col] + sub_mag > 255.0:
                    thresh_image[row, col] = 255.0
                else:
                    thresh_image[row, col] = image[row, col] + sub_mag

        return ImageProcessing.normalize_image(thresh_image)

    @staticmethod
    def apply_unit_light_absorbtion(image, abs_coef):
        """
        This function turns a density map into an image by applying photon absorption. This captures the effects
        of XRF or laser shadowgraph photography.
        Inputs:
            image: 2D Numpy Array: Density map
            abs_coef: Float, Absorption coefficient
        Outputs:
            image: 2D Numpy Array: Image reflecting photon absorption through density map. This
                image can be compared to experimental images of debris clouds.
        """

        laser_image = np.zeros(np.shape(image))

        for row in range(0, np.shape(image)[0]):
            for col in range(0, np.shape(image)[1]):
                laser_image[row, col] = math.exp(-1.0 * abs_coef * image[row, col])

        return laser_image
    
    @staticmethod 
    def find_abs_coefficient(density_image,
                             exp_image,
                             range_max,
                             range_min):
        """
        This function performs an optimization to determine the best absorption coefficient based on mass conservation.
        Inputs:
            density_image: 2D Numpy Array: Density map
            exp_image: 2D Numpy Array: Experimental debris cloud image
            range_max: Float, Upper bound for optimization
            range_min: Float, Lower bound for optimization. Should be > 0.0
        Outputs:
            absorption"""
        
        if range_min == 0.0:
            range_min = 1.0e-10
        
        def opt_wrapper(exposure, density_img, exp_img, scoring_method):
            image = ImageProcessing.apply_unit_light_absorbtion(density_img, exposure)
            image_norm = ImageProcessing.normalize_image(image)
            return -scoring_method(image_norm, exp_img)
        
        def image_mass_diff(analysis_image, experimental_image):
        
            img_height = np.shape(analysis_image)[0]
            img_width = np.shape(analysis_image)[1]

            total_error = 0
            for row in range(0, img_height):
                for col in range(0, img_width):
                    total_error = total_error + (analysis_image[row, col] - experimental_image[row, col])

            return -1.0*abs(total_error)
        
        init_val = (range_max + range_min) / 2.0
        opt_abs = minimize(opt_wrapper, 
                           init_val,
                           args=(density_image, exp_image, image_mass_diff),
                           bounds=[(range_min, range_max)],
                           tol=1e-7).x
        
        return opt_abs
    
class SPHDataProcessing:
    """
    This class performs data processing on SPH data sets.
    """

    @staticmethod
    def bspline_kernel(pix_1, pix_2, node_1, node_2, h_dist):
        """
        This function is the bspline kernel function popular in many of the SPH implementations
        Inputs:
            pix_1: Float, Pixel location in dimension 1.
            pix_2: Float, Pixel location in dimension 2.
            node_1: Float, Node location in dimension 1.
            node_2: Float, Node location in dimension 2.
            h_dist: Float, Kernel radius.
        Outputs:
            kernel value: Float, Value of kernel evaluated for given inputs.
        """

        euc_dist = math.sqrt((pix_1 - node_1)**2 + (pix_2 - node_2)**2)
        veta = euc_dist / h_dist

        if (veta <= 1.0):
            return (1 / (math.pi * h_dist**3)) * (1 - (3 / 2) * veta**2 + (3 / 4) * veta**3)
        elif (veta > 1.0 and veta <= 2.0):
            return (1 / (math.pi * h_dist**3)) * (1 / 4) * (2 - veta)**3
        else:
            return 0

    @staticmethod
    def sph_data_to_dens_map_single(sph_data,
                                     kernel,
                                     kernel_radius,
                                     px_size,
                                     img_min_x,
                                     img_max_x,
                                     img_min_y,
                                     img_max_y,
                                     verbose=False):
        """
        This function produces a 2D density map from a set of SPH data.
        Inputs:
            sph_data: SPHData, Data imported from solver with all necessary expansions and rotations performed.
            kernel: Function, The kernel to use for smoothing and mass distributing.
            kernel_radius: Float, The smoothing kernel radius. Recommended to be greater than pixel size.
            px_size: Float, Pixel size of generated density map. Expressing in physical length units to match experimental
                image.
            img_min_x: Integer, Pixels from origin to left side of image.
            img_max_x: Integer, Pixels from origin to right side of image.
            img_min_y: Integer, Pixels from origin to top of image.
            img_max_y: Integer, Pixels from origin to bottom of image.
            verbose: Boolean, Expanded status/text output setting.
        Outputs:
            density map: 2D Numpy Array: density map produced from SPH data.
        """
        
        # Partition the nodes into pixels for faster rendering
        SPHDataProcessing._partition_nodes(sph_data, px_size)      
        
        # Determine the number of adjacent pixels to look in for rastering
        adj_px = math.ceil(2.0 * kernel_radius / px_size)
        
        # Initialize the mass_map
        dens_map = np.zeros((img_max_y - img_min_y, img_max_x - img_min_x))

        # Iterate over all the pixels
        for row in range(img_min_y, img_max_y):
            if verbose is True:
                print("Processing row " + str(row))
            for col in range(img_min_x, img_max_x):

                # Calculate the hash value of the current pixel
                c_pixel_hash_num = (row, col)

                # Calculate the 2D physical position of the pixel
                c_pixel_coord_1 = row * px_size # row direction
                c_pixel_coord_2 = col * px_size # column direction
                
                # Survey around pixel to determine value using kernel
                for s_row in range(c_pixel_hash_num[0] - adj_px, c_pixel_hash_num[0] + adj_px + 1):
                    for s_col in range(c_pixel_hash_num[1] - adj_px, c_pixel_hash_num[1] + adj_px + 1):
                        
                        s_hash = str((s_row, s_col))
                        
                        if s_hash in sph_data.node_dict:
                            for node in sph_data.node_dict[s_hash]:
                                dens_map[row - img_min_y, col - img_min_x] += node.mass*kernel(c_pixel_coord_1,
                                                                                                c_pixel_coord_2,
                                                                                                node.pos[1],
                                                                                                node.pos[2],
                                                                                                kernel_radius)

        return dens_map

    @staticmethod
    def sph_data_to_dens_map_multi_rot(sph_data,
                                       kernel,
                                       kernel_radius,
                                       px_size,
                                       img_min_x,
                                       img_max_x,
                                       img_min_y,
                                       img_max_y,
                                       adl_rotation_qty,
                                       rotation_min,
                                       rotation_max,
                                       x_angle,
                                       verbose=False):
        """
        This function produces a 2D density map from a set of SPH data and implements radial particle distributing for
        axisymmetric debris clouds
        Inputs:
            sph_data: SPHData, Data imported from solver with all necessary expansions and rotations performed.
            kernel: Function, The kernel to use for smoothing and mass distributing.
            kernel_radius: Float, The smoothing kernel radius. Recommended to be greater than pixel size.
            px_size: Float, Pixel size of generated density map. Expressing in physical length units to match experimental
                image.
            img_min_x: Integer, Pixels from origin to left side of image.
            img_max_x: Integer, Pixels from origin to right side of image.
            img_min_y: Integer, Pixels from origin to top of image.
            img_max_y: Integer, Pixels from origin to bottom of image.
            adl_rotation_qty: Integer, Number of additiona random rotations about z axis to add.
            rotation_min: Float, Lower bound for random rotation.
            rotation_max: Float, Upper bound for random rotation.
            x_angle: Float, Set rotation for sph data about x axis (image normal) prior to radial distributing
            verbose: Boolean, Expanded status/text output setting.
        Outputs:
            density map: 2D Numpy Array: density map produced from SPH data.
        """

        # Create rotated versions of analysis data and append as datapacks
        dens_map_datapack = []
        for i in range(0, adl_rotation_qty + 1):
            angle = random.uniform(rotation_min, rotation_max)
            data_pack = {"data": sph_data,
                         "kernel": kernel,
                         "kernel_radius": kernel_radius,
                         "px_size": px_size,
                         "img_min_x": img_min_x,
                         "img_max_x": img_max_x,
                         "img_min_y": img_min_y,
                         "img_max_y": img_max_y,
                         "z_angle": angle,
                         "x_angle": x_angle,
                         "verbose": verbose,
                         "is_copy": True}
            dens_map_datapack.append(data_pack)

        # Process data
        with Pool() as pool:
            result = pool.map(SPHDataProcessing._dens_map_parallel, dens_map_datapack)

        # Sum the density from all rotations and divide by the number of rotations to normalize
        img_height = np.shape(result[0])[0]
        img_width = np.shape(result[0])[1]
        sum_density_map = np.zeros([img_height, img_width])
        for row in range(0, img_height):
            for col in range(0, img_width):
                for sample in range(0, len(result)):
                    sum_density_map[row, col] = sum_density_map[row, col] + result[sample][row, col]

        norm_density_map = sum_density_map / (adl_rotation_qty + 1)

        return norm_density_map
        
    @staticmethod
    def _get_node_hash(px_size, pos_1, pos_2):

        return str(SPHDataProcessing._get_node_hash_numbers(px_size, pos_1, pos_2))
    
    @staticmethod
    def _get_node_hash_numbers(px_size, pos_1, pos_2):

        hash_coord_1 = round(pos_1/px_size)
        hash_coord_2 = round(pos_2/px_size)
        return (hash_coord_1, hash_coord_2)
    
    @staticmethod
    def _partition_nodes(sph_data, px_size):
        
        for node in sph_data.sph_nodes:
            hash_coord_1 = 0
            hash_coord_2 = 0
            hash_key = None

            (hash_coord_1, hash_coord_2) = SPHDataProcessing._get_node_hash_numbers(px_size, 
                                                                                    node.pos[1], 
                                                                                    node.pos[2])
            hash_key = SPHDataProcessing._get_node_hash(px_size, 
                                                        node.pos[1], 
                                                        node.pos[2])

            node.set_group_hash((hash_coord_1, hash_coord_2))

            if hash_key in sph_data.node_dict:
                sph_data.node_dict[hash_key].append(node)
            else:
                sph_data.node_dict[hash_key] = []
                sph_data.node_dict[hash_key].append(node)

    # Define wrapper for parallel processing
    @staticmethod
    def _dens_map_parallel(sph_data_pack):
        sph_data_copy = copy.deepcopy(sph_data_pack["data"])
        sph_data_copy.transform_data_zrot(sph_data_pack["z_angle"])
        sph_data_copy.transform_data_xrot(sph_data_pack["x_angle"])
        density_map = SPHDataProcessing.sph_data_to_dens_map_single(sph_data_copy,
                                                                    sph_data_pack["kernel"],
                                                                    sph_data_pack["kernel_radius"],
                                                                    sph_data_pack["px_size"],
                                                                    sph_data_pack["img_min_x"],
                                                                    sph_data_pack["img_max_x"],
                                                                    sph_data_pack["img_min_y"],
                                                                    sph_data_pack["img_max_y"],
                                                                    sph_data_pack["verbose"])
        return density_map
    
class scoring:
    """
    This class contains the scoring methods used to perform HVI SPH similarity scoring.
    """

    @staticmethod
    def score_images(analysis_image,
                     experimental_image,
                     threshold):
        """
        This function performs similarity scoring of two images using the image normalized MSE approached described in
        the paper by Marinsek and Stiles
        Inputs: 
            analysis_image: 2D Numpy Array: Image generated from the analysis SPH data.
            experimental_image: 2D Numpy Array: Processed experimental image. Can also be different analysis SPH image.
            threshold: Float: Intensity level below which pixels aren't considered in scoring. Used to remove background.
        Outputs:
            score: Float: Scoring value from comparing both images.
        """

        img_height = np.shape(analysis_image)[0]
        img_width = np.shape(analysis_image)[1]

        mse_error = 0
        limit = 255.0 - threshold
        exp_thresh_int_total = 0

        for y in range(0, img_height):
            for x in range(0, img_width):
                if analysis_image[y, x] < limit or experimental_image[y, x] < limit:
                    mse_error = mse_error + math.pow(analysis_image[y, x] - experimental_image[y, x], 2)
                    
                if experimental_image[y, x] < limit:
                    exp_thresh_int_total = exp_thresh_int_total + math.pow(255.0 - experimental_image[y, x], 2)
        
        return 1.0 - mse_error / exp_thresh_int_total