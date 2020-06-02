# lshash/lshash.py
# Copyright 2012 Kay Zhu (a.k.a He Zhu) and contributors (see CONTRIBUTORS.txt)
#
# This module is part of lshash and is released under
# the MIT License: http://www.opensource.org/licenses/mit-license.php

import os
import json
import numpy as np
import time
from lshash.utils import numpy_array_from_list_or_numpy_array, perform_pca
from lshash.storage import storage

try:
    from bitarray import bitarray
except ImportError:
    bitarray = None


from sklearn.neighbors import NearestNeighbors


class LSHash(object):
    """ LSHash implments locality sensitive hashing using random projection for
    input vectors of dimension `input_dim`.

    Attributes:

    :param hash_size:
        The length of the resulting binary hash in integer. E.g., 32 means the
        resulting binary hash will be 32-bit long.
    :param input_dim:
        The dimension of the input vector. E.g., a grey-scale picture of 30x30
        pixels will have an input dimension of 900.
    :param num_hashtables:
        (optional) The number of hash tables used for multiple lookups.
    :param storage_config:
        (optional) A dictionary of the form `{backend_name: config}` where
        `backend_name` is the either `dict` or `redis`, and `config` is the
        configuration used by the backend. For `redis` it should be in the
        format of `{"redis": {"host": hostname, "port": port_num}}`, where
        `hostname` is normally `localhost` and `port` is normally 6379.
    :param matrices_filename:
        (optional) Specify the path to the compressed numpy file ending with
        extension `.npz`, where the uniform random planes are stored, or to be
        stored if the file does not exist yet.
    :param overwrite:
        (optional) Whether to overwrite the matrices file if it already exist
    """

    def __init__(self, hash_size, input_dim, num_hashtables= 1,num_hash_per_tables=1,
                 hash_function = None,hash_type = None,no_of_nearest_neighbour= 20,storage_config=None, matrices_filename=None,
                 overwrite=False):

        self.hash_size = hash_size
        self.input_dim = input_dim
        self.num_hashtables = num_hashtables
        self.num_hash_per_tables = num_hash_per_tables

        self.hash_keys = []
        self.hash_keys_array = []

        self.hash_type = hash_type
        self.hash_function = hash_function

        if storage_config is None:

            storage_config = {'dict': None}
            #storage_config = ['redis']
        self.storage_config = storage_config

        if matrices_filename and not matrices_filename.endswith('.npz'):
            raise ValueError("The specified file name must end with .npz")
        self.matrices_filename = matrices_filename
        self.overwrite = overwrite

        self._init_uniform_planes()
        self._init_hashtables()



    def build_NN(self,keys,n_of_neighbour):

        nbrs = NearestNeighbors(n_neighbors=n_of_neighbour, algorithm='ball_tree').fit(keys)
        self.nbrs = nbrs

        print(" \n \n Nearest Neighbour computed  :: ")




    def arr_to_str(self,m):

       if self.hash_type == 'discrete':

        return "".join([str(i) for i in m])

       elif self.hash_type == 'bin':

        return "".join(['1' if i > 0 else '0' for i in m])


    def str_to_arr(self,k):

        ar = [int(x) for x in list(k)]
        return ar

    def get_keys(self):

        for table in self.hash_tables:

            keys = list(table.keys())

            print(keys)


            for index, key in enumerate(keys):
                ar = [int(x) for x in list(key)]
                keys[index] = ar


        return keys

    def _init_uniform_planes(self):

        """ Initialize uniform planes used to calculate the hashes

        if file `self.matrices_filename` exist and `self.overwrite` is
        selected, save the uniform planes to the specified file.

        if file `self.matrices_filename` exist and `self.overwrite` is not
        selected, load the matrix with `np.load`.

        if file `self.matrices_filename` does not exist and regardless of
        `self.overwrite`, only set `self.uniform_planes`.
        """

        if "uniform_planes" in self.__dict__:
            return

        if self.matrices_filename:

            file_exist = os.path.isfile(self.matrices_filename)

            if file_exist and not self.overwrite:
                try:
                    npzfiles = np.load(self.matrices_filename)
                except IOError:
                    print("Cannot load specified file as a numpy array")
                    raise
                else:
                    npzfiles = sorted(npzfiles.items(), key=lambda x: x[0])
                    self.uniform_planes = [t[1] for t in npzfiles]

            else:

                self.uniform_planes = [self._generate_uniform_planes()
                                       for _ in range(self.num_hash_per_tables)]
                try:
                    np.savez_compressed(self.matrices_filename,
                                        *self.uniform_planes)
                except IOError:
                    print("IOError when saving matrices to specificed path")
                    raise
        else:

            if self.hash_function == "pca":

                print("Collecting PCA planes...")


                self.uniform_planes = [self.get_PCA_planes(np.random.randn(self.num_hash_per_tables*self.hash_size+10, self.input_dim),self.num_hash_per_tables*self.hash_size)]

                self.uniform_planes = np.array(self.uniform_planes)

                self.uniform_planes = np.resize(self.uniform_planes,(self.num_hash_per_tables,self.hash_size,self.input_dim))

            else:

                self.uniform_planes = [self._generate_uniform_planes()
                                       for _ in range(self.num_hash_per_tables)]

    def _init_hashtables(self):
        """ Initialize the hash tables such that each record will be in the
        form of "[storage1, storage2, ...]" """

        self.hash_tables = [storage(self.storage_config, i)
                            for i in range(self.num_hashtables)]

    def _generate_uniform_planes(self):


        """ Generate uniformly distributed hyperplanes and return it as a 2D
        numpy array.
        """

        return np.random.randn(self.hash_size, self.input_dim)

    def get_PCA_planes(self, training_set,projection_count):


        if not training_set is None:
            # Get numpy array representation of input
            training_set = numpy_array_from_list_or_numpy_array(training_set)

            # Get subspace size from training matrix
            self.dim = training_set.shape[0]

            # Get transposed training set matrix for PCA
            training_set_t = np.transpose(training_set)

            # Compute principal components
            (eigenvalues, eigenvectors) = perform_pca(training_set_t)

            # Get largest N eigenvalue/eigenvector indices
            largest_eigenvalue_indices = np.flipud(
                np.argsort(eigenvalues))[:projection_count]

            # Create matrix for first N principal components
            self.components = np.zeros((self.dim,
                                           len(largest_eigenvalue_indices)))

            # Put first N principal components into matrix
            for index in range(len(largest_eigenvalue_indices)):
                self.components[:, index] = \
                    eigenvectors[:, largest_eigenvalue_indices[index]]

            # We need the component vectors to be in the rows
            self.components = np.transpose(self.components)
        else:
            self.dim = None
            self.components = None

        # This is only used in case we need to process sparse vectors
        self.components_csr = None

        return self.components



    def _hash(self, planes, input_point):

        """ Generates the binary hash for `input_point` and returns it.

        :param planes:
            The planes are random uniform planes with a dimension of
            `hash_size` * `input_dim`.
        :param input_point:
            A Python tuple or list object that contains only numbers.
            The dimension needs to be 1 * `input_dim`.
        """

        try:

            input_point = np.array(input_point)  # for faster dot product
            projections = np.dot(planes, input_point)
            norm_projections = (1 / np.max(np.abs(projections))) * projections
            #norm_projections = projections

            if self.hash_type == 'discrete':

                norm_projections = 1000*norm_projections
                norm_projections = [np.int(i) for i in norm_projections]
                #print(" \n ",norm_projections)


        except TypeError as e:
            print("""The input point needs to be an array-like object with
                  numbers only elements""")
            raise

        except ValueError as e:
            print("""The input point needs to be of the same dimension as
                  `input_dim` when initializing this LSHash instance""", e)
            raise

        else:

            if self.hash_type =='discrete':
                return "".join([str(i) for i in norm_projections]), norm_projections

            elif self.hash_type == 'bin':
                return "".join(['1' if i > 0 else '0' for i in projections]), norm_projections



    def _as_np_array(self, json_or_tuple):

        """ Takes either a JSON-serialized data structure or a tuple that has
        the original input points stored, and returns the original input point
        in numpy array format.
        """

        if isinstance(json_or_tuple, str):
            # JSON-serialized in the case of Redis
            try:
                # Return the point stored as list, without the extra data
                tuples = json.loads(json_or_tuple)[0]
            except TypeError:
                print("The value stored is not JSON-serilizable")
                raise
        else:
            # If extra_data exists, `tuples` is the entire
            # (point:tuple, extra_data). Otherwise (i.e., extra_data=None),
            # return the point stored as a tuple
            tuples = json_or_tuple

        if isinstance(tuples[0], tuple):
            # in this case extra data exists
            return np.asarray(tuples[0])

        elif isinstance(tuples, (tuple, list)):
            try:
                return np.asarray(tuples)
            except ValueError as e:
                print("The input needs to be an array-like object", e)
                raise
        else:
            raise TypeError("query data is not supported")

    def index(self, input_point, extra_data=None):


        """ Index a single input point by adding it to the selected storage.

        If `extra_data` is provided, it will become the value of the dictionary
        {input_point: extra_data}, which in turn will become the value of the
        hash table. `extra_data` needs to be JSON serializable if in-memory
        dict is not used as storage.

        :param input_point:
            A list, or tuple, or numpy ndarray object that contains numbers
            only. The dimension needs to be 1 * `input_dim`.
            This object will be converted to Python tuple and stored in the
            selected storage.
        :param extra_data:
            (optional) Needs to be a JSON-serializable object: list, dicts and
            basic types such as strings and integers.
        """

        if isinstance(input_point, np.ndarray):
            input_point = input_point.tolist()

        if extra_data:
            value = (tuple(input_point), extra_data)
        else:
            value = tuple(input_point)


        for i, hash in enumerate(self.uniform_planes):


            for table in self.hash_tables:

                str_hash, arr_hash = self._hash(self.uniform_planes[i], input_point)

                self.hash_keys_array.append(arr_hash)
                #print(self.hash_keys_array)
                table.append_val(str_hash,value)




    def query(self, query_point, num_results=None, distance_func=None):

        #print("query_shape :: ",query_point.shape)

        """ Takes `query_point` which is either a tuple or a list of numbers,
        returns `num_results` of results as a list of tuples that are ranked
        based on the supplied metric function `distance_func`.

        :param query_point:
            A list, or tuple, or numpy ndarray that only contains numbers.
            The dimension needs to be 1 * `input_dim`.
            Used by :meth:`._hash`.
        :param num_results:
            (optional) Integer, specifies the max amount of results to be
            returned. If not specified all candidates will be returned as a
            list in ranked order.
        :param distance_func:
            (optional) The distance function to be used. Currently it needs to
            be one of ("hamming", "euclidean", "true_euclidean",
            "centred_euclidean", "cosine", "l1norm"). By default "euclidean"
            will used.
        """

        candidates = set()

        if not distance_func:
            distance_func = "euclidean"

        if distance_func == "hamming":
            if not bitarray:
                raise ImportError(" Bitarray is required for hamming distance")

            for i, table in enumerate(self.hash_tables):

                binary_hash, _ = self._hash(self.uniform_planes[i], query_point)

                for key in table.keys():

                    distance = LSHash.hamming_dist(key, binary_hash)
                    if distance < 2:
                        candidates.update(table.get_list(key))

            d_func = LSHash.euclidean_dist_square

        else:

            if distance_func == "euclidean":
                d_func = LSHash.euclidean_dist_square
            elif distance_func == "true_euclidean":
                d_func = LSHash.euclidean_dist
            elif distance_func == "centred_euclidean":
                d_func = LSHash.euclidean_dist_centred
            elif distance_func == "cosine":
                d_func = LSHash.cosine_dist
            elif distance_func == "l1norm":
                d_func = LSHash.l1norm_dist

            elif distance_func == "np_bin_dist":
                d_func = LSHash.hamming_np_bin_dist

            elif distance_func == "normalised_euclidean":
                d_func = LSHash.normalised_euclidean

            elif distance_func == "normalised_block_euclidean":
                d_func = LSHash.normalise_block_euclidean



            else:
                raise ValueError("The distance function name is invalid.")


            binary_hash_list = []

            for i ,hash in enumerate(self.uniform_planes):

                for table in self.hash_tables:


                    #ss = time.time()
                    #print("\n  uniform_plane_shape ",np.array(self.uniform_planes[i]).shape)
                    binary_hash, arr_hash = self._hash(self.uniform_planes[i], query_point)
                    binary_hash_list.append(arr_hash)


                    # array_hash = self.str_to_arr(binary_hash)
                    # print (" array_hash  ::",array_hash)


                    # print("\n")
                    # str_hash = self.arr_to_str(array_hash)
                    # print(" string_hash ::", str_hash)

                    candidates.update(table.get_list(binary_hash))

            s2 = time.time()

            if len(candidates) == 0 or len(candidates)<=50:

                print("\n\n  performing NN Search")

                for h in binary_hash_list:

                    distances, indices = self.nbrs.kneighbors(np.array(self.str_to_arr(h)).reshape(1,-1), 50)

                    s1 = time.time()
                    for table in self.hash_tables:

                        for inddice in indices[0]:

                            str_bin_hash = self.arr_to_str(self.hash_keys_array[inddice])
                            #str_bin_hash = "".join(['1' if i > 0 else '0' for i in array_bin_hash])
                            candidates.update(table.get_list(str_bin_hash))
                    e1 = time.time()

                    #print("\n candidate loop update ::", e1 - s1)



        # rank candidates by distance function


        if distance_func == "normalised_euclidean" :

            var_x= np.var(query_point)

            candidates = [(ix[1][0], d_func(query_point,self._as_np_array(ix[0]), var_x,ix[1][1]))
                          for ix in candidates]

        else:
            candidates = [(ix[1][0], d_func(query_point, self._as_np_array(ix[0])))
                          for ix in candidates]


        candidates.sort(key=lambda x: x[1])

        e2 = time.time()


        print("\n candidate loop calc time ::", e2-s2,",  length of candidate", len(candidates))

        return candidates[:num_results] if num_results else candidates

    ### distance functions

    @staticmethod
    def hamming_dist(bitarray1, bitarray2):
        xor_result = bitarray(bitarray1) ^ bitarray(bitarray2)
        return xor_result.count()

    @staticmethod
    def euclidean_dist(x, y):
        """ This is a hot function, hence some optimizations are made. """
        diff = np.array(x) - y
        return np.sqrt(np.dot(diff, diff))

    @staticmethod
    def euclidean_dist_square(x, y):
        """ This is a hot function, hence some optimizations are made. """
        diff = np.array(x) - y
        return np.dot(diff, diff)

    @staticmethod
    def euclidean_dist_centred(x, y):
        """ This is a hot function, hence some optimizations are made. """
        diff = np.mean(x) - np.mean(y)
        return np.dot(diff, diff)

    @staticmethod
    def l1norm_dist(x, y):
        return sum(abs(x - y))

    @staticmethod
    def cosine_dist(x, y):
        return 1 - np.dot(x, y) / ((np.dot(x, x) * np.dot(y, y)) ** 0.5)

    @staticmethod
    def hamming_np_bin_dist(x, y):
        return np.count_nonzero(x!=y)

    @staticmethod
    def normalised_euclidean(x, y, var_x, var_y):
        return np.var(x - y)/(var_x + var_y)

    @staticmethod
    def normalise_block_euclidean(x,y):

         no_of_block = 128

         x= x.reshape(-1,no_of_block)
         y= y.reshape(-1,no_of_block)
         z = x-y

         var_x = x.var(axis=1)
         var_y = y.var(axis=1)
         var_z = z.var(axis=1)

         #print("var x ,y",len(var_x+var_y))
         dist = (var_z/(var_x+var_y))/len(var_z)
         #d_mean = np.mean(dist)
         #dist = np.sum([x for x in dist if x <=d_mean])
         return np.sum(dist)


