"""
The :mod:`sklearn.feature_extraction.image` submodule gathers utilities to
extract features from images.
"""

# Authors: Emmanuelle Gouillart <emmanuelle.gouillart@normalesup.org>
#          Gael Varoquaux <gael.varoquaux@normalesup.org>
#          Olivier Grisel
#          Vlad Niculae
# License: BSD 3 clause

from itertools import product
import numbers
import numpy as np
from scipy import sparse
from numpy.lib.stride_tricks import as_strided

import os
import six

import warnings
import numbers
import scipy.sparse as sp
from scipy import __version__ as scipy_version
from distutils.version import LooseVersion

from numpy.core.numeric import ComplexWarning

LARGE_SPARSE_SUPPORTED = LooseVersion(scipy_version) >= '0.14.0'

###############################################################################
# From an image to a graph

_global_config = {
    'assume_finite': bool(os.environ.get('SKLEARN_ASSUME_FINITE', False)),
    'working_memory': int(os.environ.get('SKLEARN_WORKING_MEMORY', 1024))
}


def _get_config():
    """Retrieve current values for configuration set by :func:`set_config`

    Returns
    -------
    config : dict
        Keys are parameter names that can be passed to :func:`set_config`.
    """
    return _global_config.copy()


def _make_edges_3d(n_x, n_y, n_z=1):
    """Returns a list of edges for a 3D image.

    Parameters
    ===========
    n_x : integer
        The size of the grid in the x direction.
    n_y : integer
        The size of the grid in the y direction.
    n_z : integer, optional
        The size of the grid in the z direction, defaults to 1
    """
    vertices = np.arange(n_x * n_y * n_z).reshape((n_x, n_y, n_z))
    edges_deep = np.vstack((vertices[:, :, :-1].ravel(),
                            vertices[:, :, 1:].ravel()))
    edges_right = np.vstack((vertices[:, :-1].ravel(),
                             vertices[:, 1:].ravel()))
    edges_down = np.vstack((vertices[:-1].ravel(), vertices[1:].ravel()))
    edges = np.hstack((edges_deep, edges_right, edges_down))
    return edges


def _compute_gradient_3d(edges, img):
    n_x, n_y, n_z = img.shape
    gradient = np.abs(img[edges[0] // (n_y * n_z),
                      (edges[0] % (n_y * n_z)) // n_z,
                      (edges[0] % (n_y * n_z)) % n_z] -
                      img[edges[1] // (n_y * n_z),
                      (edges[1] % (n_y * n_z)) // n_z,
                      (edges[1] % (n_y * n_z)) % n_z])
    return gradient


# XXX: Why mask the image after computing the weights?

def _mask_edges_weights(mask, edges, weights=None):
    """Apply a mask to edges (weighted or not)"""
    inds = np.arange(mask.size)
    inds = inds[mask.ravel()]
    ind_mask = np.logical_and(np.in1d(edges[0], inds),
                              np.in1d(edges[1], inds))
    edges = edges[:, ind_mask]
    if weights is not None:
        weights = weights[ind_mask]
    if len(edges.ravel()):
        maxval = edges.max()
    else:
        maxval = 0
    order = np.searchsorted(np.unique(edges.ravel()), np.arange(maxval + 1))
    edges = order[edges]
    if weights is None:
        return edges
    else:
        return edges, weights


def _to_graph(n_x, n_y, n_z, mask=None, img=None,
              return_as=sparse.coo_matrix, dtype=None):
    """Auxiliary function for img_to_graph and grid_to_graph
    """
    edges = _make_edges_3d(n_x, n_y, n_z)

    if dtype is None:
        if img is None:
            dtype = np.int
        else:
            dtype = img.dtype

    if img is not None:
        img = np.atleast_3d(img)
        weights = _compute_gradient_3d(edges, img)
        if mask is not None:
            edges, weights = _mask_edges_weights(mask, edges, weights)
            diag = img.squeeze()[mask]
        else:
            diag = img.ravel()
        n_voxels = diag.size
    else:
        if mask is not None:
            mask = mask.astype(dtype=np.bool, copy=False)
            mask = np.asarray(mask, dtype=np.bool)
            edges = _mask_edges_weights(mask, edges)
            n_voxels = np.sum(mask)
        else:
            n_voxels = n_x * n_y * n_z
        weights = np.ones(edges.shape[1], dtype=dtype)
        diag = np.ones(n_voxels, dtype=dtype)

    diag_idx = np.arange(n_voxels)
    i_idx = np.hstack((edges[0], edges[1]))
    j_idx = np.hstack((edges[1], edges[0]))
    graph = sparse.coo_matrix((np.hstack((weights, weights, diag)),
                              (np.hstack((i_idx, diag_idx)),
                               np.hstack((j_idx, diag_idx)))),
                              (n_voxels, n_voxels),
                              dtype=dtype)
    if return_as is np.ndarray:
        return graph.toarray()
    return return_as(graph)


def img_to_graph(img, mask=None, return_as=sparse.coo_matrix, dtype=None):
    """Graph of the pixel-to-pixel gradient connections

    Edges are weighted with the gradient values.

    Read more in the :ref:`User Guide <image_feature_extraction>`.

    Parameters
    ----------
    img : ndarray, 2D or 3D
        2D or 3D image
    mask : ndarray of booleans, optional
        An optional mask of the image, to consider only part of the
        pixels.
    return_as : np.ndarray or a sparse matrix class, optional
        The class to use to build the returned adjacency matrix.
    dtype : None or dtype, optional
        The data of the returned sparse matrix. By default it is the
        dtype of img

    Notes
    -----
    For scikit-learn versions 0.14.1 and prior, return_as=np.ndarray was
    handled by returning a dense np.matrix instance.  Going forward, np.ndarray
    returns an np.ndarray, as expected.

    For compatibility, user code relying on this method should wrap its
    calls in ``np.asarray`` to avoid type issues.
    """
    img = np.atleast_3d(img)
    n_x, n_y, n_z = img.shape
    return _to_graph(n_x, n_y, n_z, mask, img, return_as, dtype)


def grid_to_graph(n_x, n_y, n_z=1, mask=None, return_as=sparse.coo_matrix,
                  dtype=np.int):
    """Graph of the pixel-to-pixel connections

    Edges exist if 2 voxels are connected.

    Parameters
    ----------
    n_x : int
        Dimension in x axis
    n_y : int
        Dimension in y axis
    n_z : int, optional, default 1
        Dimension in z axis
    mask : ndarray of booleans, optional
        An optional mask of the image, to consider only part of the
        pixels.
    return_as : np.ndarray or a sparse matrix class, optional
        The class to use to build the returned adjacency matrix.
    dtype : dtype, optional, default int
        The data of the returned sparse matrix. By default it is int

    Notes
    -----
    For scikit-learn versions 0.14.1 and prior, return_as=np.ndarray was
    handled by returning a dense np.matrix instance.  Going forward, np.ndarray
    returns an np.ndarray, as expected.

    For compatibility, user code relying on this method should wrap its
    calls in ``np.asarray`` to avoid type issues.
    """
    return _to_graph(n_x, n_y, n_z, mask=mask, return_as=return_as,
                     dtype=dtype)


###############################################################################
# From an image to a set of small image patches

def _compute_n_patches(i_h, i_w, p_h, p_w, max_patches=None):
    """Compute the number of patches that will be extracted in an image.

    Read more in the :ref:`User Guide <image_feature_extraction>`.

    Parameters
    ----------
    i_h : int
        The image height
    i_w : int
        The image with
    p_h : int
        The height of a patch
    p_w : int
        The width of a patch
    max_patches : integer or float, optional default is None
        The maximum number of patches to extract. If max_patches is a float
        between 0 and 1, it is taken to be a proportion of the total number
        of patches.
    """
    n_h = i_h - p_h + 1
    n_w = i_w - p_w + 1
    all_patches = n_h * n_w

    if max_patches:
        if (isinstance(max_patches, (numbers.Integral))
                and max_patches < all_patches):
            return max_patches
        elif (isinstance(max_patches, (numbers.Integral))
              and max_patches >= all_patches):
            return all_patches
        elif (isinstance(max_patches, (numbers.Real))
                and 0 < max_patches < 1):
            return int(max_patches * all_patches)
        else:
            raise ValueError("Invalid value for max_patches: %r" % max_patches)
    else:
        return all_patches


def extract_patches(arr, patch_shape=8, extraction_step=1):
    """Extracts patches of any n-dimensional array in place using strides.

    Given an n-dimensional array it will return a 2n-dimensional array with
    the first n dimensions indexing patch position and the last n indexing
    the patch content. This operation is immediate (O(1)). A reshape
    performed on the first n dimensions will cause numpy to copy data, leading
    to a list of extracted patches.

    Read more in the :ref:`User Guide <image_feature_extraction>`.

    Parameters
    ----------
    arr : ndarray
        n-dimensional array of which patches are to be extracted

    patch_shape : integer or tuple of length arr.ndim
        Indicates the shape of the patches to be extracted. If an
        integer is given, the shape will be a hypercube of
        sidelength given by its value.

    extraction_step : integer or tuple of length arr.ndim
        Indicates step size at which extraction shall be performed.
        If integer is given, then the step is uniform in all dimensions.


    Returns
    -------
    patches : strided ndarray
        2n-dimensional array indexing patches on first n dimensions and
        containing patches on the last n dimensions. These dimensions
        are fake, but this way no data is copied. A simple reshape invokes
        a copying operation to obtain a list of patches:
        result.reshape([-1] + list(patch_shape))
    """

    arr_ndim = arr.ndim

    if isinstance(patch_shape, numbers.Number):
        patch_shape = tuple([patch_shape] * arr_ndim)
    if isinstance(extraction_step, numbers.Number):
        extraction_step = tuple([extraction_step] * arr_ndim)

    patch_strides = arr.strides

    slices = tuple(slice(None, None, st) for st in extraction_step)
    indexing_strides = arr[slices].strides

    patch_indices_shape = ((np.array(arr.shape) - np.array(patch_shape)) //
                           np.array(extraction_step)) + 1

    shape = tuple(list(patch_indices_shape) + list(patch_shape))
    strides = tuple(list(indexing_strides) + list(patch_strides))

    patches = as_strided(arr, shape=shape, strides=strides)
    return patches


def extract_patches_2d(image, patch_size, max_patches=None, random_state=None):
    """Reshape a 2D image into a collection of patches

    The resulting patches are allocated in a dedicated array.

    Read more in the :ref:`User Guide <image_feature_extraction>`.

    Parameters
    ----------
    image : array, shape = (image_height, image_width) or
        (image_height, image_width, n_channels)
        The original image data. For color images, the last dimension specifies
        the channel: a RGB image would have `n_channels=3`.

    patch_size : tuple of ints (patch_height, patch_width)
        the dimensions of one patch

    max_patches : integer or float, optional default is None
        The maximum number of patches to extract. If max_patches is a float
        between 0 and 1, it is taken to be a proportion of the total number
        of patches.

    random_state : int, RandomState instance or None, optional (default=None)
        Pseudo number generator state used for random sampling to use if
        `max_patches` is not None.  If int, random_state is the seed used by
        the random number generator; If RandomState instance, random_state is
        the random number generator; If None, the random number generator is
        the RandomState instance used by `np.random`.

    Returns
    -------
    patches : array, shape = (n_patches, patch_height, patch_width) or
         (n_patches, patch_height, patch_width, n_channels)
         The collection of patches extracted from the image, where `n_patches`
         is either `max_patches` or the total number of patches that can be
         extracted.

    Examples
    --------

    >>> from sklearn.feature_extraction import image
    >>> one_image = np.arange(16).reshape((4, 4))
    >>> one_image
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])
    >>> patches = image.extract_patches_2d(one_image, (2, 2))
    >>> print(patches.shape)
    (9, 2, 2)
    >>> patches[0]
    array([[0, 1],
           [4, 5]])
    >>> patches[1]
    array([[1, 2],
           [5, 6]])
    >>> patches[8]
    array([[10, 11],
           [14, 15]])
    """
    i_h, i_w = image.shape[:2]
    p_h, p_w = patch_size

    if p_h > i_h:
        raise ValueError("Height of the patch should be less than the height"
                         " of the image.")

    if p_w > i_w:
        raise ValueError("Width of the patch should be less than the width"
                         " of the image.")

    image = check_array(image, allow_nd=True)
    image = image.reshape((i_h, i_w, -1))
    n_colors = image.shape[-1]

    extracted_patches = extract_patches(image,
                                        patch_shape=(p_h, p_w, n_colors),
                                        extraction_step=1)

    n_patches = _compute_n_patches(i_h, i_w, p_h, p_w, max_patches)
    if max_patches:
        rng = check_random_state(random_state)
        i_s = rng.randint(i_h - p_h + 1, size=n_patches)
        j_s = rng.randint(i_w - p_w + 1, size=n_patches)
        patches = extracted_patches[i_s, j_s, 0]
    else:
        patches = extracted_patches

    print(patches.shape)
    print(p_h, p_w, n_colors)
    patches = patches.reshape(-1, p_h, p_w, n_colors)
    # remove the color dimension if useless
    if patches.shape[-1] == 1:
        return patches.reshape((n_patches, p_h, p_w))
    else:
        return patches


def reconstruct_from_patches_2d(patches, image_size):
    """Reconstruct the image from all of its patches.

    Patches are assumed to overlap and the image is constructed by filling in
    the patches from left to right, top to bottom, averaging the overlapping
    regions.

    Read more in the :ref:`User Guide <image_feature_extraction>`.

    Parameters
    ----------
    patches : array, shape = (n_patches, patch_height, patch_width) or
        (n_patches, patch_height, patch_width, n_channels)
        The complete set of patches. If the patches contain colour information,
        channels are indexed along the last dimension: RGB patches would
        have `n_channels=3`.

    image_size : tuple of ints (image_height, image_width) or
        (image_height, image_width, n_channels)
        the size of the image that will be reconstructed

    Returns
    -------
    image : array, shape = image_size
        the reconstructed image

    """
    i_h, i_w = image_size[:2]
    p_h, p_w = patches.shape[1:3]
    img = np.zeros(image_size)
    # compute the dimensions of the patches array
    n_h = i_h - p_h + 1
    n_w = i_w - p_w + 1
    for p, (i, j) in zip(patches, product(range(n_h), range(n_w))):
        img[i:i + p_h, j:j + p_w] += p

    for i in range(i_h):
        for j in range(i_w):
            # divide by the amount of overlap
            # XXX: is this the most efficient way? memory-wise yes, cpu wise?
            img[i, j] /= float(min(i + 1, p_h, i_h - i) *
                               min(j + 1, p_w, i_w - j))
    return img

def _ensure_no_complex_data(array):
    if hasattr(array, 'dtype') and array.dtype is not None \
            and hasattr(array.dtype, 'kind') and array.dtype.kind == "c":
        raise ValueError("Complex data not supported\n"
                         "{}\n".format(array))


def check_array(array, accept_sparse=False, accept_large_sparse=True,
                dtype="numeric", order=None, copy=False, force_all_finite=True,
                ensure_2d=True, allow_nd=False, ensure_min_samples=1,
                ensure_min_features=1, warn_on_dtype=False, estimator=None):

    """Input validation on an array, list, sparse matrix or similar.

    By default, the input is converted to an at least 2D numpy array.
    If the dtype of the array is object, attempt converting to float,
    raising on failure.

    Parameters
    ----------
    array : object
        Input object to check / convert.

    accept_sparse : string, boolean or list/tuple of strings (default=False)
        String[s] representing allowed sparse matrix formats, such as 'csc',
        'csr', etc. If the input is sparse but not in the allowed format,
        it will be converted to the first listed format. True allows the input
        to be any format. False means that a sparse matrix input will
        raise an error.

        .. deprecated:: 0.19
           Passing 'None' to parameter ``accept_sparse`` in methods is
           deprecated in version 0.19 "and will be removed in 0.21. Use
           ``accept_sparse=False`` instead.

    accept_large_sparse : bool (default=True)
        If a CSR, CSC, COO or BSR sparse matrix is supplied and accepted by
        accept_sparse, accept_large_sparse=False will cause it to be accepted
        only if its indices are stored with a 32-bit dtype.

        .. versionadded:: 0.20

    dtype : string, type, list of types or None (default="numeric")
        Data type of result. If None, the dtype of the input is preserved.
        If "numeric", dtype is preserved unless array.dtype is object.
        If dtype is a list of types, conversion on the first type is only
        performed if the dtype of the input is not in the list.

    order : 'F', 'C' or None (default=None)
        Whether an array will be forced to be fortran or c-style.
        When order is None (default), then if copy=False, nothing is ensured
        about the memory layout of the output array; otherwise (copy=True)
        the memory layout of the returned array is kept as close as possible
        to the original array.

    copy : boolean (default=False)
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    force_all_finite : boolean or 'allow-nan', (default=True)
        Whether to raise an error on np.inf and np.nan in X. The possibilities
        are:

        - True: Force all values of X to be finite.
        - False: accept both np.inf and np.nan in X.
        - 'allow-nan':  accept  only  np.nan  values in  X.  Values  cannot  be
          infinite.

        .. versionadded:: 0.20
           ``force_all_finite`` accepts the string ``'allow-nan'``.

    ensure_2d : boolean (default=True)
        Whether to raise a value error if X is not 2d.

    allow_nd : boolean (default=False)
        Whether to allow X.ndim > 2.

    ensure_min_samples : int (default=1)
        Make sure that the array has a minimum number of samples in its first
        axis (rows for a 2D array). Setting to 0 disables this check.

    ensure_min_features : int (default=1)
        Make sure that the 2D array has some minimum number of features
        (columns). The default value of 1 rejects empty datasets.
        This check is only enforced when the input data has effectively 2
        dimensions or is originally 1D and ``ensure_2d`` is True. Setting to 0
        disables this check.

    warn_on_dtype : boolean (default=False)
        Raise DataConversionWarning if the dtype of the input data structure
        does not match the requested dtype, causing a memory copy.

    estimator : str or estimator instance (default=None)
        If passed, include the name of the estimator in warning messages.

    Returns
    -------
    X_converted : object
        The converted and validated X.

    """
    # accept_sparse 'None' deprecation check
    if accept_sparse is None:
        warnings.warn(
            "Passing 'None' to parameter 'accept_sparse' in methods "
            "check_array and check_X_y is deprecated in version 0.19 "
            "and will be removed in 0.21. Use 'accept_sparse=False' "
            " instead.", DeprecationWarning)
        accept_sparse = False

    # store reference to original array to check if copy is needed when
    # function returns
    array_orig = array

    # store whether originally we wanted numeric dtype
    dtype_numeric = isinstance(dtype, six.string_types) and dtype == "numeric"

    dtype_orig = getattr(array, "dtype", None)
    if not hasattr(dtype_orig, 'kind'):
        # not a data type (e.g. a column named dtype in a pandas DataFrame)
        dtype_orig = None

    # check if the object contains several dtypes (typically a pandas
    # DataFrame), and store them. If not, store None.
    dtypes_orig = None
    if hasattr(array, "dtypes") and hasattr(array, "__array__"):
        dtypes_orig = np.array(array.dtypes)

    if dtype_numeric:
        if dtype_orig is not None and dtype_orig.kind == "O":
            # if input is object, convert to float.
            dtype = np.float64
        else:
            dtype = None

    if isinstance(dtype, (list, tuple)):
        if dtype_orig is not None and dtype_orig in dtype:
            # no dtype conversion required
            dtype = None
        else:
            # dtype conversion required. Let's select the first element of the
            # list of accepted types.
            dtype = dtype[0]

    if force_all_finite not in (True, False, 'allow-nan'):
        raise ValueError('force_all_finite should be a bool or "allow-nan"'
                         '. Got {!r} instead'.format(force_all_finite))

    if estimator is not None:
        if isinstance(estimator, six.string_types):
            estimator_name = estimator
        else:
            estimator_name = estimator.__class__.__name__
    else:
        estimator_name = "Estimator"
    context = " by %s" % estimator_name if estimator is not None else ""

    if sp.issparse(array):
        _ensure_no_complex_data(array)
        array = _ensure_sparse_format(array, accept_sparse=accept_sparse,
                                      dtype=dtype, copy=copy,
                                      force_all_finite=force_all_finite,
                                      accept_large_sparse=accept_large_sparse)
    else:
        # If np.array(..) gives ComplexWarning, then we convert the warning
        # to an error. This is needed because specifying a non complex
        # dtype to the function converts complex to real dtype,
        # thereby passing the test made in the lines following the scope
        # of warnings context manager.
        with warnings.catch_warnings():
            try:
                warnings.simplefilter('error', ComplexWarning)
                array = np.asarray(array, dtype=dtype, order=order)
            except ComplexWarning:
                raise ValueError("Complex data not supported\n"
                                 "{}\n".format(array))

        # It is possible that the np.array(..) gave no warning. This happens
        # when no dtype conversion happened, for example dtype = None. The
        # result is that np.array(..) produces an array of complex dtype
        # and we need to catch and raise exception for such cases.
        _ensure_no_complex_data(array)

        if ensure_2d:
            # If input is scalar raise error
            if array.ndim == 0:
                raise ValueError(
                    "Expected 2D array, got scalar array instead:\narray={}.\n"
                    "Reshape your data either using array.reshape(-1, 1) if "
                    "your data has a single feature or array.reshape(1, -1) "
                    "if it contains a single sample.".format(array))
            # If input is 1D raise error
            if array.ndim == 1:
                raise ValueError(
                    "Expected 2D array, got 1D array instead:\narray={}.\n"
                    "Reshape your data either using array.reshape(-1, 1) if "
                    "your data has a single feature or array.reshape(1, -1) "
                    "if it contains a single sample.".format(array))

        # in the future np.flexible dtypes will be handled like object dtypes
        if dtype_numeric and np.issubdtype(array.dtype, np.flexible):
            warnings.warn(
                "Beginning in version 0.22, arrays of bytes/strings will be "
                "converted to decimal numbers if dtype='numeric'. "
                "It is recommended that you convert the array to "
                "a float dtype before using it in scikit-learn, "
                "for example by using "
                "your_array = your_array.astype(np.float64).",
                FutureWarning)

        # make sure we actually converted to numeric:
        if dtype_numeric and array.dtype.kind == "O":
            array = array.astype(np.float64)
        if not allow_nd and array.ndim >= 3:
            raise ValueError("Found array with dim %d. %s expected <= 2."
                             % (array.ndim, estimator_name))
        if force_all_finite:
            _assert_all_finite(array,
                               allow_nan=force_all_finite == 'allow-nan')

    shape_repr = _shape_repr(array.shape)
    if ensure_min_samples > 0:
        n_samples = _num_samples(array)
        if n_samples < ensure_min_samples:
            raise ValueError("Found array with %d sample(s) (shape=%s) while a"
                             " minimum of %d is required%s."
                             % (n_samples, shape_repr, ensure_min_samples,
                                context))

    if ensure_min_features > 0 and array.ndim == 2:
        n_features = array.shape[1]
        if n_features < ensure_min_features:
            raise ValueError("Found array with %d feature(s) (shape=%s) while"
                             " a minimum of %d is required%s."
                             % (n_features, shape_repr, ensure_min_features,
                                context))

    if warn_on_dtype and dtype_orig is not None and array.dtype != dtype_orig:
        msg = ("Data with input dtype %s was converted to %s%s."
               % (dtype_orig, array.dtype, context))
        print("Data Conversion Warning")

    if copy and np.may_share_memory(array, array_orig):
        array = np.array(array, dtype=dtype, order=order)

    if (warn_on_dtype and dtypes_orig is not None and
            {array.dtype} != set(dtypes_orig)):
        # if there was at the beginning some other types than the final one
        # (for instance in a DataFrame that can contain several dtypes) then
        # some data must have been converted
        msg = ("Data with input dtype %s were all converted to %s%s."
               % (', '.join(map(str, sorted(set(dtypes_orig)))), array.dtype,
                  context))
        print("Data Conversion Warning")

    return array

def _ensure_sparse_format(spmatrix, accept_sparse, dtype, copy,
                          force_all_finite, accept_large_sparse):
    """Convert a sparse matrix to a given format.

    Checks the sparse format of spmatrix and converts if necessary.

    Parameters
    ----------
    spmatrix : scipy sparse matrix
        Input to validate and convert.

    accept_sparse : string, boolean or list/tuple of strings
        String[s] representing allowed sparse matrix formats ('csc',
        'csr', 'coo', 'dok', 'bsr', 'lil', 'dia'). If the input is sparse but
        not in the allowed format, it will be converted to the first listed
        format. True allows the input to be any format. False means
        that a sparse matrix input will raise an error.

    dtype : string, type or None
        Data type of result. If None, the dtype of the input is preserved.

    copy : boolean
        Whether a forced copy will be triggered. If copy=False, a copy might
        be triggered by a conversion.

    force_all_finite : boolean or 'allow-nan', (default=True)
        Whether to raise an error on np.inf and np.nan in X. The possibilities
        are:

        - True: Force all values of X to be finite.
        - False: accept both np.inf and np.nan in X.
        - 'allow-nan':  accept  only  np.nan  values in  X.  Values  cannot  be
          infinite.

        .. versionadded:: 0.20
           ``force_all_finite`` accepts the string ``'allow-nan'``.

    Returns
    -------
    spmatrix_converted : scipy sparse matrix.
        Matrix that is ensured to have an allowed type.
    """
    if dtype is None:
        dtype = spmatrix.dtype

    changed_format = False

    if isinstance(accept_sparse, six.string_types):
        accept_sparse = [accept_sparse]

    # Indices dtype validation
    _check_large_sparse(spmatrix, accept_large_sparse)

    if accept_sparse is False:
        raise TypeError('A sparse matrix was passed, but dense '
                        'data is required. Use X.toarray() to '
                        'convert to a dense numpy array.')
    elif isinstance(accept_sparse, (list, tuple)):
        if len(accept_sparse) == 0:
            raise ValueError("When providing 'accept_sparse' "
                             "as a tuple or list, it must contain at "
                             "least one string value.")
        # ensure correct sparse format
        if spmatrix.format not in accept_sparse:
            # create new with correct sparse
            spmatrix = spmatrix.asformat(accept_sparse[0])
            changed_format = True
    elif accept_sparse is not True:
        # any other type
        raise ValueError("Parameter 'accept_sparse' should be a string, "
                         "boolean or list of strings. You provided "
                         "'accept_sparse={}'.".format(accept_sparse))

    if dtype != spmatrix.dtype:
        # convert dtype
        spmatrix = spmatrix.astype(dtype)
    elif copy and not changed_format:
        # force copy
        spmatrix = spmatrix.copy()

    if force_all_finite:
        if not hasattr(spmatrix, "data"):
            warnings.warn("Can't check %s sparse matrix for nan or inf."
                          % spmatrix.format)
        else:
            _assert_all_finite(spmatrix.data,
                               allow_nan=force_all_finite == 'allow-nan')

    return spmatrix

def _assert_all_finite(X, allow_nan=False):
    """Like assert_all_finite, but only for ndarray."""
    if _get_config()['assume_finite']:
        return
    X = np.asanyarray(X)
    # First try an O(n) time, O(1) space solution for the common case that
    # everything is finite; fall back to O(n) space np.isfinite to prevent
    # false positives from overflow in sum method.
    is_float = X.dtype.kind in 'fc'
    if is_float and np.isfinite(X.sum()):
        pass
    elif is_float:
        msg_err = "Input contains {} or a value too large for {!r}."
        if (allow_nan and np.isinf(X).any() or
                not allow_nan and not np.isfinite(X).all()):
            type_err = 'infinity' if allow_nan else 'NaN, infinity'
            raise ValueError(msg_err.format(type_err, X.dtype))

def _num_samples(x):
    """Return number of samples in array-like x."""
    if hasattr(x, 'fit') and callable(x.fit):
        # Don't get num_samples from an ensembles length!
        raise TypeError('Expected sequence or array-like, got '
                        'estimator %s' % x)
    if not hasattr(x, '__len__') and not hasattr(x, 'shape'):
        if hasattr(x, '__array__'):
            x = np.asarray(x)
        else:
            raise TypeError("Expected sequence or array-like, got %s" %
                            type(x))
    if hasattr(x, 'shape'):
        if len(x.shape) == 0:
            raise TypeError("Singleton array %r cannot be considered"
                            " a valid collection." % x)
        return x.shape[0]
    else:
        return len(x)

def _shape_repr(shape):
    """Return a platform independent representation of an array shape

    Under Python 2, the `long` type introduces an 'L' suffix when using the
    default %r format for tuples of integers (typically used to store the shape
    of an array).

    Under Windows 64 bit (and Python 2), the `long` type is used by default
    in numpy shapes even when the integer dimensions are well below 32 bit.
    The platform specific type causes string messages or doctests to change
    from one platform to another which is not desirable.

    Under Python 3, there is no more `long` type so the `L` suffix is never
    introduced in string representation.

    >>> _shape_repr((1, 2))
    '(1, 2)'
    >>> one = 2 ** 64 / 2 ** 64  # force an upcast to `long` under Python 2
    >>> _shape_repr((one, 2 * one))
    '(1, 2)'
    >>> _shape_repr((1,))
    '(1,)'
    >>> _shape_repr(())
    '()'
    """
    if len(shape) == 0:
        return "()"
    joined = ", ".join("%d" % e for e in shape)
    if len(shape) == 1:
        # special notation for singleton tuples
        joined += ','
    return "(%s)" % joined

def _check_large_sparse(X, accept_large_sparse=False):
    """Raise a ValueError if X has 64bit indices and accept_large_sparse=False
    """
    if not (accept_large_sparse and LARGE_SPARSE_SUPPORTED):
        supported_indices = ["int32"]
        if X.getformat() == "coo":
            index_keys = ['col', 'row']
        elif X.getformat() in ["csr", "csc", "bsr"]:
            index_keys = ['indices', 'indptr']
        else:
            return
        for key in index_keys:
            indices_datatype = getattr(X, key).dtype
            if (indices_datatype not in supported_indices):
                if not LARGE_SPARSE_SUPPORTED:
                    raise ValueError("Scipy version %s does not support large"
                                     " indices, please upgrade your scipy"
                                     " to 0.14.0 or above" % scipy_version)
                raise ValueError("Only sparse matrices with 32-bit integer"
                                 " indices are accepted. Got %s indices."
                                 % indices_datatype)

def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)
