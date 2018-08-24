# -*- coding: utf-8 -*-
"""
dicom2nifti

@author: abrys
"""
import nibabel
import nibabel.affines
import numpy
import scipy.ndimage


def resample_image(input_nifti):
    """
    Resample a gantry tilted image in place
    """
    # read the input image
    input_image = nibabel.load(input_nifti)
    # check if there is gantry tilting in the image
    if _is_gantry_tilted(input_image.affine):
        output_image = _resample_gantry_tilted(input_image)
        output_image.to_filename(input_nifti)


def _is_gantry_tilted(affine):
    # calculate the x, y and z directions in world coordinates
    world_x = numpy.transpose(numpy.dot(affine, [[1], [0], [0], [0]]))[0, :3]
    world_y = numpy.transpose(numpy.dot(affine, [[0], [1], [0], [0]]))[0, :3]
    world_z = numpy.transpose(numpy.dot(affine, [[0], [0], [1], [0]]))[0, :3]

    # normalize the vectors
    world_x /= numpy.linalg.norm(world_x)
    world_y /= numpy.linalg.norm(world_y)
    world_z /= numpy.linalg.norm(world_z)

    # calulate new orthogonal z and y and normalize
    orthogonal_z = numpy.cross(world_y, world_x)
    orthogonal_y = numpy.cross(world_x, world_z)
    orthogonal_z /= numpy.linalg.norm(orthogonal_z)
    orthogonal_y /= numpy.linalg.norm(orthogonal_y)

    if numpy.allclose(world_y, orthogonal_y) and numpy.allclose(world_z, orthogonal_z):
        return False

    return True


def _resample_gantry_tilted(original_image):
    """
    In this function we will create an orthogonal image and resample the original data to this space

    In this calculation we work in 3 spaces / coordinate systems

    - original image coordinates
    - world coordinates
    - "projected" coordinates

    This last one is a new rotated "orthogonal" coordinates system in mm where
    x and y are perpendicular with the x and y or the image

    We do the following steps
    - calculate a new "projection" coordinate system
    - calculate the world coordinates of all corners of the image in world coordinates
    - project the world coordinates of the corners on the projection coordinate system
    - calculate the min and max corners to get the orthogonal bounding box of the image in projected space
    - translate the origin back to world coordinages

    We now have the new xyz axis, origin and size and can create the new affine used for resampling
    """

    original_size = original_image.get_data().shape
    voxel_size = original_image.header.get_zooms()

    # Calculate the x and y axis as is (we assume these to be at 90 deg to each other)
    # We will then create a new z axis that is perpendicular to the new x,y plane
    x_axis_world = numpy.transpose(numpy.dot(original_image.affine, [[1], [0], [0], [0]]))[0, :3]
    y_axis_world = numpy.transpose(numpy.dot(original_image.affine, [[0], [1], [0], [0]]))[0, :3]
    x_axis_world /= numpy.linalg.norm(x_axis_world)  # normalization
    y_axis_world /= numpy.linalg.norm(y_axis_world)  # normalization
    z_axis_world = numpy.cross(y_axis_world, x_axis_world)  # calculate new z
    y_axis_world = numpy.cross(x_axis_world, z_axis_world)  # recalculate y in case x and y where not perpendicular

    # get all corners in image coordinates
    points_image = [[0, 0, 0],
                    [original_size[0], 0, 0],
                    [0, original_size[1], 0],
                    [0, 0, original_size[2]],
                    [original_size[0], original_size[1], 0],
                    [original_size[0], 0, original_size[2]],
                    [0, original_size[1], original_size[2]],
                    [original_size[0], original_size[1], original_size[2]]]

    # get all corners in world coordinates
    points_world = []
    for point in points_image:
        points_world.append(numpy.transpose(numpy.dot(original_image.affine,
                                                      [[point[0]], [point[1]], [point[2]], [1]]))[0, :3])

    # project all points on the axis
    # this will give the cooridites in "mm" but with the orientation of the new image
    # image coordinates projection = numpy.dot(point, axis)
    projections = []
    for point in points_world:
        projection = [numpy.dot(point, x_axis_world),
                      numpy.dot(point, y_axis_world),
                      numpy.dot(point, z_axis_world)]
        projections.append(projection)

    projections = numpy.array(projections)

    # get the lowest and highest x, y, z in "projection" space
    min_projected = numpy.amin(projections, axis=0)
    max_projected = numpy.amax(projections, axis=0)

    # calculate the image origin in world coordinates
    origin = min_projected[0] * x_axis_world + \
             min_projected[1] * y_axis_world + \
             min_projected[2] * z_axis_world

    # calculate the new image size in mm
    new_size_mm = max_projected - min_projected

    new_voxelsize = [abs(numpy.dot([voxel_size[0], 0, 0], x_axis_world)),
                     abs(numpy.dot([0, voxel_size[1], 0], y_axis_world)),
                     abs(numpy.dot([0, 0, voxel_size[2]], z_axis_world))]

    new_shape = numpy.ceil(new_size_mm / new_voxelsize).astype(numpy.int16)

    new_affine = _create_affine(x_axis_world, y_axis_world, z_axis_world, origin, new_voxelsize)

    combined_affine = numpy.linalg.inv(new_affine).dot(original_image.affine)
    matrix, offset = nibabel.affines.to_matvec(numpy.linalg.inv(combined_affine))
    new_data = scipy.ndimage.affine_transform(original_image.get_data(),
                                              matrix=matrix,
                                              offset=offset,
                                              output_shape=new_shape,
                                              output=original_image.get_data().dtype,
                                              order=1,  # 0 nn, 1 bilinear, ...
                                              mode='constant',
                                              cval=-1000,
                                              prefilter=False)

    return nibabel.Nifti1Image(new_data, new_affine)


def _create_affine(x_axis, y_axis, z_axis, image_pos, voxel_sizes):
    """
    Function to generate the affine matrix for a dicom series
    This method was based on (http://nipy.org/nibabel/dicom/dicom_orientation.html)

    :param sorted_dicoms: list with sorted dicom files
    """

    # Create affine matrix (http://nipy.sourceforge.net/nibabel/dicom/dicom_orientation.html#dicom-slice-affine)

    affine = numpy.array(
        [[x_axis[0] * voxel_sizes[0], y_axis[0] * voxel_sizes[1], z_axis[0] * voxel_sizes[2], image_pos[0]],
         [x_axis[1] * voxel_sizes[0], y_axis[1] * voxel_sizes[1], z_axis[1] * voxel_sizes[2], image_pos[1]],
         [x_axis[2] * voxel_sizes[0], y_axis[2] * voxel_sizes[1], z_axis[2] * voxel_sizes[2], image_pos[2]],
         [0, 0, 0, 1]])
    return affine
