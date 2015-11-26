"""
A simplified version of the svh converter wihtout bounding box
it looks for JPG images in the file `train.tar.gz` and `test.tar.gz`
that are of the size 64x64 with 3 channgels
and ignores everything else
"""
import os
import tarfile
import tempfile
import shutil
from collections import namedtuple, OrderedDict

import h5py
import numpy
from scipy.io import loadmat
from six import iteritems
from six.moves import range, zip
from PIL import Image

from fuel.converters.base import fill_hdf5_file, check_exists, progress_bar
from fuel.datasets import H5PYDataset


FORMAT_1_FILES = ['{}.tar.gz'.format(s) for s in ['train', 'test']]

@check_exists(required_files=FORMAT_1_FILES[:1])
def convert_jpgtgz(directory, output_directory,
                 output_filename=None):
    """Converts jpg tar.gz dataset to HDF5.

    Converts a jpg tar.gz dataset to an HDF5 dataset.

    Parameters
    ----------
    directory : str
        Directory in which input files reside.
    output_directory : str
        Directory in which to save the converted dataset.
    output_filename : str, optional
        Name of the saved dataset. Defaults to 'jpg.hdf5'

    Returns
    -------
    output_paths : tuple of str
        Single-element tuple containing the path to the converted dataset.

    """
    if not output_filename:
        output_filename = 'jpg.hdf5'

    try:
        output_path = os.path.join(output_directory, output_filename)
        h5file = h5py.File(output_path, mode='w')
        TMPDIR = tempfile.mkdtemp()

        sources = ('features',)
        source_dtypes = dict([(source, 'uint8') for source in sources])
        source_axis_labels = {
            'features': ('channel', 'height', 'width'),
        }

        splits = ('train','test')
        file_paths = dict(zip(splits, FORMAT_1_FILES))
        for split, path in file_paths.items():
            file_paths[split] = os.path.join(directory, path)

        # We first extract the data files in a temporary directory. While doing
        # that, we also count the number of examples for each split. Files are
        # extracted individually, which allows to display a progress bar. Since
        # the splits will be concatenated in the HDF5 file, we also compute the
        # start and stop intervals of each split within the concatenated array.
        def extract_tar(split):
            with tarfile.open(file_paths[split], 'r:gz') as f:
                members = f.getmembers()
                progress_bar_context = progress_bar(
                    name='{} file'.format(split), maxval=len(members),
                    prefix='Extracting')
                num_examples = 0
                with progress_bar_context as bar:
                    for i, member in enumerate(members):
                        if (member.name.endswith('.jpg') and not
                        os.path.basename(member.name).startswith('.')):
                            f.extract(member, path=os.path.join(TMPDIR,split))
                        bar.update(i)
                        num_examples += 1

                progress_bar_context = progress_bar(
                    name='{} file'.format(split), maxval=num_examples,
                    prefix='Validating')
                num_examples = 0
                with progress_bar_context as bar:
                    for root, dirs, files in os.walk(os.path.join(TMPDIR,split)):
                        for file in files:
                            if file.endswith('.jpg') and not file.startswith('.'):
                                image_path = os.path.join(root, file)
                                try:
                                    im = Image.open(image_path)
                                    im = numpy.asarray(im)
                                    assert im.shape == (64,64,3), 'Images should be 64x64 RGB and not '+str(im.shape)
                                    num_examples += 1
                                    bar.update(num_examples)
                                except:
                                    os.remove(image_path)
            return num_examples

        examples_per_split = OrderedDict(
            [(split, extract_tar(split)) for split in splits])
        cumulative_num_examples = numpy.cumsum(
            [0] + list(examples_per_split.values()))
        num_examples = cumulative_num_examples[-1]
        intervals = zip(cumulative_num_examples[:-1],
                        cumulative_num_examples[1:])
        split_intervals = dict(zip(splits, intervals))

        # The start and stop indices are used to create a split dict that will
        # be parsed into the split array required by the H5PYDataset interface.
        # The split dict is organized as follows:
        #
        #     dict(split -> dict(source -> (start, stop)))
        #
        split_dict = OrderedDict([
            (split, OrderedDict([(s, split_intervals[split])
                                 for s in sources]))
            for split in splits])
        h5file.attrs['split'] = H5PYDataset.create_split_array(split_dict)

        # We then prepare the HDF5 dataset. This involves creating datasets to
        # store data sources and datasets to store auxiliary information
        # (namely the shapes for variable-length axes, and labels to indicate
        # what these variable-length axes represent).
        def make_vlen_dataset(source):
            # Create a variable-length 1D dataset
            dtype = h5py.special_dtype(vlen=numpy.dtype(source_dtypes[source]))
            dataset = h5file.create_dataset(
                source, (num_examples,), dtype=dtype)
            # Create a dataset to store variable-length shapes.
            axis_labels = source_axis_labels[source]
            dataset_shapes = h5file.create_dataset(
                '{}_shapes'.format(source), (num_examples, len(axis_labels)),
                dtype='uint16')
            # Create a dataset to store labels for variable-length axes.
            dataset_vlen_axis_labels = h5file.create_dataset(
                '{}_vlen_axis_labels'.format(source), (len(axis_labels),),
                dtype='S{}'.format(
                    numpy.max([len(label) for label in axis_labels])))
            # Fill variable-length axis labels
            dataset_vlen_axis_labels[...] = [
                label.encode('utf8') for label in axis_labels]
            # Attach auxiliary datasets as dimension scales of the
            # variable-length 1D dataset. This is in accordance with the
            # H5PYDataset interface.
            dataset.dims.create_scale(dataset_shapes, 'shapes')
            dataset.dims[0].attach_scale(dataset_shapes)
            dataset.dims.create_scale(dataset_vlen_axis_labels, 'shape_labels')
            dataset.dims[0].attach_scale(dataset_vlen_axis_labels)
            # Tag fixed-length axis with its label
            dataset.dims[0].label = 'batch'

        for source in sources:
            make_vlen_dataset(source)

        # The final step is to fill the HDF5 file.
        def fill_split(split, bar=None):
            jpgfiles = []
            for root, dirs, files in os.walk(os.path.join(TMPDIR,split)):
                for file in files:
                    if file.endswith('.jpg') and not file.startswith('.'):
                        jpgfiles.append(os.path.join(root,file))
            assert len(jpgfiles) == examples_per_split[split],'%d %d'%(len(jpgfiles), examples_per_split[split])

            for image_number in range(examples_per_split[split]):
                image_path = jpgfiles[image_number]
                image = numpy.asarray(
                    Image.open(image_path)).transpose(2, 0, 1)
                index = image_number + split_intervals[split][0]

                h5file['features'][index] = image.flatten()
                h5file['features'].dims[0]['shapes'][index] = image.shape

                if image_number % 1000 == 0:
                    h5file.flush()
                if bar:
                    bar.update(index)

        with progress_bar('jpgtgz', num_examples) as bar:
            for split in splits:
                fill_split(split, bar=bar)
    finally:
        if os.path.isdir(TMPDIR):
            shutil.rmtree(TMPDIR)
        h5file.flush()
        h5file.close()

    return (output_path,)


def fill_subparser(subparser):
    """Sets up a subparser to convert the jpg dataset files.

    Parameters
    ----------
    subparser : :class:`argparse.ArgumentParser`
        Subparser handling the `jpg` command.

    """
    return convert_jpgtgz
