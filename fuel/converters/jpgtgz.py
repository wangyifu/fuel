"""
Looks for `.jpg` images in the file `train.tar.gz` and `test.tar.gz`
all images should have the same shape as the first image
ignores everything else.
Use hash to check that each image appears only once (no duplicated samples
and no overlap between train and test.)
The images are stored in the "features" source

If a train.taregt.csv (or test.target.csv) file exists
it is read, the "name" column should match the basename of the image
(without any path and without the .jpg extension)
if a match is found for an image then
the "target" column is copied as (target,1)
if there is no match then (0,0) is copied
into the "targets" source
"""
import os
import tarfile
import tempfile
import shutil
from collections import namedtuple, OrderedDict

import h5py
import numpy
import pandas as pd
from scipy.io import loadmat
from six import iteritems
from six.moves import range, zip
from PIL import Image

from fuel.converters.base import fill_hdf5_file, check_exists, progress_bar
from fuel.datasets import H5PYDataset
import hashlib

FORMAT_1_FILES = ['{}.tar.gz'.format(s) for s in ['train', 'test']]
TARGET_FILES = ['{}.target.csv'.format(s) for s in ['train', 'test']]

S = None

# @check_exists(required_files=FORMAT_1_FILES[:1])
def convert_jpgtgz(target, onlytarget, directory, output_directory,
                 output_filename=None):
    """Converts jpg tar.gz dataset to HDF5.

    Converts a jpg tar.gz dataset to an HDF5 dataset.

    Parameters
    ----------
    target: bool
        Also addd a targets source to the file.
        The targets are computed based on train/test.target.csv.
        Each image receive two values [0] - is the target value and
        [1] - is a mask bit saying if a target is at all defined for the image
    onlytarget: bool
        same as target but dont take images that dont have a target value defined.
        there is no mask
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
    dotarget = target | onlytarget
    # if onlytarget then there is just one value otherwise
    # this will have two values 0-target 1-mask saying if the target should be used
    target_dim = 1 if onlytarget else 2
    if not output_filename:
        output_filename = 'jpg.hdf5'

    output_path = os.path.join(output_directory, output_filename)
    h5file = h5py.File(output_path, mode='w')
    try:
        TMPDIR = tempfile.mkdtemp()

        sources = ('features','targets') if dotarget else ('features',)
        source_dtypes = dict([(source, 'uint8') for source in sources])
        source_axis_labels = {
            'features': ('channel', 'height', 'width'),
            'targets': ('index',),
        }

        splits = ('train','test')
        file_paths = dict(zip(splits, FORMAT_1_FILES))
        for split, path in file_paths.items():
            file_paths[split] = os.path.join(directory, path)

        target_paths = dict(zip(splits, TARGET_FILES))
        for split, path in target_paths.items():
            target_paths[split] = os.path.join(directory, path)

        split_targets = {}
        for split in splits:
            try:
                targets = pd.read_csv(target_paths[split], index_col='name')
            except:
                targets = None
            split_targets[split] = targets

        def get_target(image_path, split):
            try:
                root_basename = os.path.splitext(os.path.basename(image_path))[0]
                target = split_targets[split].loc[root_basename].target
                mask = 1
            except:
                target = 0
                mask = 0
            return target, mask


        # We first extract the data files in a temporary directory. While doing
        # that, we also count the number of examples for each split. Files are
        # extracted individually, which allows to display a progress bar. Since
        # the splits will be concatenated in the HDF5 file, we also compute the
        # start and stop intervals of each split within the concatenated array.
        checksums = set([])
        def extract_tar(split):
            num_examples = 0
            path = file_paths[split]
            if os.path.isfile(path):
                with tarfile.open(path, 'r:gz') as f:
                    members = f.getmembers()
                    progress_bar_context = progress_bar(
                        name='{} file'.format(split), maxval=len(members),
                        prefix='Extracting')
                    with progress_bar_context as bar:
                        for i, member in enumerate(members):
                            if ((member.name.endswith('.jpg') and not
                            os.path.basename(member.name).startswith('.'))):
                                f.extract(member, path=os.path.join(TMPDIR,split))
                                num_examples += 1
                            bar.update(i)
                DIR = TMPDIR
            elif os.path.isdir(path):
                print("reading DIRECTORY %s"%path)
                DIR = path
                for root, dirs, files in os.walk(path):
                    for file in files:
                        if file.endswith('.jpg') and not file.startswith('.'):
                            num_examples += 1
            else:
                print("No file or directory named %s"%path)
                return [], None
            print('#files=%d'%num_examples)

            jpgfiles = []
            progress_bar_context = progress_bar(
                name='{} file'.format(split), maxval=num_examples,
                prefix='Validating')
            num_examples = 0
            bad_examples = 0
            duplicate_examples = 0
            count = 0
            errors = 0
            shape = None  # all images must have the same shape
            with progress_bar_context as bar:
                for root, dirs, files in os.walk(os.path.join(DIR,split)):
                    for file in files:
                        if file.endswith('.jpg') and not file.startswith('.'):
                            image_path = os.path.join(root, file)
                            count += 1
                            try:
                                im = Image.open(image_path)
                                im = numpy.asarray(im)
                                m = hashlib.md5()
                                m.update(im)
                                h = m.hexdigest()

                                if shape is None:
                                    shape = im.shape
                                if im.shape != shape:
                                    bad_examples += 1
                                    os.remove(image_path)
                                elif h  in checksums:
                                    duplicate_examples += 1
                                    os.remove(image_path)
                                else:
                                    checksums.add(h)
                                    num_examples += 1
                                    jpgfiles.append(image_path)
                            except:
                                errors += 1
                            bar.update(count)
            print('count=%d bad=%d dup=%d good=%d errors=%d'%(
                count, bad_examples, duplicate_examples,
                num_examples, errors))

            if onlytarget:
                jpgfiles = filter(lambda x: get_target(image_path, split)[1],
                                  jpgfiles)
            return jpgfiles, shape

        examples_per_split = OrderedDict(
            [(split, extract_tar(split)) for split in splits])
        cumulative_num_examples = numpy.cumsum(
            [0] + list(map(lambda x: len(x[0]),examples_per_split.values())))
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
        def make_vlen_dataset(source, shape):
            dtype = source_dtypes[source]
            shape = (num_examples,)+shape
            print("creating %s %s %s"%(source,str(shape),str(dtype)))
            dataset = h5file.create_dataset(
                source, shape, dtype=dtype)
            # Tag fixed-length axis with its label
            dataset.dims[0].label = 'batch'
            for i, label in enumerate(source_axis_labels[source]):
                dataset.dims[i+1].label = label

        shapes = filter(None,map(lambda x: x[1],examples_per_split.values()))
        assert len(set(shapes)) == 1, "splits have different image size %s"%shapes
        print('Images shape %s'%str(shapes[0]))

        source_shape = {'features':shapes[0], 'targets':(target_dim,)}

        for source in sources:
            make_vlen_dataset(source, source_shape[source])

        # The final step is to fill the HDF5 file.
        def fill_split(split, bar=None):
            print(split)

            image_count = target_count = 0
            for image_number, image_path in enumerate(examples_per_split[split][0]):
                image = numpy.asarray(Image.open(image_path))
                index = image_number + split_intervals[split][0]

                h5file['features'][index] = image
                image_count += 1

                target, mask = get_target(image_path, split)

                if dotarget:
                    if onlytarget:
                        h5file['targets'][index] = numpy.array([target,])
                    else:
                        h5file['targets'][index] = numpy.array([target,mask])
                target_count += mask

                if image_number % 1000 == 0:
                    h5file.flush()
                if bar:
                    bar.update(index)
            print('# targets %d out of %d'%(target_count, image_count))
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
    subparser.add_argument('--target', action='store_true',
                    help='add a targets source based on train/test.target.csv')
    subparser.add_argument('--onlytarget', action='store_true',
                    help='add a targets source based on train/test.target.csv.'
                         ' filter out images that dont have a target')
    return convert_jpgtgz
