
import importlib
import os
import re
import numpy as np
import logging
from pydub import AudioSegment
import pandas as pd
import SimpleITK as sitk



def load_itk(filename):
    '''
    This funciton reads a '.mhd' file using SimpleITK and return the image array, origin and spacing of the image.
    '''
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)
    ct_scan = sitk.GetArrayFromImage(itkimage)
    origin = np.array(list(reversed(itkimage.GetOrigin())))
    spacing = np.array(list(reversed(itkimage.GetSpacing())))
    return ct_scan, origin, spacing


def get_dir_list(data_path, full_path=True):
    dir_list = np.array([], dtype=object)
    for f in os.listdir(data_path):
        folder_path = os.path.join(data_path, f)
        if os.path.isdir(folder_path):
            if full_path:
                dir_list = np.append(dir_list, folder_path)
            else:
                dir_list = np.append(dir_list, os.path.split(folder_path)[1])
    return list(dir_list)


def save_aLL_files_name(path, name='file_names', keyword=[], filtering_mode='in', is_fullpath=True, shuffle=True, save_path=None):
    # file_names = get_file_names(path, keyword, filtering_mode, is_fullpath, shuffle)
    file_names = get_files(path, keys=keyword, return_fullpath=True, sort=True)
    if not save_path: save_path = path
    save_content_in_txt(
        file_names, os.path.join(save_path, f'{name}.txt'), filter_bank=[], access_mode='w+', dir=None)
    # with open(os.path.join(save_path, f'{name}.txt'), 'w+') as fw:
    #     for f in file_names:
    #         fw.write(f)    
    #         fw.write('\n')


def load_audio_waveform(filename, audio_format, sr=None, channels=None):
    """Pydub based audio waveform loading function"""
    y = AudioSegment.from_file(filename, audio_format)
    if sr: y = y.set_frame_rate(sr)
    if channels: y = y.set_channels(channels)
    return y    


def get_files(path, keys=[], return_fullpath=True, sort=True, sorting_key=None, recursive=True, get_dirs=False):
    """Get all the file name under the given path with assigned keys
    Args:
        path: (str)
        keys: (list, str)
        return_fullpath: (bool)
        sort: (bool)
        sorting_key: (func)
        recursive: The flag for searching path recursively or not(bool)
    Return:
        file_list: (list)
    """
    file_list = []
    assert isinstance(keys, (list, str))
    if isinstance(keys, str): keys = [keys]
    # Rmove repeated keys
    keys = list(set(keys))

    def push_back_filelist(root, f, file_list, is_fullpath):
        if is_fullpath:
            file_list.append(os.path.join(root, f))
        else:
            file_list.append(f)

    for i, (root, dirs, files) in enumerate(os.walk(path)):
        # print(root, dirs, files)
        if not recursive:
            if i > 0: break

        if get_dirs:
            files = dirs
            
        for j, f in enumerate(files):
            if keys:
                for key in keys:
                    if key in f:
                        push_back_filelist(root, f, file_list, return_fullpath)
            else:
                push_back_filelist(root, f, file_list, return_fullpath)

    if file_list:
        if sort: file_list.sort(key=sorting_key)
    else:
        f = 'dir' if get_dirs else 'file'
        if keys: 
            logging.warning(f'No {f} exist with key {keys}.') 
        else: 
            logging.warning(f'No {f} exist.') 
    return file_list

    
def get_class(class_name, modules):
    for module in modules:
        m = importlib.import_module(module)
        clazz = getattr(m, class_name, None)
        if clazz:
            return clazz
    raise RuntimeError(f'Unsupported dataset class: {class_name}')


def load_content_from_txt(path, access_mode='r'):
    with open(path, access_mode) as fw:
        content = fw.read().splitlines()
    return content


def get_data_path(data_path, index_root, data_split, keywords=[]):
    # TODO: save in csv
    # TODO: save with label
    split_code = inspect_data_split(data_split)
    dataset_name = os.path.split(data_path)[1]
    
    # Check if an index already exists, create one if not.
    index_path = os.path.join(index_root, dataset_name)
    if not os.path.isdir(index_path):
        os.mkdir(index_path)
        file_path_list = get_files(data_path, keys=keywords, return_fullpath=True, sort=True)
        train_split = int(data_split.get('train', 0)*len(file_path_list))
        valid_split = int(data_split.get('valid', 0)*len(file_path_list))
        test_split = int(data_split.get('test', 0)*len(file_path_list))

        train_path_list = file_path_list[:train_split]
        valid_path_list = file_path_list[train_split:train_split+valid_split]
        test_path_list = file_path_list[train_split+valid_split:train_split+valid_split+test_split]

        train_path = os.path.join(index_path, f'{dataset_name}_train_{split_code}.txt')
        valid_path = os.path.join(index_path, f'{dataset_name}_valid_{split_code}.txt')
        test_path = os.path.join(index_path, f'{dataset_name}_test_{split_code}.txt')

        save_content_in_txt(train_path_list, train_path)
        save_content_in_txt(valid_path_list, valid_path)
        save_content_in_txt(test_path_list, test_path)

        data_path_dict = {
            'train': train_path,
            'valid': valid_path,
            'test': test_path}
    else:
        file_path_list = get_files(index_root, return_fullpath=True, sort=True)
        data_path_dict
    return data_path_dict


def get_data_indices(data_name, data_path, save_path, data_split, mode, generate_index_func):
    """"Get dataset indices and create if not exist"""
    # # TODO: gt not exist condition
    # create index folder and sub-folder if not exist
    os.chdir(save_path)
    index_dir_name = f'{data_name}_data_index'
    sub_index_dir_name = f'{data_name}_{data_split[0]}_{data_split[1]}'
    input_data_path = os.path.join(save_path, index_dir_name, sub_index_dir_name)
    if not os.path.isdir(input_data_path): os.makedirs(input_data_path)
    
    # generate index list and save in txt file
    generate_index_func(data_path, data_split, input_data_path)
        
    # load index list from saved txt
    os.chdir(input_data_path)
    if os.path.isfile(f'{mode}.txt'):
        input_data_indices = load_content_from_txt(f'{mode}.txt')
        input_data_indices.sort()
    else:
        input_data_indices = None

    if os.path.isfile(f'{mode}_gt.txt'):
        ground_truth_indices = load_content_from_txt(f'{mode}_gt.txt')
        ground_truth_indices.sort()
    else:
        ground_truth_indices = None
    return input_data_indices, ground_truth_indices


def save_input_and_label_index(data_path, save_path, data_split, data_keys=None, loading_format=None):
    # TODO: test input output
    # assert 'input' in data_keys, 'Undefined input data key'
    # assert 'ground_truth' in data_keys, 'Undefined ground truth data key'
    class_name = os.listdir(data_path)
    os.chdir(save_path)
    include_or_exclude, keys = [], []
    if data_keys:
        for v in data_keys.values():
            if v:
                include_or_exclude.append(v.split('_')[0])
                keys.append(v.split('_')[1]) 
    data_dict = generate_filenames(
        data_path, keys=keys, include_or_exclude=include_or_exclude, is_fullpath=True, loading_formats=loading_format)

    def save_content_ops(data, train_name, valid_name):
        # TODO: Is this a solid solution?
        data.sort(key=len)
        split = int(len(data)*data_split[0])
        train_input_data, val_input_data = data[:split], data[split:]
        save_content_in_txt(train_input_data,  train_name, filter_bank=class_name, access_mode="w+", dir=save_path)
        save_content_in_txt(val_input_data, valid_name, filter_bank=class_name, access_mode="w+", dir=save_path)


    # if data_keys['input']:    
    #     input_data = data_dict[data_keys['input']]
    # else:
    #     input_data = data_dict
    
    if data_keys:
        if 'input' in data_keys:
            if data_keys['input']:
                input_data = data_dict[data_keys['input']]
            else:
                input_data = data_dict
            save_content_ops(input_data, 'train.txt', 'valid.txt')
        if 'ground_truth' in data_keys:
            if data_keys['ground_truth']:
                ground_truth = data_dict[data_keys['ground_truth']]
            else:
                ground_truth = data_dict
            save_content_ops(ground_truth, 'train_gt.txt', 'valid_gt.txt')
    else:
        input_data = data_dict
        save_content_ops(input_data, 'train.txt', 'valid.txt')
        
    # input_data.sort()
    # split = int(len(input_data)*data_split[0])
    # train_input_data, val_input_data = input_data[:split], input_data[:split]
    # save_content_in_txt(train_input_data, 'train.txt', filter_bank=class_name, access_mode="w+", dir=save_path)
    # save_content_in_txt(val_input_data, 'valid.txt', filter_bank=class_name, access_mode="w+", dir=save_path)
    # if 'ground_truth' in data_keys:
    #     ground_truth = data_dict[data_keys['ground_truth']]
    #     ground_truth.sort()
    #     train_ground_truth, valid_ground_truth = ground_truth[split:], ground_truth[split:]
    #     save_content_in_txt(train_ground_truth, 'train_gt.txt', filter_bank=class_name, access_mode="w+", dir=save_path)
    #     save_content_in_txt(valid_ground_truth, 'valid_gt.txt', filter_bank=class_name, access_mode="w+", dir=save_path)


# def string_filtering(s, filter):
#     filtered_s = {}
#     for f in filter:
#         if f in s:
#             filtered_s[f] = s
#     if len(filtered_s) > 0:
#         return filtered_s
#     else:
#         return None

# TODO: mkdir?
def save_content_in_txt(content, path, filter_bank=None, access_mode='a+', dir=None):
    # assert isinstance(content, (str, list, tuple, dict))
    # TODO: overwrite warning
    with open(path, access_mode) as fw:
        # def string_ops(s, dir, filter):
        #     pair = string_filtering(s, filter)
        #     return os.path.join(dir, list(pair.keys())[0], list(pair.values())[0])

        if isinstance(content, str):
            # if dir:
            #     content = string_ops(content, dir, filter=filter_bank)
            #     # content = os.path.join(dir, content)
            fw.write(content)
        else:
            for c in content:
                # if dir:
                #     c = string_ops(c, dir, filter=filter_bank)
                #     # c = os.path.join(dir, c)
                fw.write(f'{c}\n')


def get_path_generator_in_case_order(data_path, return_fullpath, load_format=[]):
    dir_list = get_dir_list(data_path)
    for d in dir_list:
        file_list = get_files(d, keys=load_format, return_fullpath=return_fullpath)
        for file_idx, f in enumerate(file_list):
            yield (file_idx, f)


def save_data_label_pair_in_csv(data_path, save_path=None, save_name=None, load_format='wav', return_fullpath=True):
    path_loader = get_path_generator_in_case_order(data_path, return_fullpath, load_format=load_format)
    nums, ids, labels = [], [], []
    for idx, file_and_idx in enumerate(path_loader, 1):
        file_idx, f = file_and_idx
        file_path, file_name = os.path.split(f)
        label = int(file_path[-1])
        file_name = file_name.split('.')[0]
        print(idx, file_name, label)
        nums.append(file_idx)
        ids.append(file_name)
        labels.append(label)
        
    pair_dict = {'case_index': nums,
                 'id': ids,
                 'label': labels}
    pair_df = pd.DataFrame(pair_dict)
    if not save_name:
       save_name = 'train.csv' 
    if save_path is not None:
        pair_df.to_csv(os.path.join(save_path, save_name))
    else:
        pair_df.to_csv(save_name)


if __name__ == "__main__":
    pass