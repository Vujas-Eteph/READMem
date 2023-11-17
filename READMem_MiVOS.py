import os
import time
import json
import tqdm
import numpy as np
from icecream import ic
from pathlib import Path
from argparse import ArgumentParser
from PIL import Image

# Deep Learning imports
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


# From MiVOS:
from MiVOS.model.propagation.prop_net import PropagationNetwork
from inference_READMem_MIVOS import InferenceCore

# FROM XMEM:
from inference.data.test_datasets import LongTestDataset, DAVISTestDataset, YouTubeVOSTestDataset, MOSETestDataset
from inference.data.mask_mapper import MaskMapper

# FROM READMEM:
from READMem_API.READMem_debugging.ST_LT_memory_plot.Plot_ST_and_LT_Memory import gram_det_plot, ST_LT_plotting_functions


try:
    import hickle as hkl
except ImportError:
    print('Failed to import hickle. Fine if not using multi-scale testing.')

"""
Arguments loading
"""
parser = ArgumentParser()
parser.add_argument('--model', default='MiVOS/saves/propagation_model.pth')          # TODO Change
parser.add_argument('--d16_path', default='../DAVIS/2016')
parser.add_argument('--d17_path', default='../DAVIS/2017')
parser.add_argument('--y18_path', default='../YouTube2018')
parser.add_argument('--y19_path', default='../YouTube')
# parser.add_argument('--lv_path', default='../long_video_set')
parser.add_argument('--lv_path', default='../long_video_set')
parser.add_argument('--mose_path', default='../MOSE/MOSE_release')
# parser.add_argument('--output', default='output_QDMN/')
# parser.add_argument('--output', default='output/')
parser.add_argument('--mem_confi', help='path 2 yaml file for configuring the memory', type=str, default='memory_configuration.yaml')
parser.add_argument('--output', default='give_me_a_name/')
parser.add_argument('--no_top', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--record_det', action='store_true')
parser.add_argument('--flip', action='store_true')
parser.add_argument('--dataset', help='D16/D17/Y18/Y19/LV1/LV3/G', default='D17')
parser.add_argument('--split', help='val/test', default='val')
parser.add_argument('--size', default=480, type=int,
            help='Resize the shorter side to this size. -1 to use original resolution. ')
parser.add_argument('--save_all', action='store_true',
            help='Save all frames. Useful only in YouTubeVOS/long-time video')
parser.add_argument('--save_scores', action='store_true', help='Also save the the logits and softmax outputs')
parser.add_argument('--silence', action='store_true', help='Mutes the icecream commands')
parser.add_argument('--record_the_gramian', action='store_true', help='Sets a flag to know if you want to record the gramian or not')

args = parser.parse_args()

out_path = args.output
if args.output is None:
    args.output = f'../output/{args.dataset}_{args.split}'
    print(f'Output path not provided. Defaulting to {args.output}')

if args.silence:
    ic.disable()
else:
    ic.enable()

torch.autograd.set_grad_enabled(False)

"""
Data preparation
"""
is_youtube = args.dataset.startswith('Y')
is_davis = args.dataset.startswith('D')
is_lv = args.dataset.startswith('LV')
is_MOSE = args.dataset.startswith('MOSE')

if is_youtube:
    if args.dataset == 'Y18':
        yv_path = args.y18_path
    elif args.dataset == 'Y19':
        yv_path = args.y19_path

    if args.split == 'val':
        args.split = 'valid'
        meta_dataset = YouTubeVOSTestDataset(data_root=yv_path, split='valid', size=args.size)
    elif args.split == 'test':
        meta_dataset = YouTubeVOSTestDataset(data_root=yv_path, split='test', size=args.size)
    else:
        raise NotImplementedError

elif is_davis:
    if args.dataset == 'D16':
        if args.split == 'val':
            # Set up Dataset, a small hack to use the image set in the 2017 folder because the 2016 one is of a different format
            meta_dataset = DAVISTestDataset(args.d16_path, imset='../../2017/trainval/ImageSets/2016/val.txt', size=args.size)
        else:
            raise NotImplementedError
        palette = None
    elif args.dataset == 'D17':
        if args.split == 'val':
            meta_dataset = DAVISTestDataset(os.path.join(args.d17_path, 'trainval'), imset='2017/val.txt', size=args.size)
        elif args.split == 'test':
            meta_dataset = DAVISTestDataset(os.path.join(args.d17_path, 'test-dev'), imset='2017/test-dev.txt', size=args.size)
        else:
            raise NotImplementedError

elif is_lv:
    if args.dataset == 'LV1':
        meta_dataset = LongTestDataset(os.path.join(args.lv_path, 'long_video'))
    elif args.dataset == 'LV3':
        meta_dataset = LongTestDataset(os.path.join(args.lv_path, 'long_video_x3'))
    else:
        raise NotImplementedError
elif args.dataset == 'G':
    meta_dataset = LongTestDataset(os.path.join(args.generic_path), size=args.size)
    if not args.save_all:
        args.save_all = True
        print('save_all is forced to be true in generic evaluation mode.')

elif is_MOSE:
    if args.dataset == 'MOSE':
        if args.split == 'val':
            meta_dataset = MOSETestDataset(args.mose_path)
        if args.split == 'train':
            meta_dataset = MOSETestDataset(args.mose_path, split='train')

    else:
        raise NotImplementedError

else:
    raise NotImplementedError


# Set up loader
meta_loader = meta_dataset.get_datasets()

# Load checkpoint
prop_saved = torch.load(args.model)
top_k = None if args.no_top else 50
if False : #args.use_km:
    prop_model = PropagationNetwork(top_k=top_k).cuda().eval()
else:
    prop_model = PropagationNetwork(top_k=top_k).cuda().eval()
prop_model.load_state_dict(prop_saved)

total_process_time = 0
total_frames = 0

args.record_the_gramian = False

# Start eval
for vid_reader in tqdm.tqdm(meta_loader):
    loader = DataLoader(vid_reader, batch_size=1, shuffle=False, num_workers=2)
    vid_name = vid_reader.vid_name
    vid_length = len(loader)
    mapper = MaskMapper()
    first_mask_loaded = False
    need_to_initialize_READMem = True

    if args.record_the_gramian:
        record_det_list = []
        n_0, n_1 = args.output.split(args.dataset)
        det_save_file_name = f'{n_0}{args.dataset}/Gramian{n_1}'


    for ti, data in enumerate(loader):
        ic(f'frame_id:{ti}')
        with torch.cuda.amp.autocast(enabled=False):
            rgb = data['rgb'].cuda()[0]
            msk = data.get('mask')
            info = data['info']
            frame = info['frame'][0]
            shape = info['shape']
            need_resize = info['need_resize'][0]

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            """
            For timing see https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964
            Seems to be very similar in testing as my previous timing method 
            with two cuda sync + time.time() in STCN though 
            """
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            if not first_mask_loaded:
                if msk is not None:
                    first_mask_loaded = True
                else:
                    # no point to do anything without a mask
                    continue

            if args.flip:
                rgb = torch.flip(rgb, dims=[-1])
                msk = torch.flip(msk, dims=[-1]) if msk is not None else None

            # Map possibly non-continuous labels to continuous ones
            if msk is not None:
                msk, labels = mapper.convert_mask(msk[0].numpy())
                msk = torch.Tensor(msk).cuda()
                if need_resize:
                    msk = vid_reader.resize_mask(msk.unsqueeze(0))[0]
            else:
                labels = None

            # Run the model on this frame
            torch.cuda.synchronize()
            process_begin = time.time()

            if need_to_initialize_READMem:
                need_to_initialize_READMem = False
                k = len(labels)
                processor = InferenceCore(prop_model, k, args.mem_confi, args.debug, args.record_det)
                prob = processor.set_annotated_frame(ti, vid_length, rgb, msk)

                if args.debug: # For plotting results
                    show_Memory = ST_LT_plotting_functions(*processor.get_size_of_ST_N_LT_memory)
                    show_Memory.set_path_for_finding_the_images(vid_reader.image_dir)

            else:
                prob = processor.step(ti,rgb)

            # Upsample to original size if needed
            if need_resize:
                # ic(prob.squeeze(1).shape)
                prob = F.interpolate(prob, shape, mode='bilinear', align_corners=False)[:,0]

            if args.record_the_gramian:
                record_det_list.append(processor.return_lt_det)



            end.record()
            torch.cuda.synchronize()
            total_process_time += (start.elapsed_time(end)/1000)
            total_frames += 1

            # Probability mask -> index mask
            prob_u = prob.clone().detach().cpu().numpy()
            out_mask = torch.argmax(prob, dim=0)
            out_mask = (out_mask.detach().cpu().numpy()).astype(np.uint8)

            ic(out_mask.shape)

            # Save the mask
            if args.save_all or info['save'][0]:
                this_out_path = os.path.join(out_path, 'Annotations', vid_name)
                os.makedirs(this_out_path, exist_ok=True)
                out_mask = mapper.remap_index_mask(out_mask)

                if 3 == len(out_mask.shape):
                    out_img = Image.fromarray(out_mask[0])
                else:
                    out_img = Image.fromarray(out_mask)
                if vid_reader.get_palette() is not None:
                    out_img.putpalette(vid_reader.get_palette())
                out_img.save(os.path.join(this_out_path, frame[:-4]+'.png'))

            if args.save_scores:
                np_path = os.path.join(args.output, 'Scores', vid_name)
                os.makedirs(np_path, exist_ok=True)
                if ti==len(loader)-1:
                    hkl.dump(mapper.remappings, os.path.join(np_path, f'backward.hkl'), mode='w')
                if args.save_all or info['save'][0]:
                    hkl.dump(prob_u, os.path.join(np_path, f'{frame[:-4]}.hkl'), mode='w', compression='lzf')


        if args.debug: # For plotting results
            show_Memory.load_relevant_images(*processor.ST_N_LT_Memories)
            show_Memory.plt_ST_and_LT_memory()
            show_Memory.plt_save_Memory_plot('./Debugging_Results/memory', ti+int(frame[:-4])) # TODO start with correct idx
            record_det_list.append(processor.return_lt_det)
            gram_det_plot(f"{os.path.join('Debugging_Results', vid_name)}.png", record_det_list)


    if args.record_the_gramian:
        if not os.path.isdir(det_save_file_name):
            Path(det_save_file_name).mkdir(parents=True, exist_ok=True)

        with open(os.path.join(det_save_file_name,vid_name+'.json'), 'w') as f:
            json.dump({'Gramian':record_det_list}, f)

print('Total processing time: ', total_process_time)
print('Total processed frames: ', total_frames)
print(f'FPS: {total_frames / total_process_time}')
print(f'Max allocated memory (MB): {torch.cuda.max_memory_allocated() / (2**20)}')