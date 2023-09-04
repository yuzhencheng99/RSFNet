import math

import numpy as np

from lib.models.rsfnet import build_RSF
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond


class RSFNet(BaseTracker):
    def __init__(self, params, dataset_name):
        super(RSFNet, self).__init__(params)
        network = build_RSF(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}

    def initialize(self, image,image_i, o1,o1_i,o2,o2_i,info,info1,info2):
        # forward the template once

        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        z_patch_arr_i, resize_factor, z_amask_arr = sample_target(image_i, info['init_bbox'], self.params.template_factor,
                                                                output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        self.z_patch_arr_i = z_patch_arr_i

        template,template_i = self.preprocessor.process(z_patch_arr,z_patch_arr_i, z_amask_arr)
        o1_patch_arr, _,o1_amask_arr = sample_target(o1, info1['curr_bbox'], self.params.template_factor,
                                                                output_sz=self.params.template_size)
        o1_patch_arr_i,_,_ = sample_target(o1_i, info1['curr_bbox'],
                                                                  self.params.template_factor,
                                                                  output_sz=self.params.template_size)


        online1,online1_i = self.preprocessor.process(o1_patch_arr, o1_patch_arr_i, o1_amask_arr)
        o2_patch_arr, _, o2_amask_arr = sample_target(o2, info2['curr_bbox'], self.params.template_factor,
                                                      output_sz=self.params.template_size)
        o2_patch_arr_i, _, _ = sample_target(o2_i, info2['curr_bbox'],
                                             self.params.template_factor,
                                             output_sz=self.params.template_size)

        online2, online2_i = self.preprocessor.process(o2_patch_arr, o2_patch_arr_i, o2_amask_arr)
        with torch.no_grad():
            self.z_dict1 = template
            self.z_dict1_i = template_i
            self.o1,self.o1_i,self.o2,self.o2_i=online1,online1_i,online2,online2_i
        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1)
            template_bbox_i = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template_i.tensors.device).squeeze(1)
            online1_bbox = self.transform_bbox_to_crop(info1['curr_bbox'], resize_factor,
                                                        online1.tensors.device).squeeze(1)
            online1_bbox_i = self.transform_bbox_to_crop(info1['curr_bbox'], resize_factor,
                                                          online1_i.tensors.device).squeeze(1)
            online2_bbox = self.transform_bbox_to_crop(info2['curr_bbox'], resize_factor,
                                                      online2.tensors.device).squeeze(1)
            online2_bbox_i = self.transform_bbox_to_crop(info2['curr_bbox'], resize_factor,
                                                        online2_i.tensors.device).squeeze(1)
            mz1 = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)
            mz1_i = generate_mask_cond(self.cfg, 1, template_i.tensors.device, template_bbox_i)
            mz2 = generate_mask_cond(self.cfg, 1, template.tensors.device, online1_bbox)
            mz2_i = generate_mask_cond(self.cfg, 1, template_i.tensors.device, online1_bbox_i)
            mz3 = generate_mask_cond(self.cfg, 1, template.tensors.device, online2_bbox)
            mz3_i = generate_mask_cond(self.cfg, 1, template_i.tensors.device, online2_bbox_i)

            # self.box_mask_z_i = generate_mask_cond(self.cfg, 1, template_i.tensors.device, template_bbox_i)
            self.box_mask_z=torch.concat((mz1,mz2,mz3,mz1_i,mz2_i,mz3_i),dim=1)
            # self.box_mask_z_i
        # save states
        self.state = info2['curr_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, image_i,info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        x_patch_arr_i, _,_= sample_target(image_i, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search,search_i = self.preprocessor.process(x_patch_arr, x_patch_arr_i,x_amask_arr)

        with torch.no_grad():
            x = search.tensors
            x_i = search_i.tensors

            # merge the template and the search
            # run the transformer
            temp,temp1,temp2=self.z_dict1.tensors,self.o1.tensors,self.o2.tensors
            temp_i,temp1_i,temp2_i=self.z_dict1_i.tensors,self.o1_i.tensors,self.o2_i.tensors
            # out_dict = self.network.forward(
            #     template=self.z_dict1.tensors, search=x_dict.tensors, ce_template_mask=self.box_mask_z)
            out_dict = self.network.forward(x, temp, temp1,temp2,x_i, temp_i,
                              temp1_i, temp2_i, ce_template_mask=self.box_mask_z)

        # add hann windows
        pred_score_map = out_dict['score_map']
        pop=torch.max(pred_score_map).item()
        response = self.output_window * pred_score_map
        pop1=response.cpu().numpy()[0][0]

        pred_boxes = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'],return_score=True)
        pred_boxes = pred_boxes[0].view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes[0] * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')

                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(x_patch_arr_i).permute(2, 0, 1), 'image', 1, 'search_region_ir')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(torch.from_numpy(self.z_patch_arr_i).permute(2, 0, 1), 'image', 1, 'template_i')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    removed_indexes_s_rgb=[]
                    removed_indexes_s_ir=[]
                    for stage_index in removed_indexes_s:
                        mask=stage_index>255
                        removed_indexes_s_rgb.append(np.expand_dims(stage_index[~mask],axis=0))
                        removed_indexes_s_ir.append([i-256 for i in np.expand_dims(stage_index[mask],axis=0)])



                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s_rgb)
                    masked_search_i = gen_visualization(x_patch_arr_i, removed_indexes_s_ir)

                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')
                    self.visdom.register(torch.from_numpy(masked_search_i).permute(2, 0, 1), 'image', 1, 'masked_search_i')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state,'confidence':pop}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return RSFNet
