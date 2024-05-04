import numpy as np
import torch
from PIL import Image
import cv2
import scipy
import copy

import grasp_det_seg.models as models
from grasp_det_seg.modules.fpn import FPN, FPNBody
from grasp_det_seg.algos.rpn import ProposalGenerator, AnchorMatcher, RPNLoss
from grasp_det_seg.algos.fpn import RPNAlgoFPN, DetectionAlgoFPN
from grasp_det_seg.modules.heads import RPNHead, FPNROIHead, FPNSemanticHeadDeeplab
from grasp_det_seg.algos.detection import (
    PredictionGenerator,
    ProposalMatcher,
    DetectionLoss,
)
from grasp_det_seg.algos.semantic_seg import SemanticSegLoss, SemanticSegAlgo
from grasp_det_seg.models.det_seg import DetSegNet

from grasp_det_seg.config import load_config
from grasp_det_seg.utils.misc import (
    config_to_string,
    norm_act_from_config,
    freeze_params,
)
from grasp_det_seg.data_OCID import OCIDTestTransform
from grasp_det_seg.utils.parallel import PackedSequence
from grasp_det_seg.data_OCID.OCID_class_dict import cls_list, colors_list
from grasp_det_seg.utils.snapshot import resume_from_snapshot


import logging


def log_debug(msg, *args):
    logging.getLogger().debug(msg, *args)


def log_info(msg, *args):
    logging.getLogger().info(msg, *args)


def Rotate2D(pts, cnt, ang):
    ang = np.deg2rad(ang)
    return (
        scipy.dot(
            pts - cnt,
            scipy.array(
                [[scipy.cos(ang), scipy.sin(ang)], [-scipy.sin(ang), scipy.cos(ang)]]
            ),
        )
        + cnt
    )


def make_config(config_path):
    log_debug("Loading configuration from %s", config_path)
    conf = load_config(config_path, config_path)
    log_debug("\n%s", config_to_string(conf))
    return conf


def make_model(config):
    body_config = config["body"]
    fpn_config = config["fpn"]
    rpn_config = config["rpn"]
    roi_config = config["roi"]
    sem_config = config["sem"]
    general_config = config["general"]
    classes = {
        "total": int(general_config["num_things"]) + int(general_config["num_stuff"]),
        "stuff": int(general_config["num_stuff"]),
        "thing": int(general_config["num_things"]),
        "semantic": int(general_config["num_semantic"]),
    }

    # BN + activation
    norm_act_static, norm_act_dynamic = norm_act_from_config(body_config)

    # Create backbone
    log_debug("Creating backbone model %s", body_config["body"])
    # body_fn = models.__dict__["net_" + body_config["body"]]
    body_fn = models.__dict__["net_resnet101"]
    body_params = (
        body_config.getstruct("body_params") if body_config.get("body_params") else {}
    )
    body = body_fn(norm_act=norm_act_static, **body_params)
    # if body_config.get("weights"):
    #     body_config["weights"] = "/app/data/weights/resnet101"
    #     body.load_state_dict(torch.load(body_config["weights"], map_location="cpu"))

    weights_path = "/app/data/weights/resnet101"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    body.load_state_dict(torch.load(weights_path, map_location=device))

    # Freeze parameters
    for n, m in body.named_modules():
        for mod_id in range(1, body_config.getint("num_frozen") + 1):
            if ("mod%d" % mod_id) in n:
                freeze_params(m)

    body_channels = body_config.getstruct("out_channels")

    # Create FPN
    fpn_inputs = fpn_config.getstruct("inputs")
    fpn = FPN(
        [body_channels[inp] for inp in fpn_inputs],
        fpn_config.getint("out_channels"),
        fpn_config.getint("extra_scales"),
        norm_act_static,
        fpn_config["interpolation"],
    )
    body = FPNBody(body, fpn, fpn_inputs)

    # Create RPN
    proposal_generator = ProposalGenerator(
        rpn_config.getfloat("nms_threshold"),
        rpn_config.getint("num_pre_nms_train"),
        rpn_config.getint("num_post_nms_train"),
        rpn_config.getint("num_pre_nms_val"),
        rpn_config.getint("num_post_nms_val"),
        rpn_config.getint("min_size"),
    )
    anchor_matcher = AnchorMatcher(
        rpn_config.getint("num_samples"),
        rpn_config.getfloat("pos_ratio"),
        rpn_config.getfloat("pos_threshold"),
        rpn_config.getfloat("neg_threshold"),
        rpn_config.getfloat("void_threshold"),
    )
    rpn_loss = RPNLoss(rpn_config.getfloat("sigma"))
    rpn_algo = RPNAlgoFPN(
        proposal_generator,
        anchor_matcher,
        rpn_loss,
        rpn_config.getint("anchor_scale"),
        rpn_config.getstruct("anchor_ratios"),
        fpn_config.getstruct("out_strides"),
        rpn_config.getint("fpn_min_level"),
        rpn_config.getint("fpn_levels"),
    )
    rpn_head = RPNHead(
        fpn_config.getint("out_channels"),
        len(rpn_config.getstruct("anchor_ratios")),
        1,
        rpn_config.getint("hidden_channels"),
        norm_act_dynamic,
    )

    # Create detection network
    prediction_generator = PredictionGenerator(
        roi_config.getfloat("nms_threshold"),
        roi_config.getfloat("score_threshold"),
        roi_config.getint("max_predictions"),
    )
    proposal_matcher = ProposalMatcher(
        classes,
        roi_config.getint("num_samples"),
        roi_config.getfloat("pos_ratio"),
        roi_config.getfloat("pos_threshold"),
        roi_config.getfloat("neg_threshold_hi"),
        roi_config.getfloat("neg_threshold_lo"),
        roi_config.getfloat("void_threshold"),
    )
    roi_loss = DetectionLoss(roi_config.getfloat("sigma"))
    roi_size = roi_config.getstruct("roi_size")
    roi_algo = DetectionAlgoFPN(
        prediction_generator,
        proposal_matcher,
        roi_loss,
        classes,
        roi_config.getstruct("bbx_reg_weights"),
        roi_config.getint("fpn_canonical_scale"),
        roi_config.getint("fpn_canonical_level"),
        roi_size,
        roi_config.getint("fpn_min_level"),
        roi_config.getint("fpn_levels"),
    )
    roi_head = FPNROIHead(
        fpn_config.getint("out_channels"), classes, roi_size, norm_act=norm_act_dynamic
    )

    # Create semantic segmentation network
    sem_loss = SemanticSegLoss(ohem=sem_config.getfloat("ohem"))
    sem_algo = SemanticSegAlgo(sem_loss, classes["semantic"])
    sem_head = FPNSemanticHeadDeeplab(
        fpn_config.getint("out_channels"),
        sem_config.getint("fpn_min_level"),
        sem_config.getint("fpn_levels"),
        classes["semantic"],
        pooling_size=sem_config.getstruct("pooling_size"),
        norm_act=norm_act_static,
    )

    # Create final network
    return DetSegNet(
        body, rpn_head, roi_head, sem_head, rpn_algo, roi_algo, sem_algo, classes
    )


def test(model, img, visualize=True, **varargs):
    model.eval()

    shortest_size = 480
    longest_max_size = 640
    rgb_mean = (0.485, 0.456, 0.406)
    rgb_std = (0.229, 0.224, 0.225)
    preprocess = OCIDTestTransform(
        shortest_size=shortest_size,
        longest_max_size=longest_max_size,
        rgb_mean=rgb_mean,
        rgb_std=rgb_std,
    )
    img_tensor, im_size = preprocess(img)

    with torch.no_grad():
        # Extract data
        packed_img = PackedSequence(img_tensor)
        print(packed_img[0].shape)
        # exit()

        # Run network
        _, pred, conf = model(img=packed_img, do_loss=False, do_prediction=True)

        # Update meters
        res = output_pred(pred, img, im_size, visualize)

    return res


def output_pred(raw_pred, img, im_size_, visualize):
    # https://github.com/stefan-ainetter/grasp_det_seg_cnn/blob/main/grasp_det_seg/data_OCID/OCID_class_dict.py
    # ^ class_list and color_list

    output = []
    for i, (sem_pred, bbx_pred, cls_pred, obj_pred) in enumerate(
        zip(
            raw_pred["sem_pred"],
            raw_pred["bbx_pred"],
            raw_pred["cls_pred"],
            raw_pred["obj_pred"],
        )
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sem_pred = sem_pred.to(device)
        sem_pred = np.asarray(sem_pred.detach().cpu().numpy(), dtype=np.uint8)
        # print(f"sem_pred: {sem_pred.shape}")
        # print(f"bbx_pred: {bbx_pred.shape}")
        # print(f"cls_pred: {cls_pred.shape}")
        # print(f"obj_pred: {obj_pred.shape}")

        seg_mask_vis = np.zeros((im_size_[0], im_size_[1], 3))
        cls_labels = np.unique(sem_pred)
        for cls in cls_labels:
            seg_mask_vis[sem_pred == cls] = colors_list[cls]
            mask_per_label = np.zeros_like(sem_pred)
            mask_per_label[sem_pred == cls] = 1
            iou_seg = np.sum(mask_per_label)
            if iou_seg < 100:
                continue

            # cv2.imshow(f"Mask {cls_list[cls]}", mask_per_label.astype(np.uint8)*255)
            # cv2.waitKey(0)

            print(f"{cls_list[cls]} {sum(map(sum,mask_per_label))}")

            # mask_per_label = mask_per_label.astype(np.uint8) * 255
        try:
            img_mask = img * 0.25 + seg_mask_vis * 0.75
        except ValueError as e:
            log_debug(f"Error: {e}")
            img_mask = seg_mask_vis
        img_mask = img_mask.astype(np.uint8) * 255

        for cls in cls_labels:
            if cls == 0:
                continue
            best_confidence = 0
            bbox_best = None
            r_bbox_best = None

            print(f"Getting best for cls: {cls} {cls_list[cls]}")

            for bbx_pred_i, cls_pred_i, obj_pred_i in zip(bbx_pred, cls_pred, obj_pred):
                threshold = 0.06

                cnt = np.array(
                    [
                        (int(bbx_pred_i[0]) + int(bbx_pred_i[2])) / 2,
                        (int(bbx_pred_i[1]) + int(bbx_pred_i[3])) / 2,
                    ]
                )

                if (int(cnt[1]) >= im_size_[0]) or (int(cnt[0]) >= im_size_[1]):
                    continue

                actual_class = sem_pred[int(cnt[1]), int(cnt[0])]
                if actual_class != cls:
                    continue

                if obj_pred_i.item() > threshold:
                    # print(f"obj_pred_i: {obj_pred_i.item()}")
                    # print(f"cls_pred_i: {cls_pred_i} {cls_list[cls_pred_i.item()]}")
                    # print(f"bbx_pred_i: {bbx_pred_i}")

                    pt1 = (int(bbx_pred_i[0]), int(bbx_pred_i[1]))
                    pt2 = (int(bbx_pred_i[2]), int(bbx_pred_i[3]))
                    newcls = cls_pred_i.item()
                    if newcls > 17:
                        assert False

                    num_classes_theta = 18
                    # theta = ((180 / num_classes_theta) * newcls) + 5 # 5 degrees offset?
                    theta = (180 / num_classes_theta) * newcls
                    pts = np.array(
                        [
                            [pt1[0], pt1[1]],
                            [pt2[0], pt1[1]],
                            [pt2[0], pt2[1]],
                            [pt1[0], pt2[1]],
                        ]
                    )
                    cnt = np.array(
                        [
                            (int(bbx_pred_i[0]) + int(bbx_pred_i[2])) / 2,
                            (int(bbx_pred_i[1]) + int(bbx_pred_i[3])) / 2,
                        ]
                    )
                    r_bbox_ = Rotate2D(pts, cnt, 90 - theta)
                    r_bbox_ = r_bbox_.astype("int16")
                    # print(f"r_bbox_: {r_bbox_}")

                    # if (int(cnt[1]) >= im_size_[0]) or (int(cnt[0]) >= im_size_[1]):
                    #     continue

                    # filter out gripper - any result with the center in the bottom 100 pixels
                    # TODO: find a better solution
                    if cnt[1] > im_size_[0] - 100:
                        continue

                    # if sem_pred[int(cnt[1]), int(cnt[0])] == cls:

                    # print(f"Seg class: {cls_list[sem_pred[int(cnt[1]), int(cnt[0])]]}")

                    if obj_pred_i.item() >= best_confidence:
                        best_confidence = obj_pred_i.item()
                        bbox_best = bbx_pred_i
                        r_bbox_best = copy.deepcopy(r_bbox_)

            if bbox_best is not None:
                res = {
                    "cls": cls,
                    "obj": best_confidence,
                    "bbox": bbox_best,
                    "r_bbox": r_bbox_best,
                }
                cnt = np.array(
                    [
                        (int(bbox_best[0]) + int(bbox_best[2])) / 2,
                        (int(bbox_best[1]) + int(bbox_best[3])) / 2,
                    ]
                )
                print(
                    f"res {cls_list[cls]} | {cls_list[sem_pred[int(cnt[1]), int(cnt[0])]]}: {res}"
                )
                output.append(res)
                pt1 = (int(bbox_best[0]), int(bbox_best[1]))
                pt2 = (int(bbox_best[2]), int(bbox_best[3]))
                # cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
                cv2.putText(
                    img_mask,
                    cls_list[cls],
                    (int(bbox_best[0]), int(bbox_best[1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                if r_bbox_best is not None:
                    cv2.line(
                        img_mask,
                        tuple(r_bbox_best[0]),
                        tuple(r_bbox_best[1]),
                        (255, 0, 0),
                        2,
                    )
                    cv2.line(
                        img_mask,
                        tuple(r_bbox_best[1]),
                        tuple(r_bbox_best[2]),
                        (0, 0, 255),
                        2,
                    )
                    cv2.line(
                        img_mask,
                        tuple(r_bbox_best[2]),
                        tuple(r_bbox_best[3]),
                        (255, 0, 0),
                        2,
                    )
                    cv2.line(
                        img_mask,
                        tuple(r_bbox_best[3]),
                        tuple(r_bbox_best[0]),
                        (0, 0, 255),
                        2,
                    )

        # print(f"output: {output}")
        # img_mask = (img * 0.25 + seg_mask_vis * 0.75)
        # img_mask = img_mask.astype(np.uint8)*255
        if visualize:
            cv2.imshow("Image Mask", img_mask)
            cv2.waitKey(0)

        return output, img_mask


class GraspServer:
    def __init__(self):
        config_path = "/app/data/config/test.ini"
        print(f"Loading configuration from {config_path}")
        config = make_config(config_path)

        print("Creating model...")
        self.model = make_model(config)
        weights_path = "/app/data/weights/model_last.pth.tar"
        log_debug("Loading snapshot from %s", weights_path)
        resume_from_snapshot(
            self.model, weights_path, ["body", "rpn_head", "roi_head", "sem_head"]
        )
        self.visualize = False

    def detect(self, img):
        res, img = test(self.model, img, visualize=self.visualize)
        # Convert to JSON serializable format
        res_dict = []
        for r in res:
            res_dict.append(
                {
                    "cls": int(r["cls"]),
                    "cls_name": cls_list[int(r["cls"])],
                    "obj": r["obj"],
                    "bbox": r["bbox"].tolist(),
                    "r_bbox": r["r_bbox"].tolist(),
                }
            )
        return res_dict, Image.fromarray(img)

    def detect_from_path(self, img_path):
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return self.detect(img_rgb)

    def test_detect(self):
        return self.detect_from_path(
            "/app/data/OCID_grasp/ARID20/table/top/seq08/rgb/result_2018-08-21-14-44-31.png"
        )


if __name__ == "__main__":
    print("Testing Grasp_Det_Seg")
    config_path = "/app/data/config/test.ini"
    print(f"Loading configuration from {config_path}")
    config = make_config(config_path)

    print("Creating model...")
    model = make_model(config)
    weights_path = "/app/data/weights/model_last.pth.tar"
    log_debug("Loading snapshot from %s", weights_path)
    snapshot = resume_from_snapshot(
        model, weights_path, ["body", "rpn_head", "roi_head", "sem_head"]
    )
    # rank, world_size = distributed.get_rank(), distributed.get_world_size()
    # model = DistributedDataParallel(model, device_ids=None, output_device=None, find_unused_parameters=True)

    print("Loading image...")
    # img_path = "/app/data/OCID_grasp/ARID20/table/top/seq12/rgb/result_2018-08-21-16-53-16.png"
    # img_path = "/app/data/OCID_grasp/ARID20/table/top/seq04/rgb/result_2018-08-21-12-13-01.png"
    # img_path="/app/data/OCID_grasp/ARID20/table/top/seq08/rgb/result_2018-08-21-14-44-31.png"
    img_path = "/app/data/test.png"
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # cv2.imshow("Image", img_rgb)
    # cv2.waitKey(0)

    print("Testing model...")
    test(model, img_rgb)
