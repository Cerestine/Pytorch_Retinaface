"""RetinaFace detector"""
import numpy as np
import torch
from torch.backends import cudnn
from layers.functions.prior_box import PriorBox
from utils.image_process import ImageStream, ImageProcess
from utils.box_utils import decode, decode_landm
from utils.nms.py_cpu_nms import py_cpu_nms

class RetinaFaceDetector():
    """RetinaFace detector class"""
    def __init__(self, net, cfg, test_length=-1):
        self.resize = 1
        self.test_length = test_length
        self.cfg = cfg
        self.device = torch.device("cpu" if self.cfg["cpu"] == "True" else "cuda")
        self.net = net.to(self.device)

    def _preprocess(self, image):
        # Get image scale
        im_height, im_width, _ = image.shape
        scale = torch.Tensor([im_width, im_height, im_width, im_height])
        scale = scale.to(self.device)

        # Scale image
        img = np.float32(image)
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        return img, im_height, im_width, scale

    def _postprocess(self, img, im_height, im_width, scale, loc, conf, landms):
        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale / self.resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / self.resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.cfg["confidence_threshold"])[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.cfg["top_k"]]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]
        return boxes, landms, scores

    def _NMS(self, boxes, landms, scores):
        """NMS implementation"""
        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.cfg["nms_threshold"])
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:self.cfg["keep_top_k"], :]
        landms = landms[:self.cfg["keep_top_k"], :]

        dets = np.concatenate((dets, landms), axis=1)
        return dets

    def detect(self, source, is_stream=False):
        """Actual detection runner"""
        image_processor = ImageProcess()
        image_generator = ImageStream(source, is_stream)
        cudnn.benchmark = True
        while self.test_length:
            # Get image from generator
            image = image_generator.get_image()

            # Preprocess image
            input_img, im_height, im_width, scale = self._preprocess(image)

            # Inference
            loc, conf, landms = self.net(input_img)

            # Postprocess
            boxes, landms, scores = self._postprocess(input_img, im_height, im_width, scale, loc, conf, landms)

            # NMS
            dets = self._NMS(boxes, landms, scores)

            # Visualization
            for b in dets:
                if b[4] < self.cfg["vis_thres"]:
                    continue
                b = list(map(int, b))
                result_image = image_processor.process_image(image, b)
            image_processor.visualize_image(result_image, self.cfg["save_folder"], self.cfg["save_image"])

    def export_model(self, output_name="FaceDetector"):
        """Export model to onnx format"""
        output_onnx = "{}.onnx".format(output_name)
        print("==> Exporting model to ONNX format at '{}'".format(output_onnx))
        input_names = ["input0"]
        output_names = ["output0"]
        inputs = torch.randn(1, 3, self.cfg["long_side"], self.cfg["long_side"]).to(self.device)

        torch_out = torch.onnx._export(self.net, inputs, output_onnx,
                                       export_params=True, verbose=False,
                                       input_names=input_names, output_names=output_names)
        return torch_out
