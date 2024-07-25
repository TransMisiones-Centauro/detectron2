import rospy
from sensor_msgs.msg import Image, CompressedImage

from cv_bridge import CvBridge, CvBridgeError # Package to convert between ROS and OpenCV Images
import cv2 # OpenCV library

import torch, detectron2
from detectron2.utils.logger import setup_logger

import numpy as np
import sys, os, json, cv2, random

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode

def callback(ros_data):
  global predictor
  global sps_metadata
  global br
  global image_pub
  
  np_arr = np.frombuffer(ros_data.data, np.uint8)
  im = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
  outputs = predictor(im)
  v = Visualizer(im[:, :, ::-1],
                metadata=sps_metadata,
                instance_mode=ColorMode.IMAGE_BW
  )
  out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
  frame = out.get_image()[:, :, ::-1]
  msg = CompressedImage()
  msg.header.stamp = ros_data.header.stamp
  msg.format = "png"
  msg.data = np.array(cv2.imencode('.png', frame)[1]).tobytes()
  image_pub.publish(msg)
  
def publish_image():
  global br
  global image_pub
  
  image_pub = rospy.Publisher('labeled_image', CompressedImage, queue_size=10)
  subscriber = rospy.Subscriber("/camera/image/compressed", CompressedImage, callback,  queue_size = 1)
  
  br = CvBridge()

  try:
    rospy.spin()
  except KeyboardInterrupt:
    pass
  
def main(args):

  global predictor
  global sps_metadata

  setup_logger()
  register_coco_instances("sps_dataset_train", {}, 
                          "/home/ecervera/Escritorio/CENTAURO/detectron2/jupyter-demo/annotations_contours/instances_Subset01.json", 
                          "/home/ecervera/Escritorio/CENTAURO/detectron2/jupyter-demo/image_SPS_1720795090")
  
  sps_metadata = MetadataCatalog.get("sps_dataset_train")
  sps_dataset = DatasetCatalog.get("sps_dataset_train")

  cfg = get_cfg()
  cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
  cfg.DATASETS.TRAIN = ("sps_dataset_train",)
  cfg.DATASETS.TEST = ()
  cfg.DATALOADER.NUM_WORKERS = 2
  cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
  cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonl
  cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
  cfg.SOLVER.MAX_ITER = 300	# 300 iterations seems good enough 
  cfg.SOLVER.STEPS = []    	# do not decay learning rate
  cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
  os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
  trainer = DefaultTrainer(cfg)
  trainer.resume_or_load(resume=False)
  trainer.train()

  cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
  predictor = DefaultPredictor(cfg)

  rospy.init_node("sps_recognition", anonymous=True)

  publish_image()
  
if __name__ == '__main__':
  main(sys.argv)