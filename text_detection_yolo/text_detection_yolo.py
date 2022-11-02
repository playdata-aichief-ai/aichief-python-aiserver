from text_detection_yolo import detect
import os
from ai.settings.settings import BASE_DIR

# model save path =os.path.join(BASE_DIR, 'text_detection_yolo', 'models', 'best_yolo5x.pt')

class Text_Detection_Yolo():
    def predict(img_dic):
        return detect.run(
                weights=os.path.join(BASE_DIR, 'text_detection_yolo', 'models', 'best_yolo5x.pt'),  # model path or triton URL
                source=None,  # file/dir/URL/glob/screen/0(webcam)
                img_dic=img_dic,
                data=os.path.join(BASE_DIR,'data/coco128.yaml'),  # dataset.yaml path
                imgsz=(640, 640),  # inference size (height, width)
                conf_thres=0.25,  # confidence threshold
                iou_thres=0.1,  # NMS IOU threshold
                max_det=10,  # maximum detections per image
                device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                view_img=False,  # show results
                save_txt=False,  # save results to *.txt
                save_conf=False,  # save confidences in --save-txt labels
                save_crop=False,  # save cropped prediction boxes
                nosave=True,  # do not save images/videos
                classes=None,  # filter by class: --class 0, or --class 0 2 3
                agnostic_nms=False,  # class-agnostic NMS
                augment=False,  # augmented inference
                visualize=False,  # visualize features
                update=False,  # update all models
                project=os.path.join(BASE_DIR / 'runs/detect'),  # save results to project/name
                name='exp',  # save results to project/name
                exist_ok=False,  # existing project/name ok, do not increment
                line_thickness=3,  # bounding box thickness (pixels)
                hide_labels=True,  # hide labels
                hide_conf=True,  # hide confidences
                half=False,  # use FP16 half-precision inference
                dnn=False,  # use OpenCV DNN for ONNX inference
                vid_stride=1,  # video frame-rate stride
                )