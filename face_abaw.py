#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import time
from pathlib import Path
import os
from queue import Queue
import random

import cv2
import numpy as np
import onnxruntime

# from trt_yunet import TrtYuNet
from centroidtracking import CentroidTracker, MyQueue
from helpers import *
from deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection

mean = np.array([0.485, 0.456, 0.406]) * 255.0
scale = 1 / 255.0
std = [0.229, 0.224, 0.225]
FACEDETECTOR=["YuNet-TRT", "YuNet-CV2", "Haar Cascade"]
RUNTIME = ["onnxruntime", "cv2.dnn"]
GENDER = ['Female', 'Male','Unknown']
WINDOW_NAME = "Facial Affective Behavior Analysis in-the-wild + CentroiTracking"


recognizer = cv2.FaceRecognizerSF.create(
    "models/yunet/face_recognition_sface_2021dec_int8.onnx",
    "",
    # backend_id=cv2.dnn.DNN_BACKEND_CUDA,
    # target_id=cv2.dnn.DNN_TARGET_CUDA
)

def extract_face_features(fe_session, face_align):
    input_blob = cv2.dnn.blobFromImage(
                    image=face_align,
                    scalefactor=scale,
                    size=(112,112),  # img target size
                    mean=mean,
                    swapRB=False,  # BGR -> RGB
                    crop=True  # center crop
                ) # (1,3,112,112)
    input_blob[0] /= np.asarray(std, dtype=np.float32).reshape(3, 1, 1) # (1,3,112,112)
              
    ort_inputs = {fe_session.get_inputs()[0].name: input_blob}
    face_features = fe_session.run(None, ort_inputs)[0]
    return np.squeeze(face_features).astype(np.float32) # (128,)


ct = CentroidTracker(maxDisappeared=10, maxDistance=70)
trackers = []
trackableObjects = {}
props = {}

metric = NearestNeighborDistanceMetric(
        "cosine", 0.2, None)
tracker = Tracker(metric)


def main(args):
    now = get_current_time_as_string()
    if args.videopath is not None:
        input_file_name = Path(args.videopath).name
    else:
        input_file_name = None
    
    output_file_txt = f"output_{now}.txt"
    output_file_video = f"output_{now}.mp4"
    if args.output is not None:
        if input_file_name is not None:
            if ".mp4" in input_file_name:
                output_file_txt = input_file_name.replace(".mp4",f"_{now}.txt")
                output_file_video = input_file_name.replace(".mp4", f"_{now}.mp4")
            elif ".avi" in input_file_name:
                output_file_txt = input_file_name.replace(".avi",f"_{now}.txt")
                output_file_video = input_file_name.replace(".avi", f"_{now}.avi")
            else:
                raise Exception("Input video should in mp4 or avi format.")
            
    # initial for runtime
    providers=[]
    if args.tensorrt:
        providers.append('TensorrtExecutionProvider')
    providers.append('CUDAExecutionProvider')

    # Initialize window.
    cv2.namedWindow(
        WINDOW_NAME, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO
    )
    cv2.moveWindow(WINDOW_NAME, 100, 50)

    # Video capture.
    pipeline = 'v4l2src device=/dev/video1 ! video/x-raw,width=640,height=480! videoconvert ! video/x-raw, format=BGR ! appsink drop=True'
    # pipeline = gstreamer_pipeline(capture_width=640, capture_height=480,display_width=640, display_height=480, flip_method=0)
    # cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if args.videopath is None:
        print("open camera.")
        # cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        cap = cv2.VideoCapture(0)
    else:
        print("open video file", args.videopath)
        cap = cv2.VideoCapture(args.videopath)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Input Video (height, width, fps): ", h, w, fps)

    start = time.time()
    # Yunet model.
    if args.face_detector==1:
        print('Using YuNetTRT as face detector!')
        input_shape=(160,120)
        # input_shape = tuple(map(int, args.input_shape.split(",")))
        # model = TrtYuNet(
        #     model_path="models/yunet/face_detection_yunet_120x160.trt",
        #     input_size=input_shape,
        #     conf_threshold=args.conf_threshold,
        #     nms_threshold=args.nms_threshold,
        #     top_k=args.top_k,
        #     keep_top_k=args.keep_top_k,
        # ) 
        # w_scale = w / input_shape[0]
        # h_scale = h / input_shape[1]
    elif args.face_detector==2:
        print('Using YuNet CV2 as face detector!')
        input_shape=(320, 320)
        model = cv2.FaceDetectorYN.create("models/yunet/face_detection_yunet_2022mar.onnx",
                                          "", 
                                          input_shape,
                                            0.9,0.3,5000,
                                            backend_id=cv2.dnn.DNN_BACKEND_CUDA,
                                            target_id=cv2.dnn.DNN_TARGET_CUDA
                                            )
        w_scale = w / input_shape[0]
        h_scale = h / input_shape[1]
    elif args.face_detector==3:
        print("Using Haar Cascade Classifier as face detector!")
        model = cv2.CascadeClassifier()
        model.load("models/haarcascades/haarcascade_frontalface_alt.xml")
        w_scale = 1.0
        h_scale = 1.0
    else:
        raise Exception("Please choose proper face detector: 1 - YuNetTRT, 2 - YuNet CV2, 3 - Haar Cascade")


    # Face-emotion model
    fer_name = args.fer_model.split("/")[-1]
    if args.fer_flag:
        if args.runtime==1:
            onnx_emot = args.fer_model
            emot_session = onnxruntime.InferenceSession(onnx_emot, providers=providers)
            dummy_input = np.random.rand(1,3, 224,224)
            ort_inputs = {emot_session.get_inputs()[0].name: dummy_input.astype(np.float32)}
            ort_outs = emot_session.run(None, ort_inputs)
        elif args.fer_flag==1 and args.runtime==2:
            cv_model = cv2.dnn.readNetFromONNX(args.fer_model)
            
            cv_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            cv_model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            raise Exception("Please choose correct runtime for application: 1 - onnxruntime, 2 - cv2.dnn.readNetFromONNX")

    print(f'Completed loading facial expression recognition!')

    if args.fer_classnames == 1:
        class_names = ['Anger', 'Contempt','Disgust','Fear','Happiness','Neutral','Sadness','Surprise'] # hsemotion 8 classes
    else:
        class_names = ['Anger','Disgust','Fear','Happiness','Sadness','Surprise','Unknown','Unknown'] # mine 6 classes

    if args.age_gender_flag:
        onnx_ag = 'models/hsemotion/onnx_tf/age_gender_tf2_224_deep-03-0.13-0.97_opset_16.onnx'
        ag_session = onnxruntime.InferenceSession(onnx_ag, providers=providers)
        dummy_input = np.random.rand(1,224,224,3)
        ort_inputs = {ag_session.get_inputs()[0].name: dummy_input.astype(np.float32)}
        ort_outs = ag_session.run(None, ort_inputs)
    end = time.time()

    # face feature extractor
    fe_model = "models/yunet/face_recognition_sface_2021dec.onnx"
    # fe_model = "models/yunet/face_recognition_sface_2021dec_int8.onnx",
    fe_session = onnxruntime.InferenceSession(fe_model, 
                                            #   providers=providers
                                            providers=['CUDAExecutionProvider']
                                              
                                              )

    dummy_input = np.random.rand(1,3,112,112)
    ort_inputs = {fe_session.get_inputs()[0].name: dummy_input.astype(np.float32)}
    ort_outs = fe_session.run(None, ort_inputs)
    print('Total model build time:', end - start)

    # Output Video file
    # Define the codec and create VideoWriter object
    video_writer = None
    if args.output is not None:
        fourcc = cv2.VideoWriter_fourcc(*"MP4V")
        video_writer = cv2.VideoWriter(args.output, fourcc, 12, (w, h))

    elapsed_list = MyQueue(maxsize=100)
    all_fps = MyQueue(maxsize=100)
    emot_ets = MyQueue(maxsize=100)
    age_gender_ets = MyQueue(maxsize=100)
    fe_ets = MyQueue(maxsize=100)
    face_detect_text = ""
    emot_et_text=""
    age_gender_text="Age:"
    fe_text="Feature extract:"
    
    i = -1
    while cap.isOpened():
        i += 1
        ret, frame = cap.read()
        if not ret:
            print("VideoCapture read return false.")
            break
        # print("frame size:", frame.shape, "type:", type(frame))
        # inference.
        fps_start = time.time()

        start = time.perf_counter()
        if args.face_detector==1:
            results = model.infer(frame)
        elif args.face_detector==2:
            img1 = cv2.resize(frame, (320, 320))
            results = model.detect(img1.astype(np.float32))
            if results[1] is not None:
                results = results[1]
            else:
                results = []
        elif args.face_detector==3:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_gray = cv2.equalizeHist(frame_gray)
            results = model.detectMultiScale(frame_gray)
        inference_time = (time.perf_counter() - start) * 1000

        im = frame
        rects = []
        emots = []
        emot_et = 0.0

        for det in results:
            xmin = int(det[0] * w_scale) # left
            xmax = int(det[2] * w_scale) # width
            ymin = int(det[1] * h_scale) # top
            ymax = int(det[3] * h_scale) # height
            roi = im[ymin:ymin+ymax, xmin:xmin+xmax,:]
            roi_w = roi.shape[0]
            roi_h = roi.shape[1]

            if min(roi_w, roi_h)< 10:
                continue

            rects.append((xmin, ymin, xmin + xmax, ymin + ymax))
            
            
            emot = ''
            conf = 0.0
            landmarks = {}
            age_gender = None
            age = ''
            gender = 'unknown'
            face_features=None

            if args.fer_flag and i % args.step == 0 and roi is not None: # and min(roi_w, roi_h)>=10:
                # print("roi shape:", roi.shape)
                start = time.time()
                if args.fer_flag:
                    input_blob = cv2.dnn.blobFromImage(
                        image=roi,
                        scalefactor=scale,
                        size=(224, 224),  # img target size
                        mean=mean,
                        swapRB=True,  # BGR -> RGB
                        crop=True  # center crop
                    ) # (1,3,224,224)
                    input_blob[0] /= np.asarray(std, dtype=np.float32).reshape(3, 1, 1) # (1,3,224,224)
                
                    if args.runtime==1:
                        ort_inputs = {emot_session.get_inputs()[0].name: input_blob}
                        out = emot_session.run(None, ort_inputs)
                    elif args.runtime==2:
                        cv_model.setInput(input_blob)
                        out = cv_model.forward()

                    # print("out:", out)
                    class_id = np.argmax(np.squeeze(out))
                    emot = class_names[class_id]
                    conf = det[-1] if args.face_detector <= 2 else 0.5
                    landmarks = det[4:14].astype(np.int32).reshape((5, 2)) if args.face_detector <= 2 else {}
            
                end = time.time()
                emot_et += end - start
               
            emots.append({
                'emotion':emot, 
                'conf': conf, 
                'landmarks': landmarks,
                'age': age,
                'gender': gender,
                'features': face_features
            })

            if emot_et > 0:
                emot_ets.append(emot_et)
   
        start = time.time()
        trackableObjects, props = ct.update(rects, emots)
        end = time.time()
        centroidtrack_et = end - start

        modified_emots = []
        rects = []
        age_gender_et = 0.0
        fe_et = 0.0
        for k, v in trackableObjects.items():
            # age gender
            age = props[k]['face']['age']
            gender=props[k]['face']['gender']
            face_features=props[k]['face']['features']
            
            if (args.age_gender_flag) and (props[k]['face']['gender'] == 'unknown' or i % 300 == 0):
                start = time.time()
                (xmin, ymin, xmax, ymax) = props[k]['rect']
                roi = im[ymin:ymin+ymax, xmin:xmin+xmax,:]
                roi_w = roi.shape[0]
                roi_h = roi.shape[1]
                if min(roi_w, roi_h)<=10:
                    continue
                roi_tf = cv2.resize(roi,(min(xmax, ymax), min(xmax, ymax)))
                roi_tf = cv2.resize(roi_tf, (224,224))
                roi_tf = roi_tf.astype(np.float32)
                roi_tf[...,0] -= 103.939
                roi_tf[...,1] -= 116.779
                roi_tf[...,2] -= 123.68
                roi_tf = np.expand_dims(roi_tf,0)
                ort_inputs = {ag_session.get_inputs()[0].name: roi_tf}
                age_gender = ag_session.run(None, ort_inputs)
                end = time.time()
                age = np.argmax(age_gender[0])
                gender = GENDER[int(age_gender[1] > 0.5)]
                ct.update_age_gender(k, age, gender)
                age_gender_et += time.time() - start
                age_gender_ets.append(age_gender_et)
                fe_ets.append(fe_et)
            
            if props[k]['face']['features'] is None or (i % 50 == 0):
                # face features
                start = time.time()
                face_align = recognizer.alignCrop(frame, det) # (112, 112, 3)
                face_features = extract_face_features(fe_session, face_align) # (1, 128)

                ct.update_face_features(k, face_features) # (128,)
                fe_et += time.time() - start
        # print('inactives:', ct.inactives.get())
        # print('active objects', ct.objects)
        # print('newIDs:', ct.newIds.get())
        # print('ct:', ct.props)
        ct.reid(cosine_threshold=args.cosine_threshold)


        # Render results
        for k,v in trackableObjects.items():
            (xmin, ymin, xmax, ymax) = props[k]['rect']
            if not ct.isdisappeared(k):
                draw_rectangle(im, (xmin, ymin, xmax-xmin, ymax-ymin))
                emot_caption = "id:{} {:.3f} {}".format(k, props[k]['face']['conf'], props[k]['face']['emotion'])
                ag_caption = "age: {}, gender: {}".format(props[k]['face']['age'], props[k]['face']['gender'])
                draw_caption(im, (xmin, ymin - 15), emot_caption)
                draw_caption(im, (xmin, ymin - 5), ag_caption)
                # for landmark in props[k]['face']['landmarks']:
                #     draw_circle(im, (int(landmark[0] * w_scale), int(landmark[1] * h_scale)))
        

        # Calc fps.
        elapsed_list.append(inference_time)

        if len(elapsed_list) > 0:
            # elapsed_list.pop(0)
            avg_elapsed_ms = np.mean(elapsed_list.get())/1000.0
            face_detect_text = "Face detection: {0:.3f}s".format(avg_elapsed_ms)
	
        if len(emot_ets) > 0:
            avg_emot_et = np.mean(emot_ets.get())
            emot_et_text = "Expression: {0:.3f}s".format(avg_emot_et)

        if len(age_gender_ets) > 0:
            avg_age_gender_et = np.mean(age_gender_ets.get())
            age_gender_text = "Age gender: {0:.3f}s".format(avg_age_gender_et)
        
        if len(fe_ets) > 0:
            avg_fe_et = np.mean(fe_ets.get())
            fe_text = "Feature extract: {0:.3f}s".format(avg_fe_et)

        # Display fps
        fps_text = '' # "Inference: {0:.2f}ms".format(inference_time)
        # FPS = cap.get(cv2.CAP_PROP_FPS)
        fps_end = time.time()
        fps_et = fps_end - fps_start
        all_fps.append(fps_et)
        avg_fps = 1/np.mean(all_fps.get())
        # display_text = model_name + " " + fps_text + avg_text
        display_model_name= f"face detector: {FACEDETECTOR[args.face_detector-1]}, fer: {fer_name}"
        display_text = "fps: {:.1f}, {}, {}".format(avg_fps, face_detect_text, emot_et_text)
        display_text_ag_fe = "{}, {}".format(age_gender_text, fe_text)
        other_text = "runtime: {}, frame step to perform fer: {}".format(RUNTIME[args.runtime-1], args.step)
        draw_caption(im, (10, 30), display_model_name)
        draw_caption(im, (10, 50), display_text)
        draw_caption(im, (10,70), display_text_ag_fe)
        draw_caption(im, (10, 90), other_text)

        # Output video file
        if video_writer is not None:
            video_writer.write(im)

        # Display
        cv2.imshow(WINDOW_NAME, im)
        key = cv2.waitKey(10) & 0xFF
        if key == ord("q") or key == ord("Q"):
            break

    # When everything done, release the window
    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = get_parser()
    print('args:', args)
    main(args)
