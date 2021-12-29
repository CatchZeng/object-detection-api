gen-tfrecord:
	python gen_label_map.py
	
	python generate_tfrecord.py \
	-x images/train \
	-l annotations/label_map.pbtxt \
	-o annotations/train.record

	python generate_tfrecord.py \
	-x images/val \
	-l annotations/label_map.pbtxt \
	-o annotations/val.record
dl-model:
	mkdir -p pre-trained-models; \
	model=ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8; \
	curl -O http://download.tensorflow.org/models/object_detection/tf2/20200711/$$model.tar.gz; \
	tar zxvf $$model.tar.gz; \
	mv -f $$model ./pre-trained-models/; \
	rm -rf $$model $$model.tar.gz;
dl-model-640:
	model=ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8; \
	curl -O http://download.tensorflow.org/models/object_detection/tf2/20200711/$$model.tar.gz; \
	tar zxvf $$model.tar.gz; \
	mv -f $$model ./pre-trained-models/; \
	rm -rf $$model $$model.tar.gz;
train:
	python model_main_tf2.py \
	--model_dir=models/ssd_mobilenet_v2_fpnlite_320x320 \
	--pipeline_config_path=models/ssd_mobilenet_v2_fpnlite_320x320/pipeline.config
train-640:
	python model_main_tf2.py \
	--model_dir=models/ssd_mobilenet_v2_fpnlite_640x640 \
	--pipeline_config_path=models/ssd_mobilenet_v2_fpnlite_640x640/pipeline.config
eval:
	python model_main_tf2.py \
    --pipeline_config_path=models/ssd_mobilenet_v2_fpnlite_320x320/pipeline.config \
    --model_dir=models/ssd_mobilenet_v2_fpnlite_320x320 \
    --checkpoint_dir=models/ssd_mobilenet_v2_fpnlite_320x320 \
    --alsologtostderr
eval-640:
	python model_main_tf2.py \
    --pipeline_config_path=models/ssd_mobilenet_v2_fpnlite_640x640/pipeline.config \
    --model_dir=models/ssd_mobilenet_v2_fpnlite_640x640 \
    --checkpoint_dir=models/ssd_mobilenet_v2_fpnlite_640x640 \
    --alsologtostderr
.PHONY: export
export:
	python exporter_main_v2.py \
	--input_type image_tensor \
	--pipeline_config_path models/ssd_mobilenet_v2_fpnlite_320x320/pipeline.config \
	--trained_checkpoint_dir models/ssd_mobilenet_v2_fpnlite_320x320/ \
	--output_directory exported-models/ssd_mobilenet_v2_fpnlite_320x320
export-640:
	python exporter_main_v2.py \
	--input_type image_tensor \
	--pipeline_config_path models/ssd_mobilenet_v2_fpnlite_640x640/pipeline.config \
	--trained_checkpoint_dir models/ssd_mobilenet_v2_fpnlite_640x640/ \
	--output_directory exported-models/ssd_mobilenet_v2_fpnlite_640x640
export-lite:
	python export_tflite_graph_tf2.py \
	--pipeline_config_path models/ssd_mobilenet_v2_fpnlite_320x320/pipeline.config \
	--trained_checkpoint_dir models/ssd_mobilenet_v2_fpnlite_320x320 \
	--output_directory exported-models/ssd_mobilenet_v2_fpnlite_320x320_lite
export-lite-640:
	python export_tflite_graph_tf2.py \
	--pipeline_config_path models/ssd_mobilenet_v2_fpnlite_640x640/pipeline.config \
	--trained_checkpoint_dir models/ssd_mobilenet_v2_fpnlite_640x640 \
	--output_directory exported-models/ssd_mobilenet_v2_fpnlite_640x640_lite
convert-lite:
	tflite_convert \
	--saved_model_dir=exported-models/ssd_mobilenet_v2_fpnlite_320x320_lite/saved_model \
	--output_file=exported-models/ssd_mobilenet_v2_fpnlite_320x320_lite/detect.tflite
convert-lite-640:
	tflite_convert \
	--saved_model_dir=exported-models/ssd_mobilenet_v2_fpnlite_640x640_lite/saved_model \
	--output_file=exported-models/ssd_mobilenet_v2_fpnlite_640x640_lite/detect.tflite
convert-quant-lite:
	python convert_quant_lite.py \
	--saved_model_dir exported-models/ssd_mobilenet_v2_fpnlite_320x320_lite/saved_model \
	--output_file exported-models/ssd_mobilenet_v2_fpnlite_320x320_quant_lite/detect.tflite
convert-quant-lite-640:
	python convert_quant_lite.py \
	--saved_model_dir exported-models/ssd_mobilenet_v2_fpnlite_640x640_lite/saved_model \
	--output_file exported-models/ssd_mobilenet_v2_fpnlite_640x640_quant_lite/detect.tflite